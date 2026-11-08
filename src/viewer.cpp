// 3D prikaz okeana: simulacija se racuna na CPU, crta se u OpenGL-u.
//
// Poboljsanja u odnosu na prvu verziju:
//   * dvije kaskade spektra: kaskada 0 (velike talase, geometrija) i kaskada 1 (sitni detalji,
//     samo nagib, u teksturi) -> efektivna rezolucija mnogo veca od N = 256,
//   * pjena preko Jacobijana (sa postepenim nestajanjem),
//   * pakovani IFFT (3 umjesto 5 transformacija po frejmu),
//   * nebo sa suncem, Fresnel, boja vode zavisna od visine, magla,
//   * visina valova se zadaje preko brzine vjetra (kalibracija amplitude).
//
// Upravljanje:
//   lijevi klik + povlacenje : rotacija kamere        kotacic misa : zoom
//   LIJEVO / DESNO           : brzina vjetra -/+ 1 m/s
//   GORE / DOLJE             : choppiness (lambda) +/- 0.1
//   [  i  ]                  : prag pjene (manje / vise pjene)
//   G                        : pjena on/off            D : sitni detalji (kaskada 1) on/off
//   O                        : OpenMP on/off           F : wireframe        P : pauza     ESC : izlaz

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <omp.h>

#include "ocean.h"

// ---------------- podesavanja ----------------
static constexpr size_t GRID = 256;        // kaskada 0: rezolucija mreze (geometrija)
static constexpr float  L0 = 400.0f;       // kaskada 0: velicina patch-a [m]
static constexpr size_t DETAIL_N = 128;    // kaskada 1: rezolucija (samo nagib u teksturi)
static constexpr float  L1 = 40.0f;        // kaskada 1: velicina patch-a [m]
static constexpr float  K_SPLIT_LO = 0.8f; // granica izmedju kaskada [rad/m]
static constexpr float  K_SPLIT_HI = 1.4f;
static constexpr float  HS_PER_V2 = 0.0214f;  // znacajna visina Hs ~ 0.21 V^2/g (potpuno razvijeno more)

// ---------------- GLSL ----------------
static const char* SKY_FN = R"(
const float FOG = 0.0018;
vec3 skyColor(vec3 d, vec3 sunDir) {
    float h = clamp(d.y, 0.0, 1.0);
    vec3 horizon = vec3(0.72, 0.83, 0.92);
    vec3 zenith  = vec3(0.16, 0.36, 0.72);
    vec3 col = mix(horizon, zenith, pow(h, 0.45));
    float s = max(dot(d, sunDir), 0.0);
    col += vec3(1.0, 0.85, 0.6) * (pow(s, 600.0) * 4.0 + pow(s, 12.0) * 0.18);
    return col;
}
)";

static const char* SKY_VS = R"(#version 430 core
out vec2 vNdc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vNdc = p * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 1.0, 1.0);
}
)";

static const char* SKY_FS_MAIN = R"(
in vec2 vNdc;
uniform mat4 uInvVP;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
out vec4 fragColor;
void main() {
    vec4 p = uInvVP * vec4(vNdc, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / p.w - uCamPos);
    vec3 sun = normalize(uSunDir);
    vec3 col = skyColor(dir, sun);
    if (dir.y < 0.0) {                           // ispod horizonta: daleka voda iza zadnjeg patch-a
        float t = uCamPos.y / max(-dir.y, 1e-4);
        float fres = 0.02 + 0.98 * pow(1.0 - clamp(-dir.y, 0.0, 1.0), 5.0);
        vec3 refl = skyColor(vec3(dir.x, -dir.y, dir.z), sun);
        vec3 water = mix(vec3(0.01, 0.08, 0.14), refl, fres);
        vec3 fogC = skyColor(normalize(vec3(dir.x, 0.05, dir.z)), sun);
        col = mix(water, fogC, 1.0 - exp(-t * FOG));
    }
    fragColor = vec4(col, 1.0);
}
)";

static const char* OCEAN_VS = R"(#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aSlope;
layout(location = 2) in float aFoam;
uniform mat4 uVP;
uniform vec3 uOffset;
out vec3 vWorld;
out vec2 vSlope;
out float vFoam;
out float vHeight;
void main() {
    vec3 w = aPos + uOffset;
    vWorld = w;
    vSlope = aSlope;
    vFoam = aFoam;
    vHeight = aPos.y;
    gl_Position = uVP * vec4(w, 1.0);
}
)";

static const char* OCEAN_FS_MAIN = R"(
in vec3 vWorld;
in vec2 vSlope;
in float vFoam;
in float vHeight;
uniform sampler2D uDetail;
uniform float uDetailL;
uniform int uUseDetail;
uniform int uUseFoam;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform float uHs;
out vec4 fragColor;
void main() {
    vec2 s1 = vec2(0.0);
    if (uUseDetail != 0) s1 = texture(uDetail, vWorld.xz / uDetailL).rg;   // mipmape = antialiasing
    vec3 N = normalize(vec3(-(vSlope.x + s1.x), 1.0, -(vSlope.y + s1.y)));
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);

    float NdV = max(dot(N, V), 0.0);
    float fres = 0.02 + 0.98 * pow(1.0 - NdV, 5.0);              // Schlickova aproksimacija
    vec3 R = reflect(-V, N);
    R.y = abs(R.y);                                              // odraz ne ide ispod horizonta
    vec3 sky = skyColor(R, L);

    float diff = max(dot(N, L), 0.0);
    float crest = smoothstep(-0.5, 1.0, vHeight / max(uHs, 0.1));
    vec3 deep = vec3(0.01, 0.07, 0.13);
    vec3 shallow = vec3(0.02, 0.30, 0.38);
    vec3 water = mix(deep, shallow, clamp(0.15 + 0.35 * diff + 0.5 * crest, 0.0, 1.0));

    float foam = 0.0;
    if (uUseFoam != 0) {
        float breakup = clamp(length(s1) * 6.0, 0.0, 1.0);      // sitni detalji "kidaju" rub pjene
        foam = smoothstep(0.35, 0.65, vFoam + (breakup - 0.5) * 0.5);
    }
    vec3 col = mix(water, sky, fres * (1.0 - foam));
    vec3 foamCol = vec3(0.92, 0.95, 0.97) * (0.45 + 0.55 * diff);
    col = mix(col, foamCol, foam);

    float dist = length(uCamPos - vWorld);
    vec3 fogDir = normalize(vec3(-V.x, 0.05, -V.z));
    col = mix(col, skyColor(fogDir, L), 1.0 - exp(-dist * FOG));
    fragColor = vec4(col, 1.0);
}
)";

static GLuint compileShader(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, 2048, nullptr, log);
        std::printf("Greska u shaderu:\n%s\n", log);
        std::exit(1);
    }
    return s;
}

static GLuint makeProgram(const std::string& vs, const std::string& fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, 2048, nullptr, log);
        std::printf("Greska pri linkanju:\n%s\n", log);
        std::exit(1);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ---------------- stanje kamere i unosa ----------------
static float gYaw = 0.8f, gPitch = 0.35f, gDist = 120.0f;
static bool gDragging = false, gWireframe = false, gPaused = false;
static bool gFoamOn = true, gDetailOn = true;
static double gLastX = 0, gLastY = 0;
static float gLambda = 1.6f;
static float gWind = 14.0f;
static float gFoamBias = 0.78f;
static bool gWindChanged = false;

static void onMouseButton(GLFWwindow* w, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        gDragging = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &gLastX, &gLastY);
    }
}

static void onCursor(GLFWwindow*, double x, double y) {
    if (gDragging) {
        gYaw += float(x - gLastX) * 0.005f;
        gPitch += float(y - gLastY) * 0.005f;
        gPitch = std::clamp(gPitch, 0.03f, 1.5f);
    }
    gLastX = x;
    gLastY = y;
}

static void onScroll(GLFWwindow*, double, double dy) {
    gDist *= std::pow(0.9f, float(dy));
    gDist = std::clamp(gDist, 5.0f, 1500.0f);
}

static void onKey(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_UP) gLambda += 0.1f;
    if (key == GLFW_KEY_DOWN) gLambda = std::max(0.0f, gLambda - 0.1f);
    if (key == GLFW_KEY_LEFT_BRACKET) gFoamBias = std::max(0.3f, gFoamBias - 0.05f);
    if (key == GLFW_KEY_RIGHT_BRACKET) gFoamBias = std::min(1.0f, gFoamBias + 0.05f);
    if (key == GLFW_KEY_RIGHT) { gWind = std::min(25.0f, gWind + 1.0f); gWindChanged = true; }
    if (key == GLFW_KEY_LEFT) { gWind = std::max(5.0f, gWind - 1.0f); gWindChanged = true; }
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_F) gWireframe = !gWireframe;
    if (key == GLFW_KEY_P) gPaused = !gPaused;
    if (key == GLFW_KEY_O) gUseOmp = !gUseOmp;
    if (key == GLFW_KEY_G) gFoamOn = !gFoamOn;
    if (key == GLFW_KEY_D) gDetailOn = !gDetailOn;
}

// Parametri obje kaskade za zadanu brzinu vjetra. Visina valova se kalibrise tako da
// znacajna visina bude Hs = HS_PER_V2 * V^2 (rms visina = Hs / 4).
static void setupCascades(float wind, OceanParams& p0, OceanParams& p1,
    std::vector<cd>& h0a, std::vector<cd>& h0b) {
    p0 = OceanParams();
    p0.N = GRID; p0.L = L0; p0.windSpeed = wind; p0.seed = 42;
    p0.band = 1; p0.splitLo = K_SPLIT_LO; p0.splitHi = K_SPLIT_HI;
    p0.A = amplitudeForRms(p0, HS_PER_V2 * wind * wind / 4.0f);

    p1 = p0;
    p1.N = DETAIL_N; p1.L = L1; p1.seed = 1337; p1.band = 2;
    // Isti fizicki spektar u obje kaskade: varijansa po celiji ~ P(k) * dk^2, dk = 2pi/L,
    // pa se A skalira sa (L0/L1)^2.
    p1.A = p0.A * (L0 / L1) * (L0 / L1);

    h0a = generateH0(p0);
    h0b = generateH0(p1);
}

int main() {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Ocean", nullptr, nullptr);
    if (!win) {
        std::printf("Ne mogu napraviti prozor\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::printf("glewInit nije uspio\n");
        return 1;
    }
    glfwSetMouseButtonCallback(win, onMouseButton);
    glfwSetCursorPosCallback(win, onCursor);
    glfwSetScrollCallback(win, onScroll);
    glfwSetKeyCallback(win, onKey);
    gUseOmp = true;

    const std::string skyFn = SKY_FN;
    GLuint skyProg = makeProgram(SKY_VS, "#version 430 core\n" + skyFn + SKY_FS_MAIN);
    GLuint oceanProg = makeProgram(OCEAN_VS, "#version 430 core\n" + skyFn + OCEAN_FS_MAIN);
    const GLint sInvVP = glGetUniformLocation(skyProg, "uInvVP");
    const GLint sCam = glGetUniformLocation(skyProg, "uCamPos");
    const GLint sSun = glGetUniformLocation(skyProg, "uSunDir");
    const GLint oVP = glGetUniformLocation(oceanProg, "uVP");
    const GLint oOff = glGetUniformLocation(oceanProg, "uOffset");
    const GLint oCam = glGetUniformLocation(oceanProg, "uCamPos");
    const GLint oSun = glGetUniformLocation(oceanProg, "uSunDir");
    const GLint oHs = glGetUniformLocation(oceanProg, "uHs");
    const GLint oDet = glGetUniformLocation(oceanProg, "uDetail");
    const GLint oDetL = glGetUniformLocation(oceanProg, "uDetailL");
    const GLint oUseDet = glGetUniformLocation(oceanProg, "uUseDetail");
    const GLint oUseFoam = glGetUniformLocation(oceanProg, "uUseFoam");

    // ---- simulacija ----
    OceanParams p0, p1;
    std::vector<cd> h0a, h0b;
    setupCascades(gWind, p0, p1, h0a, h0b);
    const size_t N = p0.N;
    const unsigned n = unsigned(N);
    const float dx = p0.L / float(N);
    std::vector<float> foam;

    // ---- mesh (N+1) x (N+1) tjemena; zadnji red/kolona su kopija prvih ----
    const unsigned M = n + 1;
    std::vector<unsigned> idx;
    idx.reserve(N * N * 6);
    for (unsigned r = 0; r < n; ++r)
        for (unsigned c = 0; c < n; ++c) {
            const unsigned i00 = r * M + c, i01 = r * M + c + 1;
            const unsigned i10 = (r + 1) * M + c, i11 = (r + 1) * M + c + 1;
            idx.insert(idx.end(), { i00, i10, i01, i01, i10, i11 });
        }

    // tjeme: pozicija (3) + nagib kaskade 0 (2) + pjena (1) = 6 floatova
    GLuint vao = 0, vbo = 0, ebo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(size_t(M) * M * 6 * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, GLsizeiptr(idx.size() * sizeof(unsigned)), idx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(5 * sizeof(float)));
    std::vector<float> verts(size_t(M) * M * 6);

    GLuint skyVao = 0;
    glGenVertexArrays(1, &skyVao);

    // ---- tekstura nagiba kaskade 1 (RG32F, REPEAT, mipmape) ----
    GLuint detailTex = 0;
    glGenTextures(1, &detailTex);
    glBindTexture(GL_TEXTURE_2D, detailTex);
    int levels = 1;
    for (size_t s = DETAIL_N; s > 1; s >>= 1) ++levels;
    glTexStorage2D(GL_TEXTURE_2D, levels, GL_RG32F, GLsizei(DETAIL_N), GLsizei(DETAIL_N));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    if (GLEW_EXT_texture_filter_anisotropic) glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);

    glEnable(GL_DEPTH_TEST);

    float simTime = 0.0f;
    auto last = clk::now();
    double computeAccum = 0.0, titleTimer = 0.0;
    int frames = 0;

    while (!glfwWindowShouldClose(win)) {
        const auto now = clk::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        const float dtSim = gPaused ? 0.0f : dt;
        simTime += dtSim;

        if (gWindChanged) {                      // promjena vjetra: novi spektri, pjena se brise
            setupCascades(gWind, p0, p1, h0a, h0b);
            foam.assign(N * N, 0.0f);
            gWindChanged = false;
        }

        // ---- 1) simulacija na CPU ----
        const auto c0 = clk::now();
        const OceanFields f = computeFieldsPacked(h0a, p0, simTime);
        updateFoam(f, p0, gLambda, dtSim, gFoamBias, 4.0f, 1.5f, foam);

        const int Ni = int(N);
#pragma omp parallel for if(gUseOmp)
        for (int r = 0; r <= Ni; ++r)
            for (int c = 0; c <= Ni; ++c) {
                const size_t i = (size_t(r) % N) * N + (size_t(c) % N);
                const size_t o = (size_t(r) * M + size_t(c)) * 6;
                verts[o + 0] = float(c) * dx - 0.5f * p0.L + gLambda * f.dispX[i];
                verts[o + 1] = f.height[i];
                verts[o + 2] = float(r) * dx - 0.5f * p0.L + gLambda * f.dispZ[i];
                verts[o + 3] = f.slopeX[i];
                verts[o + 4] = f.slopeZ[i];
                verts[o + 5] = foam[i];
            }
        std::vector<float> detail;
        if (gDetailOn) detail = computeSlopesRG(h0b, p1, simTime);
        const auto c1 = clk::now();
        computeAccum += ms(c1 - c0);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, GLsizeiptr(verts.size() * sizeof(float)), verts.data());
        if (gDetailOn) {
            glBindTexture(GL_TEXTURE_2D, detailTex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GLsizei(DETAIL_N), GLsizei(DETAIL_N), GL_RG, GL_FLOAT, detail.data());
            glGenerateMipmap(GL_TEXTURE_2D);
        }

        // ---- 2) crtanje ----
        int w = 0, h = 0;
        glfwGetFramebufferSize(win, &w, &h);
        if (h == 0) h = 1;
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const glm::vec3 cam = gDist * glm::vec3(std::cos(gPitch) * std::cos(gYaw),
            std::sin(gPitch),
            std::cos(gPitch) * std::sin(gYaw));
        const glm::mat4 view = glm::lookAt(cam, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 proj = glm::perspective(glm::radians(50.0f), float(w) / float(h), 0.5f, 6000.0f);
        const glm::mat4 vp = proj * view;
        const glm::vec3 sun = glm::normalize(glm::vec3(-0.55f, 0.32f, -0.75f));

        // nebo (fullscreen trougao, bez dubine)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(skyProg);
        glUniformMatrix4fv(sInvVP, 1, GL_FALSE, glm::value_ptr(glm::inverse(vp)));
        glUniform3f(sCam, cam.x, cam.y, cam.z);
        glUniform3f(sSun, sun.x, sun.y, sun.z);
        glBindVertexArray(skyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEnable(GL_DEPTH_TEST);

        // okean: 3 x 3 kopije patch-a
        glPolygonMode(GL_FRONT_AND_BACK, gWireframe ? GL_LINE : GL_FILL);
        glUseProgram(oceanProg);
        glUniformMatrix4fv(oVP, 1, GL_FALSE, glm::value_ptr(vp));
        glUniform3f(oCam, cam.x, cam.y, cam.z);
        glUniform3f(oSun, sun.x, sun.y, sun.z);
        glUniform1f(oHs, HS_PER_V2 * gWind * gWind);
        glUniform1f(oDetL, L1);
        glUniform1i(oUseDet, gDetailOn ? 1 : 0);
        glUniform1i(oUseFoam, gFoamOn ? 1 : 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, detailTex);
        glUniform1i(oDet, 0);
        glBindVertexArray(vao);
        for (int tz = -1; tz <= 1; ++tz)
            for (int tx = -1; tx <= 1; ++tx) {
                glUniform3f(oOff, float(tx) * p0.L, 0.0f, float(tz) * p0.L);
                glDrawElements(GL_TRIANGLES, GLsizei(idx.size()), GL_UNSIGNED_INT, nullptr);
            }

        glfwSwapBuffers(win);
        glfwPollEvents();

        // ---- 3) naslov prozora ----
        ++frames;
        titleTimer += dt;
        if (titleTimer >= 0.5) {
            char title[220];
            std::snprintf(title, sizeof(title),
                "Ocean | %zu+%zu | %.0f FPS | CPU %.1f ms | OpenMP %s (%d) | vjetar %.0f m/s | lambda %.1f | pjena %s (prag %.2f) | detalji %s",
                N, DETAIL_N, double(frames) / titleTimer, computeAccum / frames,
                gUseOmp ? "ON" : "OFF", gUseOmp ? omp_get_max_threads() : 1,
                double(gWind), double(gLambda), gFoamOn ? "ON" : "OFF", double(gFoamBias),
                gDetailOn ? "ON" : "OFF");
            glfwSetWindowTitle(win, title);
            titleTimer = 0.0;
            computeAccum = 0.0;
            frames = 0;
        }
    }
    glfwTerminate();
    return 0;
}