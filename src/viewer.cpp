// 3D prikaz okeana: simulacija se racuna na CPU (sekvencijalno), crta se u OpenGL-u.
// Upravljanje:
//   lijevi klik + povlacenje : rotacija kamere
//   kotacic misa             : zoom
//   strelice GORE / DOLJE    : choppiness (lambda) +/- 0.1
//   F                        : zicani prikaz (wireframe) on/off
//   P                        : pauza
//   ESC                      : izlaz

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
#include <vector>

#include "ocean.h"

// ---------------- podesavanja ----------------
static constexpr size_t GRID = 256;   // rezolucija simulacije (stepen dvojke): 128 ili 256

// ---------------- shaderi ----------------
static const char* VS = R"(#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uVP;
uniform vec3 uOffset;
out vec3 vNormal;
out vec3 vWorld;
void main() {
    vec3 world = aPos + uOffset;
    vWorld = world;
    vNormal = aNormal;
    gl_Position = uVP * vec4(world, 1.0);
}
)";

static const char* FS = R"(#version 430 core
in vec3 vNormal;
in vec3 vWorld;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
out vec4 fragColor;
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    vec3 H = normalize(L + V);

    vec3 horizon = vec3(0.70, 0.82, 0.92);
    vec3 zenith  = vec3(0.20, 0.40, 0.75);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 200.0);
    float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);

    vec3 water = mix(vec3(0.00, 0.08, 0.18), vec3(0.00, 0.32, 0.42), 0.25 + 0.5 * diff);
    vec3 R = reflect(-V, N);
    vec3 sky = mix(horizon, zenith, clamp(R.y, 0.0, 1.0));

    vec3 col = mix(water, sky, fres) + spec * vec3(1.0, 0.95, 0.85);

    float dist = length(uCamPos - vWorld);
    col = mix(col, horizon, 1.0 - exp(-dist * 0.0012));
    fragColor = vec4(col, 1.0);
}
)";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
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

static GLuint makeProgram(const char* vs, const char* fs) {
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
static double gLastX = 0, gLastY = 0;
static float gLambda = 1.0f;

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
        gPitch = std::clamp(gPitch, 0.05f, 1.5f);
    }
    gLastX = x;
    gLastY = y;
}

static void onScroll(GLFWwindow*, double, double dy) {
    gDist *= std::pow(0.9f, float(dy));
    gDist = std::clamp(gDist, 10.0f, 1500.0f);
}

static void onKey(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_F) gWireframe = !gWireframe;
    if (key == GLFW_KEY_P) gPaused = !gPaused;
    if (key == GLFW_KEY_UP) gLambda += 0.1f;
    if (key == GLFW_KEY_DOWN) gLambda = std::max(0.0f, gLambda - 0.1f);
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
    glfwSwapInterval(0);   // bez vsync-a, da se vidi stvarni FPS
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::printf("glewInit nije uspio\n");
        return 1;
    }
    glfwSetMouseButtonCallback(win, onMouseButton);
    glfwSetCursorPosCallback(win, onCursor);
    glfwSetScrollCallback(win, onScroll);
    glfwSetKeyCallback(win, onKey);

    GLuint prog = makeProgram(VS, FS);
    const GLint locVP = glGetUniformLocation(prog, "uVP");
    const GLint locOff = glGetUniformLocation(prog, "uOffset");
    const GLint locCam = glGetUniformLocation(prog, "uCamPos");
    const GLint locSun = glGetUniformLocation(prog, "uSunDir");

    // ---- simulacija ----
    OceanParams p;
    p.N = GRID;
    const size_t N = p.N;
    const unsigned n = unsigned(N);
    const std::vector<cd> h0 = generateH0(p);
    const float dx = p.L / float(N);

    // ---- mesh: (N+1) x (N+1) tjemena; zadnji red/kolona su kopija prvih (periodicnost),
    //      pa se susjedni patch-evi spajaju bez sava ----
    const unsigned M = n + 1;
    std::vector<unsigned> idx;
    idx.reserve(N * N * 6);
    for (unsigned r = 0; r < n; ++r)
        for (unsigned c = 0; c < n; ++c) {
            const unsigned i00 = r * M + c, i01 = r * M + c + 1;
            const unsigned i10 = (r + 1) * M + c, i11 = (r + 1) * M + c + 1;
            idx.insert(idx.end(), { i00, i10, i01, i01, i10, i11 });
        }

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
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    std::vector<float> verts(size_t(M) * M * 6);
    glEnable(GL_DEPTH_TEST);

    float simTime = 0.0f;
    auto last = clk::now();
    double computeAccum = 0.0, titleTimer = 0.0;
    int frames = 0;

    while (!glfwWindowShouldClose(win)) {
        const auto now = clk::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        if (!gPaused) simTime += dt;

        // ---- 1) simulacija na CPU + priprema tjemena ----
        const auto c0 = clk::now();
        const OceanFields f = computeFields(h0, p, simTime);
        for (size_t r = 0; r <= N; ++r)
            for (size_t c = 0; c <= N; ++c) {
                const size_t i = (r % N) * N + (c % N);      // podaci simulacije (periodicni)
                const size_t o = (r * M + c) * 6;            // pozicija u nizu tjemena
                const float x = float(c) * dx - 0.5f * p.L + gLambda * f.dispX[i];
                const float z = float(r) * dx - 0.5f * p.L + gLambda * f.dispZ[i];
                const glm::vec3 nrm = glm::normalize(glm::vec3(-f.slopeX[i], 1.0f, -f.slopeZ[i]));
                verts[o + 0] = x;  verts[o + 1] = f.height[i];  verts[o + 2] = z;
                verts[o + 3] = nrm.x;  verts[o + 4] = nrm.y;  verts[o + 5] = nrm.z;
            }
        const auto c1 = clk::now();
        computeAccum += ms(c1 - c0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, GLsizeiptr(verts.size() * sizeof(float)), verts.data());

        // ---- 2) crtanje ----
        int w = 0, h = 0;
        glfwGetFramebufferSize(win, &w, &h);
        if (h == 0) h = 1;
        glViewport(0, 0, w, h);
        glClearColor(0.70f, 0.82f, 0.92f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, gWireframe ? GL_LINE : GL_FILL);

        const glm::vec3 cam = gDist * glm::vec3(std::cos(gPitch) * std::cos(gYaw),
            std::sin(gPitch),
            std::cos(gPitch) * std::sin(gYaw));
        const glm::mat4 view = glm::lookAt(cam, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 proj = glm::perspective(glm::radians(50.0f), float(w) / float(h), 0.5f, 6000.0f);
        const glm::mat4 vp = proj * view;

        glUseProgram(prog);
        glUniformMatrix4fv(locVP, 1, GL_FALSE, glm::value_ptr(vp));
        glUniform3f(locCam, cam.x, cam.y, cam.z);
        glUniform3f(locSun, -0.6f, 0.45f, -0.62f);
        glBindVertexArray(vao);
        for (int tz = -1; tz <= 1; ++tz)          // 3 x 3 kopije istog patch-a
            for (int tx = -1; tx <= 1; ++tx) {
                glUniform3f(locOff, float(tx) * p.L, 0.0f, float(tz) * p.L);
                glDrawElements(GL_TRIANGLES, GLsizei(idx.size()), GL_UNSIGNED_INT, nullptr);
            }

        glfwSwapBuffers(win);
        glfwPollEvents();

        // ---- 3) naslov prozora: FPS i vrijeme simulacije ----
        ++frames;
        titleTimer += dt;
        if (titleTimer >= 0.5) {
            char title[160];
            std::snprintf(title, sizeof(title),
                "Ocean | N=%zu | %.0f FPS | simulacija na CPU: %.1f ms/frame | lambda=%.1f",
                N, double(frames) / titleTimer, computeAccum / frames, double(gLambda));
            glfwSetWindowTitle(win, title);
            titleTimer = 0.0;
            computeAccum = 0.0;
            frames = 0;
        }
    }
    glfwTerminate();
    return 0;
}