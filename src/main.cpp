#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // ---- Test B: OpenMP ----
    std::printf("OpenMP max niti: %d\n", omp_get_max_threads());
    const int N = 50000000;
    std::vector<float> v(N);
    for (int threads : {1, 2, 4, 8}) {
        omp_set_num_threads(threads);
        auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < N; ++i) v[i] = std::sin(i * 0.001f);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("%d niti: %.1f ms\n", threads, ms);
    }

    // ---- Test A: OpenGL prozor ----
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(800, 600, "ocean", nullptr, nullptr);
    if (!win) { std::printf("Ne mogu napraviti prozor\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::printf("glewInit nije uspio\n"); return 1; }
    std::printf("GL: %s\nGPU: %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    // ---- Test C: compute shader ----
    const char* src = R"(#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Data { float d[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    d[i] += 1.0;
}
)";
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, &src, nullptr);
    glCompileShader(cs);
    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(cs, 1024, nullptr, log);
        std::printf("Shader greska: %s\n", log);
        return 1;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);

    std::vector<float> data(1024);
    for (int i = 0; i < 1024; ++i) data[i] = float(i);
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glUseProgram(prog);
    glDispatchCompute(1024 / 64, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.size() * sizeof(float), data.data());

    bool good = true;
    for (int i = 0; i < 1024; ++i)
        if (data[i] != float(i) + 1.0f) good = false;
    std::printf("Compute test: %s\n", good ? "OK" : "GRESKA");

    while (!glfwWindowShouldClose(win)) {
        glClearColor(0.1f, 0.3f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}