#include "fft.h"
#include <cstdio>
#include <random>
#include <algorithm>
#include <chrono>

static float max_err(const std::vector<cd>& a, const std::vector<cd>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

int main() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Test 1: FFT2D vs naivni 2D DFT (N = 16)
    {
        const size_t N = 16;
        std::vector<cd> x(N * N);
        for (auto& v : x) v = cd(dist(gen), dist(gen));
        std::vector<cd> y = x;
        fft2d(y, N, false);
        float e = max_err(y, dft2d_naive(x, N));
        std::printf("Test 1 (fft2d vs naivni): greska = %g  %s\n", e, e < 1e-3f ? "PASS" : "FAIL");

        // Test 2: round-trip
        fft2d(y, N, true);
        float e2 = max_err(y, x);
        std::printf("Test 2 (ifft2d(fft2d(x)) == x): greska = %g  %s\n", e2, e2 < 1e-4f ? "PASS" : "FAIL");
    }

    // Test 3: ravni val cos(2*pi*(2r + 5c)/N) -> vrhovi na (2,5) i (N-2,N-5), svaki N^2/2 = 128
    {
        const size_t N = 16;
        std::vector<cd> w(N * N);
        for (size_t r = 0; r < N; ++r)
            for (size_t c = 0; c < N; ++c)
                w[r * N + c] = std::cos(2.0f * PI * float(2 * r + 5 * c) / float(N));
        fft2d(w, N, false);
        std::printf("Test 3 (ravni val): vrhovi:");
        for (size_t r = 0; r < N; ++r)
            for (size_t c = 0; c < N; ++c)
                if (std::abs(w[r * N + c]) > 1e-2f)
                    std::printf("  (%zu,%zu)=%.1f", r, c, std::abs(w[r * N + c]));
        std::printf("\n");
    }

    // Test 4: sekvencijalno vrijeme (osnova za kasnije poredjenje), medijan od 5 mjerenja
    std::printf("\nTest 4 (vrijeme fft2d, sekvencijalno):\n");
    for (size_t N : {128, 256, 512}) {
        std::vector<cd> x(N * N);
        for (auto& v : x) v = cd(dist(gen), dist(gen));
        std::vector<double> times;
        for (int rep = 0; rep < 5; ++rep) {
            std::vector<cd> y = x;
            auto t0 = std::chrono::high_resolution_clock::now();
            fft2d(y, N, false);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(times.begin(), times.end());
        std::printf("  N = %zu: %.2f ms\n", N, times[times.size() / 2]);
    }
    return 0;
}