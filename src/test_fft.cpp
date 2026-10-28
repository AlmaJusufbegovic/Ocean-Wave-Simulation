#include "fft.h"
#include <cstdio>
#include <random>
#include <algorithm>

static float max_err(const std::vector<cd>& a, const std::vector<cd>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

int main() {
    const size_t N = 64;

    // Test 1: FFT delta-impulsa mora dati sve jedinice
    std::vector<cd> a(N, 0.0f);
    a[0] = 1.0f;
    fft(a, false);
    float e1 = max_err(a, std::vector<cd>(N, 1.0f));
    std::printf("Test 1 (delta): greska = %g  %s\n", e1, e1 < 1e-4f ? "PASS" : "FAIL");

    // Test 2: cos(2*pi*3*i/N) mora dati vrhove na k=3 i k=61, svaki velicine N/2 = 32
    std::vector<cd> c(N);
    for (size_t i = 0; i < N; ++i) c[i] = std::cos(2.0f * PI * 3.0f * float(i) / float(N));
    fft(c, false);
    std::printf("Test 2 (cos): vrhovi na indeksima:");
    for (size_t k = 0; k < N; ++k)
        if (std::abs(c[k]) > 1e-3f) std::printf("  k=%zu (|X|=%.2f)", k, std::abs(c[k]));
    std::printf("\n");

    // Test 3 i 4: slucajan ulaz, round-trip i poredjenje sa naivnim DFT-om
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<cd> x(N);
    for (auto& v : x) v = cd(dist(gen), dist(gen));

    std::vector<cd> y = x;
    fft(y, false);
    float e4 = max_err(y, dft_naive(x));
    std::printf("Test 4 (FFT vs naivni DFT): greska = %g  %s\n", e4, e4 < 1e-3f ? "PASS" : "FAIL");

    fft(y, true);
    float e3 = max_err(y, x);
    std::printf("Test 3 (ifft(fft(x)) == x): greska = %g  %s\n", e3, e3 < 1e-4f ? "PASS" : "FAIL");
    return 0;
}