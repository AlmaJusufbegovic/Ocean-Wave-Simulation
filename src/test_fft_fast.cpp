#include "ocean.h"
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <random>
static double relDiff(const std::vector<cd>& a, const std::vector<cd>& b) {
    double m = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, double(std::abs(a[i] - b[i]))); r = std::max(r, double(std::abs(a[i]))); }
    return m / r;
}
static double relDiffF(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, double(std::fabs(a[i] - b[i]))); r = std::max(r, double(std::fabs(a[i]))); }
    return m / r;
}
int main() {
    std::mt19937 g(7); std::normal_distribution<float> d;
    // 1) 1D vs naivni DFT
    {
        const size_t n = 64; std::vector<cd> x(n); for (auto& v : x) v = cd(d(g), d(g));
        std::vector<cd> y = x; fftPlanned(y.data(), makeFftPlan(n), false);
        std::printf("1D fftPlanned vs naivni DFT: %.2e\n", relDiff(y, dft_naive(x)));
        std::vector<cd> z = y; fftPlanned(z.data(), makeFftPlan(n), true); for (auto& v : z) v /= float(n);
        std::printf("1D round-trip: %.2e\n", relDiff(z, x));
    }
    // 2) 2D fast vs stara
    for (size_t N : {16, 64, 256}) {
        std::vector<cd> x(N * N); for (auto& v : x) v = cd(d(g), d(g));
        std::vector<cd> a = x, b = x; fft2d(a, N, false); fft2d_fast(b, N, false, false);
        std::vector<cd> c = x, e = x; fft2d(c, N, true); fft2d_fast(e, N, true, true);
        std::printf("N=%zu: forward fast vs stara %.2e | inverse(normalizirano) %.2e\n", N, relDiff(a, b), relDiff(c, e));
    }
    {
        const size_t N = 8; std::vector<cd> x(N * N); for (auto& v : x) v = cd(d(g), d(g));
        std::vector<cd> a = x; fft2d_fast(a, N, false, false); std::printf("N=8 vs naivni 2D DFT: %.2e\n", relDiff(a, dft2d_naive(x, N)));
    }
    {
        const size_t N = 4; std::vector<cd> x(N * N); for (auto& v : x) v = cd(d(g), d(g));
        std::vector<cd> a = x; fft2d_fast(a, N, false, false); std::printf("N=4 (blok < 8) vs naivni 2D DFT: %.2e\n", relDiff(a, dft2d_naive(x, N)));
    }
    // 3) polja: gFastFft true vs false, i OpenMP vs seq
    OceanParams p; p.N = 256; p.L = 400; p.windSpeed = 14; p.A = 2e-7f;
    auto h0 = generateH0(p);
    gUseOmp = false; gFastFft = false; auto o1 = computeFieldsPacked(h0, p, 3.0f); auto o5 = computeFields(h0, p, 3.0f);
    gFastFft = true;  auto n1 = computeFieldsPacked(h0, p, 3.0f); auto n5 = computeFields(h0, p, 3.0f);
    std::printf("polja (3 IFFT) nova vs stara: h=%.1e sx=%.1e sz=%.1e Dx=%.1e Dz=%.1e\n", relDiffF(o1.height, n1.height),
        relDiffF(o1.slopeX, n1.slopeX), relDiffF(o1.slopeZ, n1.slopeZ), relDiffF(o1.dispX, n1.dispX), relDiffF(o1.dispZ, n1.dispZ));
    std::printf("polja (5 IFFT) nova vs stara: h=%.1e sx=%.1e Dx=%.1e\n", relDiffF(o5.height, n5.height), relDiffF(o5.slopeX, n5.slopeX), relDiffF(o5.dispX, n5.dispX));
    gUseOmp = true; omp_set_num_threads(4); auto n1p = computeFieldsPacked(h0, p, 3.0f);
    std::printf("nova FFT: OpenMP naspram sekvencijalno: %g\n", std::max(relDiffF(n1.height, n1p.height), relDiffF(n1.dispX, n1p.dispX)));
    // 4) vrijeme, jedna nit
    gUseOmp = false;
    using clk = std::chrono::high_resolution_clock;
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    for (size_t N : {128, 256, 512}) {
        std::vector<cd> x(N * N); for (auto& v : x) v = cd(d(g), d(g));
        std::vector<double> t0, t1;
        for (int r = 0; r < 12; ++r) {
            auto a = x; auto s = clk::now(); fft2d(a, N, true); auto e = clk::now(); t0.push_back(std::chrono::duration<double, std::milli>(e - s).count());
            auto b = x; s = clk::now(); fft2d_fast(b, N, true, false); e = clk::now(); t1.push_back(std::chrono::duration<double, std::milli>(e - s).count());
        }
        std::printf("fft2d 1 nit N=%zu: stara %.2f ms, nova %.2f ms (%.2fx)\n", N, med(t0), med(t1), med(t0) / med(t1));
    }
}