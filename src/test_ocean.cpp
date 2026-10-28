#include "ocean.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>

static void writeBin(const char* path, const std::vector<float>& h) {
    FILE* f = std::fopen(path, "wb");
    std::fwrite(h.data(), sizeof(float), h.size(), f);
    std::fclose(f);
}

static void stats(const char* name, const std::vector<float>& h, float imagRatio) {
    auto mm = std::minmax_element(h.begin(), h.end());
    double sum = 0;
    for (float v : h) sum += v;
    std::printf("%s: min=%.3f  max=%.3f  mean=%.4f  imag/real=%.2e\n",
        name, *mm.first, *mm.second, sum / double(h.size()), imagRatio);
}

int main() {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };

    // 1) visine za t = 0 i t = 5 s
    OceanParams p;
    auto h0 = generateH0(p);
    float im0, im5;
    auto ht0 = computeHeights(h0, p, 0.0f, &im0);
    auto ht5 = computeHeights(h0, p, 5.0f, &im5);
    stats("t=0", ht0, im0);
    stats("t=5", ht5, im5);
    writeBin("heights_t0.bin", ht0);
    writeBin("heights_t5.bin", ht5);
    std::printf("Fajlovi su u: %s\n\n", std::filesystem::current_path().string().c_str());

    // 2) sekvencijalna mjerenja po koracima (medijan od 5)
    std::printf("Vrijeme po koracima (sekvencijalno):\n");
    for (size_t N : {128, 256, 512}) {
        OceanParams q;
        q.N = N;
        auto h0q = generateH0(q);
        std::vector<double> tSpec, tFFT;
        for (int rep = 0; rep < 5; ++rep) {
            auto a = clk::now();
            auto s = spectrumAtTime(h0q, q, 1.0f);
            auto b = clk::now();
            fft2d(s, N, true);
            auto c = clk::now();
            tSpec.push_back(ms(b - a));
            tFFT.push_back(ms(c - b));
        }
        std::printf("  N=%zu: spektar %.2f ms | ifft2d %.2f ms\n", N, median(tSpec), median(tFFT));
    }
    return 0;
}