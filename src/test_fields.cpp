#include "ocean.h"
#include <cmath>
#include <cstdio>
#include <filesystem>

static void writeBin(const char* path, const std::vector<float>& h) {
    FILE* f = std::fopen(path, "wb");
    std::fwrite(h.data(), sizeof(float), h.size(), f);
    std::fclose(f);
}

static double rms(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += double(x) * x;
    return std::sqrt(s / double(v.size()));
}

int main() {
    OceanParams p;
    const size_t N = p.N;
    auto h0 = generateH0(p);
    OceanFields f = computeFields(h0, p, 0.0f);

    // 1) nagib iz spektra vs konacne razlike nad visinama (kolone = x osa)
    const float dx = p.L / float(N);
    std::vector<float> fd(N * N);
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < N; ++c) {
            const size_t cp = (c + 1) % N, cm = (c + N - 1) % N;
            fd[r * N + c] = (f.height[r * N + cp] - f.height[r * N + cm]) / (2.0f * dx);
        }
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < N * N; ++i) {
        sab += double(f.slopeX[i]) * fd[i];
        saa += double(f.slopeX[i]) * f.slopeX[i];
        sbb += double(fd[i]) * fd[i];
    }
    std::printf("Nagib x: korelacija sa konacnim razlikama = %.4f (treba biti > 0.95)\n",
        sab / std::sqrt(saa * sbb));

    // 2) Parseval: rms(Dx)^2 + rms(Dz)^2 mora biti jednako rms(h)^2
    const double rh = rms(f.height), rx = rms(f.dispX), rz = rms(f.dispZ);
    std::printf("Choppy: (rms Dx^2 + rms Dz^2) / rms h^2 = %.4f (treba biti ~1.0)\n",
        (rx * rx + rz * rz) / (rh * rh));

    std::printf("rms visina = %.3f m, rms |nagib x| = %.4f\n", rh, rms(f.slopeX));

    writeBin("heights_t0.bin", f.height);
    writeBin("slopeX_t0.bin", f.slopeX);
    writeBin("dispX_t0.bin", f.dispX);
    std::printf("Fajlovi su u: %s\n", std::filesystem::current_path().string().c_str());
    return 0;
}