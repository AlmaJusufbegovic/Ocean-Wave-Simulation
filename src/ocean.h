#pragma once
#include "fft.h"
#include <random>

struct OceanParams {
    size_t N = 256;          // rezolucija (stepen dvojke)
    float  L = 250.0f;       // fizicka velicina mreze u metrima
    float  A = 5e-7f;        // amplituda (skalira visinu valova)
    float  windSpeed = 12.0f;
    float  windDirX = 1.0f, windDirZ = 0.0f;
    float  g = 9.81f;
    unsigned seed = 42;
};

// 0..N/2-1 -> 0..N/2-1,  N/2..N-1 -> -N/2..-1
inline int freqIndex(size_t i, size_t N) {
    return i < N / 2 ? int(i) : int(i) - int(N);
}

inline float phillips(float kx, float kz, const OceanParams& p) {
    const float k2 = kx * kx + kz * kz;
    if (k2 < 1e-12f) return 0.0f;                       // k = 0 -> nema vala
    const float k = std::sqrt(k2);
    const float Lw = p.windSpeed * p.windSpeed / p.g;   // najveci val za dati vjetar
    const float wl = std::sqrt(p.windDirX * p.windDirX + p.windDirZ * p.windDirZ);
    const float d = (kx * p.windDirX + kz * p.windDirZ) / (k * wl);   // cos ugla prema vjetru
    const float l = Lw / 1000.0f;                      // gusenje sitnih valova
    return p.A * std::exp(-1.0f / (k2 * Lw * Lw)) / (k2 * k2) * d * d * std::exp(-k2 * l * l);
}

// h0(k) = (1/sqrt2) * (xi_r + i*xi_i) * sqrt(P(k)); poziva se jednom
inline std::vector<cd> generateH0(const OceanParams& p) {
    const size_t N = p.N;
    std::mt19937 gen(p.seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<cd> h0(N * N);
    const float dk = 2.0f * PI / p.L;
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < N; ++c) {
            const float kx = dk * float(freqIndex(c, N));
            const float kz = dk * float(freqIndex(r, N));
            const float amp = std::sqrt(0.5f * phillips(kx, kz, p));
            h0[r * N + c] = cd(gauss(gen), gauss(gen)) * amp;
        }
    return h0;
}

// h(k,t) = h0(k) e^{i w t} + conj(h0(-k)) e^{-i w t};   poziva se svaki frame
inline std::vector<cd> spectrumAtTime(const std::vector<cd>& h0, const OceanParams& p, float t) {
    const size_t N = p.N;
    std::vector<cd> h(N * N);
    const float dk = 2.0f * PI / p.L;
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < N; ++c) {
            const float kx = dk * float(freqIndex(c, N));
            const float kz = dk * float(freqIndex(r, N));
            const float w = std::sqrt(p.g * std::sqrt(kx * kx + kz * kz));   // disperzija
            const size_t rm = (N - r) % N, cm = (N - c) % N;                    // indeks od -k
            const cd e = std::polar(1.0f, w * t);
            h[r * N + c] = h0[r * N + c] * e + std::conj(h0[rm * N + cm]) * std::conj(e);
        }
    return h;
}

// Visine valova u trenutku t. Ako je zadan imagRatio, upisuje omjer
// max|imag| / max|real| (mora biti ~0, to je test simetrije spektra).
inline std::vector<float> computeHeights(const std::vector<cd>& h0, const OceanParams& p,
    float t, float* imagRatio = nullptr) {
    const size_t N = p.N;
    std::vector<cd> h = spectrumAtTime(h0, p, t);
    fft2d(h, N, true);                         // inverzni 2D FFT (dijeli sa N^2)
    std::vector<float> out(N * N);
    float maxRe = 0, maxIm = 0;
    for (size_t i = 0; i < N* N; ++i) {
        const cd v = h[i] * float(N * N);      // vracamo N^2: treba nam suma, ne prosjek
        out[i] = v.real();
        maxRe = std::max(maxRe, std::abs(v.real()));
        maxIm = std::max(maxIm, std::abs(v.imag()));
    }
    if (imagRatio) *imagRatio = maxRe > 0 ? maxIm / maxRe : 0.0f;
    return out;
}

struct OceanFields {
    std::vector<float> height;           // h
    std::vector<float> slopeX, slopeZ;   // dh/dx, dh/dz
    std::vector<float> dispX, dispZ;     // horizontalni (choppy) pomak
};

inline OceanFields computeFields(const std::vector<cd>& h0, const OceanParams& p, float t) {
    const size_t N = p.N;
    const float dk = 2.0f * PI / p.L;
    const std::vector<cd> hk = spectrumAtTime(h0, p, t);

    std::vector<cd> hh = hk;
    std::vector<cd> sx(N * N), sz(N * N), dx(N * N), dz(N * N);
    const cd I(0.0f, 1.0f);
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < N; ++c) {
            const size_t i = r * N + c;
            const float kx = dk * float(freqIndex(c, N));
            const float kz = dk * float(freqIndex(r, N));
            const float k = std::sqrt(kx * kx + kz * kz);
            const cd v = hk[i];
            sx[i] = I * kx * v;
            sz[i] = I * kz * v;
            if (k > 1e-6f) {
                dx[i] = -I * (kx / k) * v;
                dz[i] = -I * (kz / k) * v;
            }
            else {
                dx[i] = 0.0f;
                dz[i] = 0.0f;
            }
        }

    auto toReal = [&](std::vector<cd>& a) {
        fft2d(a, N, true);
        std::vector<float> o(N * N);
        for (size_t i = 0; i < N * N; ++i) o[i] = a[i].real() * float(N * N);
        return o;
        };

    OceanFields f;
    f.height = toReal(hh);
    f.slopeX = toReal(sx);
    f.slopeZ = toReal(sz);
    f.dispX = toReal(dx);
    f.dispZ = toReal(dz);
    return f;
}