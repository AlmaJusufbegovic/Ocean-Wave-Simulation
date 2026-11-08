#pragma once
#include "fft.h"
#include <chrono>
#include <random>

struct OceanParams {
    size_t N = 256;          // rezolucija (stepen dvojke)
    float  L = 250.0f;       // fizicka velicina mreze u metrima
    float  A = 5e-7f;        // amplituda (skalira visinu valova)
    float  windSpeed = 12.0f;
    float  windDirX = 1.0f, windDirZ = 0.0f;
    float  g = 9.81f;
    unsigned seed = 42;
    // Frekventni pojas (kaskade): band 0 = cijeli spektar, 1 = niske frekvencije (k < split),
    // 2 = visoke frekvencije (k > split). Prijelaz je gladak (cos/sin), pa je zbir snaga
    // dvije kaskade tacno jednak snazi originalnog spektra.
    int   band = 0;
    float splitLo = 0.0f, splitHi = 0.0f;   // rad/m
};

// 0..N/2-1 -> 0..N/2-1,  N/2..N-1 -> -N/2..-1
inline int freqIndex(size_t i, size_t N) {
    return i < N / 2 ? int(i) : int(i) - int(N);
}

inline float phillips(float kx, float kz, const OceanParams& p) {
    const float k2 = kx * kx + kz * kz;
    if (k2 < 1e-12f) return 0.0f;
    const float k = std::sqrt(k2);
    const float Lw = p.windSpeed * p.windSpeed / p.g;
    const float wl = std::sqrt(p.windDirX * p.windDirX + p.windDirZ * p.windDirZ);
    const float d = (kx * p.windDirX + kz * p.windDirZ) / (k * wl);
    const float l = Lw / 1000.0f;
    return p.A * std::exp(-1.0f / (k2 * Lw * Lw)) / (k2 * k2) * d * d * std::exp(-k2 * l * l);
}

// Tezina amplitude za pojas kaskade (w0^2 + w1^2 = 1).
inline float bandWeight(float k, const OceanParams& p) {
    if (p.band == 0) return 1.0f;
    float t = (k - p.splitLo) / std::max(p.splitHi - p.splitLo, 1e-6f);
    t = std::min(1.0f, std::max(0.0f, t));
    t = t * t * (3.0f - 2.0f * t);                       // smoothstep
    return p.band == 1 ? std::cos(0.5f * PI * t) : std::sin(0.5f * PI * t);
}

// h0(k): poziva se jednom. Ostaje SEKVENCIJALNO (generator slucajnih brojeva
// daje reproducibilan niz samo kad se poziva redom).
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
            const float kk = std::sqrt(kx * kx + kz * kz);
            const float amp = std::sqrt(0.5f * phillips(kx, kz, p)) * bandWeight(kk, p);
            h0[r * N + c] = cd(gauss(gen), gauss(gen)) * amp;
        }
    return h0;
}

// h(k,t) = h0(k) e^{i w t} + conj(h0(-k)) e^{-i w t}   (svaki element je nezavisan)
inline std::vector<cd> spectrumAtTime(const std::vector<cd>& h0, const OceanParams& p, float t) {
    const size_t N = p.N;
    const int n = int(N);
    std::vector<cd> h(N * N);
    const float dk = 2.0f * PI / p.L;
#pragma omp parallel for if(gUseOmp)
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            const float kx = dk * float(freqIndex(size_t(c), N));
            const float kz = dk * float(freqIndex(size_t(r), N));
            const float w = std::sqrt(p.g * std::sqrt(kx * kx + kz * kz));
            const size_t rm = (N - size_t(r)) % N, cm = (N - size_t(c)) % N;
            const cd e = std::polar(1.0f, w * t);
            h[size_t(r) * N + size_t(c)] =
                h0[size_t(r) * N + size_t(c)] * e + std::conj(h0[rm * N + cm]) * std::conj(e);
        }
    return h;
}

// Samo visine (koristi test_ocean). imagRatio = max|imag| / max|real| (test simetrije).
inline std::vector<float> computeHeights(const std::vector<cd>& h0, const OceanParams& p,
    float t, float* imagRatio = nullptr) {
    const size_t N = p.N;
    std::vector<cd> h = spectrumAtTime(h0, p, t);
    fft2d(h, N, true);
    std::vector<float> out(N * N);
    float maxRe = 0, maxIm = 0;
    for (size_t i = 0; i < N* N; ++i) {
        const cd v = h[i] * float(N * N);
        out[i] = v.real();
        maxRe = std::max(maxRe, std::abs(v.real()));
        maxIm = std::max(maxIm, std::abs(v.imag()));
    }
    if (imagRatio) *imagRatio = maxRe > 0 ? maxIm / maxRe : 0.0f;
    return out;
}

struct OceanFields {
    std::vector<float> height;
    std::vector<float> slopeX, slopeZ;
    std::vector<float> dispX, dispZ;
};

// Vremena (ms) po koracima jednog framea, za profilisanje.
struct FrameTimes {
    double spectrum = 0;   // h(k,t)
    double prep = 0;   // priprema spektara nagiba i pomaka
    double ifft = 0;   // 5 x inverzni 2D FFT
    double extract = 0;   // realni dio * N^2
};

inline OceanFields computeFields(const std::vector<cd>& h0, const OceanParams& p, float t,
    FrameTimes* tm = nullptr) {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
        };

    const size_t N = p.N;
    const int n = int(N);
    const float dk = 2.0f * PI / p.L;

    // 1) spektar u trenutku t
    const auto t0 = clk::now();
    const std::vector<cd> hk = spectrumAtTime(h0, p, t);
    const auto t1 = clk::now();

    // 2) priprema spektara polja
    std::vector<cd> hh = hk;
    std::vector<cd> sx(N * N), sz(N * N), dx(N * N), dz(N * N);
    const cd I(0.0f, 1.0f);
#pragma omp parallel for if(gUseOmp)
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            const size_t i = size_t(r) * N + size_t(c);
            const float kx = dk * float(freqIndex(size_t(c), N));
            const float kz = dk * float(freqIndex(size_t(r), N));
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
    const auto t2 = clk::now();

    // 3) pet inverznih 2D FFT-ova (svaki je interno paralelan)
    fft2d(hh, N, true);
    fft2d(sx, N, true);
    fft2d(sz, N, true);
    fft2d(dx, N, true);
    fft2d(dz, N, true);
    const auto t3 = clk::now();

    // 4) realni dio, pomnozen sa N^2
    auto toReal = [&](const std::vector<cd>& a) {
        std::vector<float> o(N * N);
        const int total = int(N * N);
        const float s = float(N * N);
#pragma omp parallel for if(gUseOmp)
        for (int i = 0; i < total; ++i) o[size_t(i)] = a[size_t(i)].real() * s;
        return o;
        };
    OceanFields f;
    f.height = toReal(hh);
    f.slopeX = toReal(sx);
    f.slopeZ = toReal(sz);
    f.dispX = toReal(dx);
    f.dispZ = toReal(dz);
    const auto t4 = clk::now();

    if (tm) {
        tm->spectrum += ms(t0, t1);
        tm->prep += ms(t1, t2);
        tm->ifft += ms(t2, t3);
        tm->extract += ms(t3, t4);
    }
    return f;
}

// ---------------------------------------------------------------------------
// Kalibracija amplitude: visina valova se zadaje preko rms visine (sigma).
// Za realizaciju iz generateH0 vrijedi E|h(k,t)|^2 = 2 P(k) w(k)^2, pa je
// varijansa visine u tacki: sigma^2 = 2 * sum_k P(k) w(k)^2.
// ---------------------------------------------------------------------------
inline double spectrumVariance(const OceanParams& p) {
    const size_t N = p.N;
    const float dk = 2.0f * PI / p.L;
    double sum = 0.0;
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < N; ++c) {
            const float kx = dk * float(freqIndex(c, N));
            const float kz = dk * float(freqIndex(r, N));
            const float k = std::sqrt(kx * kx + kz * kz);
            const float w = bandWeight(k, p);
            sum += double(phillips(kx, kz, p)) * double(w) * double(w);
        }
    return 2.0 * sum;
}

// A za koju je rms visina (kaskade p) jednaka targetRms.
inline float amplitudeForRms(OceanParams p, float targetRms) {
    p.A = 1.0f;
    const double v = spectrumVariance(p);
    return v > 0 ? float(double(targetRms) * double(targetRms) / v) : 1.0f;
}

// ---------------------------------------------------------------------------
// Pakovani IFFT: dva REALNA polja se sakriju u jednu kompleksnu transformaciju.
// IFFT(A + i B) = a + i b, jer su spektri A i B Hermitovski simetricni.
// Umjesto 5 IFFT-ova po frejmu potrebna su 3: h, (nagibX + i nagibZ), (Dx + i Dz).
// Nyquistove frekvencije se u operatorima izvoda/pomaka postavljaju na 0 (nisu
// Hermitovski simetricne pa bi "curile" u drugo polje).
// ---------------------------------------------------------------------------
inline OceanFields computeFieldsPacked(const std::vector<cd>& h0, const OceanParams& p, float t,
    FrameTimes* tm = nullptr) {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
        };
    const size_t N = p.N;
    const int n = int(N);
    const float dk = 2.0f * PI / p.L;

    const auto t0 = clk::now();
    const std::vector<cd> hk = spectrumAtTime(h0, p, t);
    const auto t1 = clk::now();

    std::vector<cd> hh = hk, sp(N * N), dp(N * N);
    const cd I(0.0f, 1.0f);
#pragma omp parallel for if(gUseOmp)
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            const size_t i = size_t(r) * N + size_t(c);
            const float kx = dk * float(freqIndex(size_t(c), N));
            const float kz = dk * float(freqIndex(size_t(r), N));
            const float k = std::sqrt(kx * kx + kz * kz);
            const float kxd = (size_t(c) == N / 2) ? 0.0f : kx;
            const float kzd = (size_t(r) == N / 2) ? 0.0f : kz;
            const cd v = hk[i];
            sp[i] = (I * kxd - kzd) * v;                        // (i kx h) + i (i kz h)
            dp[i] = (k > 1e-6f) ? (kzd / k - I * (kxd / k)) * v : cd(0.0f, 0.0f);  // Dx + i Dz
        }
    const auto t2 = clk::now();

    fft2d(hh, N, true);
    fft2d(sp, N, true);
    fft2d(dp, N, true);
    const auto t3 = clk::now();

    OceanFields f;
    const int total = int(N * N);
    const float s = float(N * N);
    f.height.resize(N * N); f.slopeX.resize(N * N); f.slopeZ.resize(N * N);
    f.dispX.resize(N * N);  f.dispZ.resize(N * N);
#pragma omp parallel for if(gUseOmp)
    for (int i = 0; i < total; ++i) {
        const size_t u = size_t(i);
        f.height[u] = hh[u].real() * s;
        f.slopeX[u] = sp[u].real() * s;
        f.slopeZ[u] = sp[u].imag() * s;
        f.dispX[u] = dp[u].real() * s;
        f.dispZ[u] = dp[u].imag() * s;
    }
    const auto t4 = clk::now();
    if (tm) {
        tm->spectrum += ms(t0, t1);
        tm->prep += ms(t1, t2);
        tm->ifft += ms(t2, t3);
        tm->extract += ms(t3, t4);
    }
    return f;
}

// Samo nagibi, jedna IFFT; rezultat je interleaved (nagibX, nagibZ) po tacki, spreman za
// RG32F teksturu (kaskada sa sitnim detaljima).
inline std::vector<float> computeSlopesRG(const std::vector<cd>& h0, const OceanParams& p, float t) {
    const size_t N = p.N;
    const int n = int(N);
    const float dk = 2.0f * PI / p.L;
    const std::vector<cd> hk = spectrumAtTime(h0, p, t);
    std::vector<cd> sp(N * N);
    const cd I(0.0f, 1.0f);
#pragma omp parallel for if(gUseOmp)
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            const size_t i = size_t(r) * N + size_t(c);
            const float kxd = (size_t(c) == N / 2) ? 0.0f : dk * float(freqIndex(size_t(c), N));
            const float kzd = (size_t(r) == N / 2) ? 0.0f : dk * float(freqIndex(size_t(r), N));
            sp[i] = (I * kxd - kzd) * hk[i];
        }
    fft2d(sp, N, true);
    std::vector<float> rg(N * N * 2);
    const float s = float(N * N);
    for (size_t i = 0; i < N * N; ++i) {
        rg[2 * i] = sp[i].real() * s;
        rg[2 * i + 1] = sp[i].imag() * s;
    }
    return rg;
}

// ---------------------------------------------------------------------------
// Pjena preko Jacobijana (Tessendorf, jed. 45). x' = x + lambda D; J = det(dx'/dx).
// J < 1: povrsina se "skuplja", J < 0: preklapa se (talas se lomi). Pjena nastaje gdje je
// J ispod praga (bias), a zatim polako nestaje (vremenska konstanta tau, u sekundama).
// foam se cuva po tacki mreze (pomjera se zajedno sa tjemenom).
// ---------------------------------------------------------------------------
inline void updateFoam(const OceanFields& f, const OceanParams& p, float lambda, float dt,
    float bias, float gain, float tau, std::vector<float>& foam) {
    const size_t N = p.N;
    const int n = int(N);
    if (foam.size() != N * N) foam.assign(N * N, 0.0f);
    const float inv2dx = 1.0f / (2.0f * p.L / float(N));
    const float decay = tau > 0.0f ? std::exp(-dt / tau) : 0.0f;
#pragma omp parallel for if(gUseOmp)
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            const size_t rp = size_t((r + 1) % n), rm = size_t((r + n - 1) % n);
            const size_t cp = size_t((c + 1) % n), cm = size_t((c + n - 1) % n);
            const size_t ru = size_t(r), cu = size_t(c);
            const float Dxx = (f.dispX[ru * N + cp] - f.dispX[ru * N + cm]) * inv2dx;
            const float Dxz = (f.dispX[rp * N + cu] - f.dispX[rm * N + cu]) * inv2dx;
            const float Dzx = (f.dispZ[ru * N + cp] - f.dispZ[ru * N + cm]) * inv2dx;
            const float Dzz = (f.dispZ[rp * N + cu] - f.dispZ[rm * N + cu]) * inv2dx;
            const float J = (1.0f + lambda * Dxx) * (1.0f + lambda * Dzz) - lambda * lambda * Dxz * Dzx;
            const float inst = std::min(1.0f, std::max(0.0f, (bias - J) * gain));
            float& fo = foam[ru * N + cu];
            fo = std::max(fo * decay, inst);
        }
}