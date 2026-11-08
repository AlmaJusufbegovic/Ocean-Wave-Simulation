#pragma once
#include <complex>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>

using cd = std::complex<float>;
constexpr float PI = 3.14159265358979f;

// Globalni prekidac: true = OpenMP paralelizacija ukljucena, false = sekvencijalno.
// (omogucava poredjenje obje verzije u ISTOM programu)
inline bool gUseOmp = false;

// In-place radix-2 FFT jedne linije. a.size() mora biti stepen dvojke.
// inverse = true: inverzna transformacija, sa dijeljenjem sa N.
inline void fft(std::vector<cd>& a, bool inverse) {
    const size_t n = a.size();

    // 1) bit-reversal permutacija
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // 2) butterfly prolazi
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = 2.0f * PI / float(len) * (inverse ? 1.0f : -1.0f);
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; ++k) {
                cd w = std::polar(1.0f, ang * float(k));
                cd u = a[i + k];
                cd v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }

    if (inverse)
        for (auto& x : a) x /= float(n);
}

// Spori referentni DFT (N^2), u double preciznosti, samo za testiranje.
inline std::vector<cd> dft_naive(const std::vector<cd>& x) {
    const size_t n = x.size();
    std::vector<cd> out(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> s = 0;
        for (size_t t = 0; t < n; ++t) {
            double ang = -2.0 * 3.14159265358979323846 * double(k) * double(t) / double(n);
            s += std::complex<double>(x[t]) * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        out[k] = cd(float(s.real()), float(s.imag()));
    }
    return out;
}

// 2D FFT nad nn x nn mrezom (row-major).
// Paralelizacija: redovi su medjusobno nezavisni, kolone su medjusobno nezavisne.
// Izmedju ta dva prolaza mora postojati barijera (implicitna na kraju "omp for").
inline void fft2d(std::vector<cd>& a, size_t nn, bool inverse) {
    const int n = int(nn);                  // OpenMP 2.0 (MSVC) trazi oznacen int kao brojac
#pragma omp parallel if(gUseOmp)
    {
        std::vector<cd> line(nn);           // svaka nit ima SVOJ privremeni niz

#pragma omp for
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) line[c] = a[size_t(r) * nn + size_t(c)];
            fft(line, inverse);
            for (int c = 0; c < n; ++c) a[size_t(r) * nn + size_t(c)] = line[c];
        }
        // <- barijera: kolone smiju poceti tek kad su svi redovi gotovi

#pragma omp for
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) line[r] = a[size_t(r) * nn + size_t(c)];
            fft(line, inverse);
            for (int r = 0; r < n; ++r) a[size_t(r) * nn + size_t(c)] = line[r];
        }
    }
}

// Spori referentni 2D DFT (N^4), samo za testiranje na malim N.
inline std::vector<cd> dft2d_naive(const std::vector<cd>& x, size_t n) {
    const double TWO_PI = 6.28318530717958647692;
    std::vector<cd> out(n * n);
    for (size_t u = 0; u < n; ++u)
        for (size_t v = 0; v < n; ++v) {
            std::complex<double> s = 0;
            for (size_t r = 0; r < n; ++r)
                for (size_t c = 0; c < n; ++c) {
                    double ang = -TWO_PI * double(u * r + v * c) / double(n);
                    s += std::complex<double>(x[r * n + c]) *
                        std::complex<double>(std::cos(ang), std::sin(ang));
                }
            out[u * n + v] = cd(float(s.real()), float(s.imag()));
        }
    return out;
}