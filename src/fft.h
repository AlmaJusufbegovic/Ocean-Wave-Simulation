#pragma once
#include <complex>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <map>

using cd = std::complex<float>;
constexpr float PI = 3.14159265358979f;

// Globalni prekidac: true = OpenMP paralelizacija ukljucena, false = sekvencijalno.
// (omogucava poredjenje obje verzije u ISTOM programu)
inline bool gUseOmp = false;

// Globalni prekidac: true = brza FFT (tablice, blokovi kolona, bez dijeljenja), false = stara FFT.
inline bool gFastFft = false;

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

// ===========================================================================
// BRZA FFT (optimizacije nastale na osnovu profilisanja u AMD uProf-u)
//   1) tablica twiddle faktora umjesto std::polar (sin/cos) u svakom leptiru,
//   2) tablica bit-reversal permutacije,
//   3) redovi se transformisu NA MJESTU (bez kopiranja u privremeni niz),
//   4) kolone se obradjuju u BLOKOVIMA od 8 (8 x 8 B = jedna keš linija),
//   5) bez dijeljenja sa N unutar FFT-a (za okean je potrebna cista suma).
// Rezultat je isti kao kod stare FFT (do na zaokruzivanje), osim normalizacije:
// fft2d_fast(a, n, true, false) odgovara  fft2d(a, n, true) * n^2.
// ===========================================================================
struct FftPlan {
    size_t n = 0;
    std::vector<unsigned> rev;   // bit-reversal permutacija
    std::vector<cd> tw;          // tw[j] = exp(-2 pi i j / n), j = 0..n/2-1
};

inline FftPlan makeFftPlan(size_t n) {
    FftPlan p;
    p.n = n;
    p.rev.resize(n);
    size_t bits = 0;
    while ((size_t(1) << bits) < n) ++bits;
    for (size_t i = 0; i < n; ++i) {
        size_t r = 0;
        for (size_t b = 0; b < bits; ++b)
            if (i & (size_t(1) << b)) r |= size_t(1) << (bits - 1 - b);
        p.rev[i] = unsigned(r);
    }
    p.tw.resize(n / 2 > 0 ? n / 2 : 1);
    for (size_t j = 0; j < n / 2; ++j) {
        const double ang = -2.0 * 3.14159265358979323846 * double(j) / double(n);
        p.tw[j] = cd(float(std::cos(ang)), float(std::sin(ang)));
    }
    return p;
}

// Plan se pravi jednom po velicini. Poziva se IZVAN paralelnog regiona (nije thread-safe).
inline const FftPlan& fftPlan(size_t n) {
    static std::map<size_t, FftPlan> cache;
    auto it = cache.find(n);
    if (it == cache.end()) it = cache.emplace(n, makeFftPlan(n)).first;
    return it->second;
}

// 1D FFT jedne linije od plan.n elemenata, na mjestu, BEZ normalizacije.
inline void fftPlanned(cd* a, const FftPlan& p, bool inverse) {
    const size_t n = p.n;
    for (size_t i = 0; i < n; ++i) {
        const size_t j = p.rev[i];
        if (i < j) std::swap(a[i], a[j]);
    }
    const float sgn = inverse ? -1.0f : 1.0f;
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t half = len >> 1, step = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < half; ++k) {
                const cd w = p.tw[k * step];
                const float wr = w.real(), wi = sgn * w.imag();
                const cd x = a[i + k];
                const cd y = a[i + k + half];
                // v = y * w (rucno, bez poziva std::complex operatora)
                const float vr = y.real() * wr - y.imag() * wi;
                const float vi = y.real() * wi + y.imag() * wr;
                a[i + k] = cd(x.real() + vr, x.imag() + vi);
                a[i + k + half] = cd(x.real() - vr, x.imag() - vi);
            }
        }
    }
}

// 2D FFT nn x nn. normalize = true dijeli rezultat sa nn^2 (kao stara fft2d pri inverse = true).
inline void fft2d_fast(std::vector<cd>& a, size_t nn, bool inverse, bool normalize = false) {
    const FftPlan& plan = fftPlan(nn);          // izvan paralelnog regiona
    const int n = int(nn);
    const int Bc = n >= 8 ? 8 : n;              // blok kolona
    const int nb = n / Bc;
#pragma omp parallel if(gUseOmp)
    {
        std::vector<cd> tile(size_t(Bc) * nn);  // privatni bafer svake niti

#pragma omp for
        for (int r = 0; r < n; ++r)
            fftPlanned(&a[size_t(r) * nn], plan, inverse);      // redovi na mjestu
        // barijera: kolone smiju poceti tek kad su svi redovi gotovi

#pragma omp for
        for (int cb = 0; cb < nb; ++cb) {
            const size_t c0 = size_t(cb) * size_t(Bc);
            for (size_t r = 0; r < nn; ++r)                       // ucitaj blok kolona
                for (int j = 0; j < Bc; ++j)
                    tile[size_t(j) * nn + r] = a[r * nn + c0 + size_t(j)];
            for (int j = 0; j < Bc; ++j)
                fftPlanned(&tile[size_t(j) * nn], plan, inverse);
            for (size_t r = 0; r < nn; ++r)                       // vrati blok kolona
                for (int j = 0; j < Bc; ++j)
                    a[r * nn + c0 + size_t(j)] = tile[size_t(j) * nn + r];
        }
    }
    if (normalize) {
        const float inv = 1.0f / float(nn * nn);
        const int total = int(nn * nn);
#pragma omp parallel for if(gUseOmp)
        for (int i = 0; i < total; ++i) a[size_t(i)] *= inv;
    }
}

// Inverzna 2D FFT za okean: vraca CISTU SUMU (bez dijeljenja sa N^2) kada je gFastFft = true;
// inace radi stara fft2d (koja dijeli sa N^2), pa pozivalac mora pomnoziti sa N^2.
inline void ifft2dRaw(std::vector<cd>& a, size_t nn) {
    if (gFastFft) fft2d_fast(a, nn, true, false);
    else fft2d(a, nn, true);
}
// Faktor kojim treba pomnoziti rezultat ifft2dRaw da se dobije suma.
inline float ifft2dScale(size_t nn) { return gFastFft ? 1.0f : float(nn * nn); }