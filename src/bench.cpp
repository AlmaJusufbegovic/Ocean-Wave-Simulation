// Profilisanje i poredjenje: sekvencijalno vs OpenMP (1, 2, 4, 8 niti), za tri varijante:
//   "5 IFFT" : osnovna varijanta (computeFields)  - 5 inverznih 2D FFT-ova po frejmu
//   "3 IFFT" : pakovana varijanta (computeFieldsPacked) - dva realna polja u jednom IFFT-u
//   "frame"  : pun frame kao u viewer-u: pakovana kaskada 0 + pjena (Jacobijan) + kaskada 1 (nagibi)
// Mjeri se samo racunanje (bez OpenGL-a, bez pripreme tjemena).
// Pokretanje:  bench.exe          (puno mjerenje, oko 3 minute)
//              bench.exe quick    (brzi probni prolaz, N = 128 i 256)
#include "ocean.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static volatile double g_sink = 0.0;   // sprjecava da kompajler odbaci rezultate

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Cfg { const char* name; bool omp; int threads; };
enum Variant { V_BASE = 0, V_PACKED = 1, V_FULL = 2 };
static const char* VLABEL[] = { "5 IFFT", "3 IFFT", "frame" };

struct Times {
    double spectrum = 0, prep = 0, ifft = 0, extract = 0, foam = 0, detail = 0;
    double total() const { return spectrum + prep + ifft + extract + foam + detail; }
};

// Sve sto treba za jedan N.
struct Setup {
    OceanParams p;                  // cijeli spektar (varijante 5 IFFT i 3 IFFT)
    std::vector<cd> h0;
    OceanParams p0, p1;             // kaskade (varijanta frame)
    std::vector<cd> h0a, h0b;
    std::vector<float> foam;
    explicit Setup(size_t N) {
        p.N = N;
        h0 = generateH0(p);
        p0.N = N; p0.L = 400.0f; p0.windSpeed = 14.0f; p0.band = 1; p0.splitLo = 0.8f; p0.splitHi = 1.4f;
        p0.A = amplitudeForRms(p0, 0.0214f * 14.0f * 14.0f / 4.0f);
        p1 = p0;
        p1.N = 128; p1.L = 40.0f; p1.seed = 1337; p1.band = 2;
        p1.A = p0.A * (400.0f / 40.0f) * (400.0f / 40.0f);
        h0a = generateH0(p0);
        h0b = generateH0(p1);
        foam.assign(N * N, 0.0f);
    }
};

static Times runOnce(Variant v, Setup& s, float t) {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
        };
    Times tm;
    FrameTimes ft;
    if (v == V_BASE) {
        const OceanFields f = computeFields(s.h0, s.p, t, &ft);
        g_sink = g_sink + double(f.height[0]);
    }
    else {
        const OceanParams& p = (v == V_PACKED) ? s.p : s.p0;
        const std::vector<cd>& h0 = (v == V_PACKED) ? s.h0 : s.h0a;
        const OceanFields f = computeFieldsPacked(h0, p, t, &ft);
        g_sink = g_sink + double(f.height[0]);
        if (v == V_FULL) {
            const auto a = clk::now();
            updateFoam(f, s.p0, 1.6f, 0.016f, 0.78f, 4.0f, 1.5f, s.foam);
            const auto b = clk::now();
            const std::vector<float> rg = computeSlopesRG(s.h0b, s.p1, t);
            const auto c = clk::now();
            g_sink = g_sink + double(rg[0]) + double(s.foam[0]);
            tm.foam = ms(a, b);
            tm.detail = ms(b, c);
        }
    }
    tm.spectrum = ft.spectrum; tm.prep = ft.prep; tm.ifft = ft.ifft; tm.extract = ft.extract;
    return tm;
}

static double maxRelDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, double(std::fabs(a[i] - b[i])));
        r = std::max(r, double(std::fabs(a[i])));
    }
    return r > 0 ? m / r : m;
}

int main(int argc, char** argv) {
    const bool quick = (argc > 1 && std::strcmp(argv[1], "quick") == 0);
    const int warmup = quick ? 3 : 20;
    const int repsFinal = quick ? 5 : 30;
    const int repsWarmPass = quick ? 2 : 5;
    const std::vector<size_t> sizes = quick ? std::vector<size_t>{128, 256} : std::vector<size_t>{ 128, 256, 512 };
    const Cfg cfgs[] = {
        { "seq", false, 1 },
        { "omp", true, 1 },
        { "omp", true, 2 },
        { "omp", true, 4 },
        { "omp", true, 8 },
    };

    // ---- 1) provjere ispravnosti ----
    {
        std::printf("PROVJERE ISPRAVNOSTI (N=256)\n");
        Setup s(256);
        gUseOmp = false;
        const OceanFields a = computeFields(s.h0, s.p, 3.0f);
        const OceanFields b = computeFieldsPacked(s.h0, s.p, 3.0f);
        std::printf("  3 IFFT naspram 5 IFFT (najveca relativna razlika): h=%.1e  nagibX=%.1e  nagibZ=%.1e  Dx=%.1e  Dz=%.1e\n",
            maxRelDiff(a.height, b.height), maxRelDiff(a.slopeX, b.slopeX), maxRelDiff(a.slopeZ, b.slopeZ),
            maxRelDiff(a.dispX, b.dispX), maxRelDiff(a.dispZ, b.dispZ));

        gUseOmp = true;
        omp_set_num_threads(8);
        const OceanFields a2 = computeFields(s.h0, s.p, 3.0f);
        const OceanFields b2 = computeFieldsPacked(s.h0, s.p, 3.0f);
        std::printf("  OpenMP naspram sekvencijalno, 5 IFFT: %g   3 IFFT: %g\n",
            std::max(maxRelDiff(a.height, a2.height), maxRelDiff(a.dispX, a2.dispX)),
            std::max(maxRelDiff(b.height, b2.height), maxRelDiff(b.dispX, b2.dispX)));

        // pjena i kaskada 1: sekvencijalno naspram OpenMP
        std::vector<float> foamSeq(256 * 256, 0.0f), foamOmp(256 * 256, 0.0f);
        gUseOmp = false;
        const OceanFields f0 = computeFieldsPacked(s.h0a, s.p0, 3.0f);
        updateFoam(f0, s.p0, 1.6f, 0.016f, 0.78f, 4.0f, 1.5f, foamSeq);
        const std::vector<float> dSeq = computeSlopesRG(s.h0b, s.p1, 3.0f);
        gUseOmp = true;
        const OceanFields f1 = computeFieldsPacked(s.h0a, s.p0, 3.0f);
        updateFoam(f1, s.p0, 1.6f, 0.016f, 0.78f, 4.0f, 1.5f, foamOmp);
        const std::vector<float> dOmp = computeSlopesRG(s.h0b, s.p1, 3.0f);
        std::printf("  OpenMP naspram sekvencijalno, pjena: %g   kaskada 1 (nagibi): %g\n\n",
            maxRelDiff(foamSeq, foamOmp), maxRelDiff(dSeq, dOmp));
    }

    // ---- 2) mjerenja ----
    FILE* csv = std::fopen("bench_results.csv", "w");
    if (csv)
        std::fprintf(csv, "N;varijanta;nacin;niti;spektar_ms;priprema_ms;ifft_ms;izvlacenje_ms;pjena_ms;detalji_ms;ukupno_ms;ubrzanje;efikasnost;ubrzanje_naspram_osnovne_seq\n");

    std::printf("MJERENJE: %d zagrijavanja, %d ponavljanja, medijan, sve u ms%s\n", warmup, repsFinal, quick ? "  [quick]" : "");
    std::printf("Varijante: '5 IFFT' = osnovna, '3 IFFT' = pakovana, 'frame' = pakovana + pjena + kaskada 1\n");
    std::printf("'ubrz.' = naspram iste varijante sekvencijalno, 'vs osn.' = naspram osnovne (5 IFFT) sekvencijalne\n");
    std::printf("%-4s %-6s %-4s %2s | %6s %6s %7s %6s %6s %6s | %8s | %6s %5s | %7s\n",
        "N", "var.", "nac.", "ni", "spekt", "prip.", "ifft", "izvl.", "pjena", "detalj", "ukupno", "ubrz.", "efik.", "vs osn.");

    // prolaz 0 samo "zagrijava" procesor (frekvencija, kesevi) i ne ispisuje se
    for (int pass = 0; pass < 2; ++pass) {
        const bool show = (pass == 1);
        const int reps = show ? repsFinal : repsWarmPass;
        for (size_t N : sizes) {
            Setup s(N);
            double baseSeqTotal = 0;
            for (int vi = 0; vi < 3; ++vi) {
                const Variant v = Variant(vi);
                double seqTotal = 0;
                for (const Cfg& cfg : cfgs) {
                    gUseOmp = cfg.omp;
                    omp_set_num_threads(cfg.threads);
                    for (int i = 0; i < warmup; ++i) runOnce(v, s, 1.0f);

                    std::vector<double> vs, vp, vf, ve, vfo, vd, vt;
                    for (int i = 0; i < reps; ++i) {
                        const Times tm = runOnce(v, s, 1.0f + 0.01f * float(i));
                        vs.push_back(tm.spectrum); vp.push_back(tm.prep); vf.push_back(tm.ifft);
                        ve.push_back(tm.extract); vfo.push_back(tm.foam); vd.push_back(tm.detail);
                        vt.push_back(tm.total());
                    }
                    const double total = median(vt);
                    if (!cfg.omp) {
                        seqTotal = total;
                        if (v == V_BASE) baseSeqTotal = total;
                    }
                    const double speedup = seqTotal / total;
                    const double eff = cfg.omp ? speedup / cfg.threads : 1.0;
                    const double vsBase = baseSeqTotal / total;
                    if (!show) continue;

                    const bool full = (v == V_FULL);
                    char foamS[16], detS[16];
                    if (full) {
                        std::snprintf(foamS, sizeof(foamS), "%6.2f", median(vfo));
                        std::snprintf(detS, sizeof(detS), "%6.2f", median(vd));
                    }
                    else {
                        std::snprintf(foamS, sizeof(foamS), "%6s", "-");
                        std::snprintf(detS, sizeof(detS), "%6s", "-");
                    }
                    std::printf("%-4zu %-6s %-4s %2d | %6.2f %6.2f %7.2f %6.2f %s %s | %8.2f | %5.2fx %4.0f%% | %6.2fx\n",
                        N, VLABEL[vi], cfg.name, cfg.threads, median(vs), median(vp), median(vf), median(ve),
                        foamS, detS, total, speedup, eff * 100.0, vsBase);
                    if (csv)
                        std::fprintf(csv, "%zu;%s;%s;%d;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f\n", N, VLABEL[vi],
                            cfg.name, cfg.threads, median(vs), median(vp), median(vf), median(ve),
                            full ? median(vfo) : 0.0, full ? median(vd) : 0.0, total, speedup, eff, vsBase);
                }
                if (show) std::printf("\n");
            }
        }
    }
    if (csv) std::fclose(csv);
    std::printf("Rezultati su snimljeni u bench_results.csv (checksum %g)\n", double(g_sink));
    return 0;
}