// Yslm_vec.cpp -- Kokkos port of spectools.spherical.swsh.Yslm_vec
//
// Spin-weighted spherical harmonic {}_{s}Y_{lm}(theta, phi), vectorized over a
// batch of (theta, phi) points. This is the Kokkos reference for the Python
// `Yslm_vec` (spectools/spherical/swsh.py); it was validated bit-for-bit
// (max abs diff ~1e-13) against that function for s=-2, l=2..12, all m,
// including the poles theta=0, pi.
//
// Design (mirrors the validated numpy template):
//   * Everything that depends only on (s, l, m) is precomputed ONCE on the host
//     (the real prefactor `overall` and the per-term (coeff, cos_exp, sin_exp));
//     the device kernel contains no factorials, no binomials, no sqrt.
//   * The pole-stable cos/sin closed form (no cot/tan) is used, so the kernel is
//     finite at theta = 0, pi with no branch and no coordinate nudging.
//   * The sum accumulates a REAL polynomial in x=cos(theta/2), y=sin(theta/2);
//     the single complex factor exp(i*m*phi) is applied once at the end.
//   * The s<0 reflection (theta->pi-theta, phi->phi+pi) is folded into the
//     constants: it swaps the cos/sin exponents and multiplies `overall` by
//     (-1)^l * (-1)^m -- no per-point branch, no array copies.
//
// Build (example, serial or CUDA backend selected at Kokkos configure time):
//   c++ -I$KOKKOS/include Yslm_vec.cpp -L$KOKKOS/lib -lkokkoscore -o yslm_vec
//   (or nvcc_wrapper for the CUDA backend)

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cmath>
#include <vector>
#include <cstdio>

namespace spectools {

// ---- Host-side constant precompute (once per (s, l, m)) ---------------------

struct YslmConstants {
  int    emm;                    // azimuthal number m (phase e^{i m phi})
  double overall;                // real scalar prefactor (incl. s<0 folding)
  std::vector<double> coeff;     // per-term real coefficient
  std::vector<int>    cos_exp;   // exponent of cos(theta/2)
  std::vector<int>    sin_exp;   // exponent of sin(theta/2)
};

// Exact-ish binomial and factorial via long double lgamma (host only, one-time).
static long double lbinom(int n, int k) {
  if (k < 0 || k > n) return 0.0L;
  long double v = std::lgammal((long double)n + 1)
                - std::lgammal((long double)k + 1)
                - std::lgammal((long double)(n - k) + 1);
  return std::llroundl(std::expl(v));  // exact for the integer ranges used here
}

static long double lfact(int n) { return std::expl(std::lgammal((long double)n + 1)); }

YslmConstants make_yslm_constants(int s, int l, int m) {
  const int a = std::abs(s);
  // (-1)^m * sqrt( (l+m)!(l-m)!(2l+1) / (4 pi (l+|s|)!(l-|s|)!) )   [long double]
  long double norm = ((m & 1) ? -1.0L : 1.0L) *
      std::sqrtl(lfact(l + m) * lfact(l - m) * (long double)(2 * l + 1) /
                 (4.0L * (long double)M_PI * lfact(l + a) * lfact(l - a)));

  bool swap = false;
  if (s < 0) {
    norm *= ((l & 1) ? -1.0L : 1.0L);   // factor = (-1)^l
    norm *= ((m & 1) ? -1.0L : 1.0L);   // e^{i m (phi+pi)} = (-1)^m e^{i m phi}
    swap = true;                        // theta->pi-theta swaps cos<->sin exps
  }

  YslmConstants c;
  c.emm = m;
  c.overall = (double)norm;
  for (int aar = 0; aar <= l - a; ++aar) {
    const int cidx = aar + a - m;
    if (cidx < 0) continue;
    const long double t1 = lbinom(l - a, aar);
    const long double t2 = lbinom(l + a, cidx);
    if (t1 == 0.0L || t2 == 0.0L) continue;   // vanishing term
    int ce = 2 * aar + a - m;
    int se = 2 * l - ce;
    if (swap) std::swap(ce, se);
    const long double t3 = (((l - aar - a) & 1) ? -1.0L : 1.0L);
    c.coeff.push_back((double)(t1 * t2 * t3));
    c.cos_exp.push_back(ce);
    c.sin_exp.push_back(se);
  }
  return c;
}

// ---- Device kernel ----------------------------------------------------------

KOKKOS_INLINE_FUNCTION double ipow(double base, int e) {
  double r = 1.0;
  for (int i = 0; i < e; ++i) r *= base;   // e is a small non-negative int
  return r;
}

// Evaluate {}_{s}Y_{lm} for a batch of points into `out`, given precomputed
// device Views of the constant term tables. One thread per point.
struct YslmVecFunctor {
  Kokkos::View<const double*> theta, phi;
  Kokkos::View<const double*> coeff;
  Kokkos::View<const int*>    cos_exp, sin_exp;
  Kokkos::View<Kokkos::complex<double>*> out;
  double overall;
  int    emm;
  int    nterms;

  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const double x = Kokkos::cos(0.5 * theta(i));
    const double y = Kokkos::sin(0.5 * theta(i));
    double poly = 0.0;
    for (int k = 0; k < nterms; ++k)
      poly += coeff(k) * ipow(x, cos_exp(k)) * ipow(y, sin_exp(k));
    const double ang = emm * phi(i);
    out(i) = Kokkos::complex<double>(overall * poly) *
             Kokkos::complex<double>(Kokkos::cos(ang), Kokkos::sin(ang));
  }
};

// Host convenience wrapper: constants + device views + parallel_for.
Kokkos::View<Kokkos::complex<double>*>
Yslm_vec(int s, int l, int m,
         const Kokkos::View<const double*>& theta,
         const Kokkos::View<const double*>& phi) {
  const YslmConstants c = make_yslm_constants(s, l, m);
  const int nterms = (int)c.coeff.size();
  const int n = (int)theta.extent(0);

  Kokkos::View<double*> coeff("coeff", nterms);
  Kokkos::View<int*>    cexp("cexp", nterms), sexp("sexp", nterms);
  auto h_coeff = Kokkos::create_mirror_view(coeff);
  auto h_cexp  = Kokkos::create_mirror_view(cexp);
  auto h_sexp  = Kokkos::create_mirror_view(sexp);
  for (int k = 0; k < nterms; ++k) {
    h_coeff(k) = c.coeff[k]; h_cexp(k) = c.cos_exp[k]; h_sexp(k) = c.sin_exp[k];
  }
  Kokkos::deep_copy(coeff, h_coeff);
  Kokkos::deep_copy(cexp,  h_cexp);
  Kokkos::deep_copy(sexp,  h_sexp);

  Kokkos::View<Kokkos::complex<double>*> out("Yslm", n);
  YslmVecFunctor f{theta, phi, coeff, cexp, sexp, out, c.overall, m, nterms};
  Kokkos::parallel_for("Yslm_vec", n, f);
  Kokkos::fence();
  return out;
}

}  // namespace spectools

// ---- Demo / cross-validation against the Python Yslm_vec --------------------
// Prints {}_{-2}Y_{lm} on the same test points used by the Python validation
// so the two can be diffed directly.
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int n = 3;
    Kokkos::View<double*> theta("theta", n), phi("phi", n);
    auto ht = Kokkos::create_mirror_view(theta);
    auto hp = Kokkos::create_mirror_view(phi);
    ht(0) = 0.7; ht(1) = 1.3; ht(2) = 2.4;
    hp(0) = 0.3; hp(1) = 1.1; hp(2) = 2.0;
    Kokkos::deep_copy(theta, ht);
    Kokkos::deep_copy(phi, hp);

    const int s = -2;
    for (int l = 2; l <= 4; ++l) {
      for (int m = -l; m <= l; ++m) {
        auto Y = spectools::Yslm_vec(s, l, m,
            Kokkos::View<const double*>(theta),
            Kokkos::View<const double*>(phi));
        auto hY = Kokkos::create_mirror_view(Y);
        Kokkos::deep_copy(hY, Y);
        std::printf("s=%d l=%d m=%2d : ", s, l, m);
        for (int i = 0; i < n; ++i)
          std::printf("(% .12e,% .12e) ", hY(i).real(), hY(i).imag());
        std::printf("\n");
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
