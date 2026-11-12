#include "vleo/tally/Statistics.hpp"

#include <cmath>
#include <stdexcept>

#include "vleo/particles/Transport.hpp"

namespace vleo {

void Moments::merge(const Moments& o) {
  if (o.n == 0) return;
  if (n == 0) {
    *this = o;
    return;
  }
  const double na = static_cast<double>(n), nb = static_cast<double>(o.n);
  const double nt = na + nb;
  const double delta = o.mean - mean;
  mean += delta * nb / nt;
  m2 += o.m2 + delta * delta * na * nb / nt;
  n += o.n;
}

double Moments::std_error_of_mean() const {
  return n > 1 ? std::sqrt(sample_variance() / static_cast<double>(n)) : 0.0;
}

double normal_quantile(double p) {
  if (!(p > 0.0 && p < 1.0)) throw std::invalid_argument("normal_quantile: p must be in (0, 1)");
  // Acklam's rational approximation (rel. error < 1.2e-9) ...
  static constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                 1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
  static constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                 6.680131188771972e+01,  -1.328068155288572e+01};
  static constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                                 -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
  static constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                 3.754408661907416e+00};
  constexpr double p_low = 0.02425;
  double x;
  if (p < p_low) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (p <= 1.0 - p_low) {
    const double q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log1p(-p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  // ... refined by two Halley steps on Phi(x) = erfc(-x / sqrt 2) / 2.
  for (int it = 0; it < 2; ++it) {
    const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    const double u = e * std::sqrt(2.0 * 3.14159265358979323846) * std::exp(0.5 * x * x);
    x -= u / (1.0 + 0.5 * x * u);
  }
  return x;
}

double critical_value(double confidence_level) {
  if (!(confidence_level > 0.0 && confidence_level < 1.0))
    throw std::invalid_argument("critical_value: confidence level must be in (0, 1)");
  return normal_quantile(0.5 + 0.5 * confidence_level);
}

Estimate make_estimate(const Moments& m, double scale, double z) {
  Estimate e;
  e.value = scale * m.mean;
  e.std_error = std::fabs(scale) * m.std_error_of_mean();
  e.ci_low = e.value - z * e.std_error;
  e.ci_high = e.value + z * e.std_error;
  return e;
}

namespace {

KOKKOS_INLINE_FUNCTION void history_quantities(const Kokkos::View<double**, Kokkos::LayoutLeft>& h, std::size_t i,
                                               const AeroAxes& ax, double* q) {
  for (int k = 0; k < kHistoryTallies; ++k) q[k] = h(i, k);
  const Vec3 dp(q[kQPx], q[kQPy], q[kQPz]);
  q[kQPDrag] = dot(dp, ax.drag);
  q[kQPLift] = dot(dp, ax.lift);
  q[kQPSide] = dot(dp, ax.side);
}

struct SumKernel {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count = kHistoryQuantities;
  Kokkos::View<double**, Kokkos::LayoutLeft> h;
  AeroAxes ax;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, value_type sum) const {
    double q[kHistoryQuantities];
    history_quantities(h, i, ax, q);
    for (int k = 0; k < kHistoryQuantities; ++k) sum[k] += q[k];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type sum) const {
    for (int k = 0; k < kHistoryQuantities; ++k) sum[k] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (int k = 0; k < kHistoryQuantities; ++k) dst[k] += src[k];
  }
};

struct DevKernel {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count = kHistoryQuantities;
  Kokkos::View<double**, Kokkos::LayoutLeft> h;
  AeroAxes ax;
  Kokkos::Array<double, kHistoryQuantities> mean;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, value_type sum) const {
    double q[kHistoryQuantities];
    history_quantities(h, i, ax, q);
    for (int k = 0; k < kHistoryQuantities; ++k) {
      const double d = q[k] - mean[k];
      sum[k] += d * d;
    }
  }
  KOKKOS_INLINE_FUNCTION void init(value_type sum) const {
    for (int k = 0; k < kHistoryQuantities; ++k) sum[k] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (int k = 0; k < kHistoryQuantities; ++k) dst[k] += src[k];
  }
};

}  // namespace

MomentArray reduce_history_moments(const Kokkos::View<double**, Kokkos::LayoutLeft>& history, std::size_t n,
                                   const AeroAxes& axes) {
  MomentArray out{};
  if (n == 0) return out;
  using policy = Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>;
  double sums[kHistoryQuantities] = {};
  SumKernel sk;
  sk.h = history;
  sk.ax = axes;
  Kokkos::parallel_reduce("vleo::history_sums", policy(0, n), sk, sums);
  DevKernel dk;
  dk.h = history;
  dk.ax = axes;
  for (int k = 0; k < kHistoryQuantities; ++k) dk.mean[k] = sums[k] / static_cast<double>(n);
  double dev[kHistoryQuantities] = {};
  Kokkos::parallel_reduce("vleo::history_deviations", policy(0, n), dk, dev);
  for (int k = 0; k < kHistoryQuantities; ++k) {
    out[static_cast<std::size_t>(k)].n = n;
    out[static_cast<std::size_t>(k)].mean = dk.mean[k];
    out[static_cast<std::size_t>(k)].m2 = dev[k];
  }
  return out;
}

}  // namespace vleo
