#pragma once

// Statistical estimation from independent samples (TPMC histories).
//
// Every TPMC history is an independent, identically distributed sample. For a
// per-history quantity x_h (e.g. the momentum it transferred) the estimate of
// the corresponding rate quantity is  Y = R * mean(x)  with the inflow rate
// R [1/s], and its standard error is  R * sqrt(var(x) / N).
//
// Moments are reduced per batch on the device (two passes: mean, then the sum
// of squared deviations) and merged across batches with the pairwise update of
// Chan, Golub and LeVeque (1979), which is numerically stable even when most
// histories contribute exactly zero.

#include <array>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "vleo/core/Frames.hpp"

namespace vleo {

struct Moments {
  std::uint64_t n = 0;
  double mean = 0.0;
  double m2 = 0.0;  // sum of squared deviations from the mean

  void merge(const Moments& o);
  double sample_variance() const { return n > 1 ? m2 / static_cast<double>(n - 1) : 0.0; }
  double std_error_of_mean() const;
};

// Inverse of the standard normal CDF, |error| < 1e-12 for p in (1e-300, 1 - 1e-16).
double normal_quantile(double p);

// Two-sided normal-approximation critical value for a confidence level, e.g. 0.95 -> 1.959964.
double critical_value(double confidence_level);

struct Estimate {
  double value = 0.0;
  double std_error = 0.0;
  double ci_low = 0.0;
  double ci_high = 0.0;
};

// Estimate of scale * mean(x) with its standard error and confidence interval.
Estimate make_estimate(const Moments& m, double scale, double z);

// Quantities reduced per history: the 9 history tallies (Transport.hpp) plus
// the momentum projected on the aerodynamic axes (drag, lift, side).
enum HistoryQuantity : int {
  kQPx = 0, kQPy, kQPz, kQLx, kQLy, kQLz, kQE, kQEPos, kQENeg,
  kQPDrag, kQPLift, kQPSide,
  kHistoryQuantities
};

using MomentArray = std::array<Moments, kHistoryQuantities>;

// Moments of all history quantities over rows [0, n) of the history tally
// array (device). Deterministic for a given backend and launch configuration.
MomentArray reduce_history_moments(const Kokkos::View<double**, Kokkos::LayoutLeft>& history, std::size_t n,
                                   const AeroAxes& axes);

}  // namespace vleo
