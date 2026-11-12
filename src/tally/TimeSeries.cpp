#include "vleo/tally/TimeSeries.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vleo {

SeriesStatistics analyze_series(const std::vector<double>& x, std::size_t blocks) {
  SeriesStatistics s;
  const std::size_t n = x.size();
  s.samples = n;
  if (n == 0) return s;
  if (blocks < 2) throw std::invalid_argument("analyze_series: need at least 2 blocks");

  double sum = 0.0;
  for (double v : x) sum += v;
  s.mean = sum / static_cast<double>(n);
  double m2 = 0.0;
  for (double v : x) m2 += (v - s.mean) * (v - s.mean);
  s.variance = n > 1 ? m2 / static_cast<double>(n - 1) : 0.0;
  s.std_error_naive = std::sqrt(s.variance / static_cast<double>(n));

  // Block averaging.
  s.blocks = std::min(blocks, n);
  s.block_length = n / s.blocks;
  if (s.block_length >= 1 && s.blocks >= 2) {
    std::vector<double> bm(s.blocks, 0.0);
    for (std::size_t b = 0; b < s.blocks; ++b) {
      double acc = 0.0;
      for (std::size_t i = 0; i < s.block_length; ++i) acc += x[b * s.block_length + i];
      bm[b] = acc / static_cast<double>(s.block_length);
    }
    double mb = 0.0;
    for (double v : bm) mb += v;
    mb /= static_cast<double>(s.blocks);
    double vb = 0.0;
    for (double v : bm) vb += (v - mb) * (v - mb);
    vb /= static_cast<double>(s.blocks - 1);
    s.std_error_blocks = std::sqrt(vb / static_cast<double>(s.blocks));
  }

  // Integrated autocorrelation time (Geyer's initial positive sequence).
  if (s.variance > 0.0 && n > 3) {
    const double var0 = m2 / static_cast<double>(n);  // biased autocovariance normalisation
    auto rho = [&](std::size_t lag) {
      double c = 0.0;
      for (std::size_t i = 0; i + lag < n; ++i) c += (x[i] - s.mean) * (x[i + lag] - s.mean);
      return c / static_cast<double>(n) / var0;
    };
    double tau = 0.5;  // tau_int = 1/2 + sum_{t >= 1} rho(t)
    const std::size_t max_lag = std::min<std::size_t>(n / 2, 2000);
    for (std::size_t t = 1; t + 1 < max_lag; t += 2) {
      const double pair = rho(t) + rho(t + 1);  // Gamma_k = rho(2k-1) + rho(2k), must stay positive
      if (!(pair > 0.0)) break;
      tau += pair;
    }
    s.tau_int = std::max(0.5, tau);
  }
  s.std_error_tau = std::sqrt(s.variance * 2.0 * s.tau_int / static_cast<double>(n));
  s.blocks_long_enough = static_cast<double>(s.block_length) >= 5.0 * s.tau_int;
  return s;
}

}  // namespace vleo
