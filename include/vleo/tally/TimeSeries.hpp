#pragma once

// Statistics of correlated time series (DSMC sampling phase).
//
// Consecutive DSMC time steps are correlated, so the naive standard error
// sqrt(var / N) is too small. The primary estimate is block averaging: the
// series is split into B contiguous blocks of equal length; the block means
// are treated as independent and SE = sd(block means) / sqrt(B). As a
// diagnostic the integrated autocorrelation time tau_int is estimated with
// Geyer's initial positive sequence; blocks shorter than ~5 tau_int are
// reported because their means are still correlated.

#include <cstddef>
#include <vector>

namespace vleo {

struct SeriesStatistics {
  std::size_t samples = 0;
  double mean = 0.0;
  double variance = 0.0;             // sample variance of the individual values
  double std_error_naive = 0.0;      // sqrt(var / N), valid only without correlation
  double std_error_blocks = 0.0;     // block-averaging estimate (primary)
  std::size_t blocks = 0;
  std::size_t block_length = 0;
  double tau_int = 0.5;              // integrated autocorrelation time in samples (0.5: uncorrelated)
  double std_error_tau = 0.0;        // sqrt(var (2 tau_int) / N)
  bool blocks_long_enough = true;    // block_length >= 5 tau_int
};

// Statistics of `x` with `blocks` blocks (the last N mod B values are dropped
// for the block estimate only).
SeriesStatistics analyze_series(const std::vector<double>& x, std::size_t blocks);

}  // namespace vleo
