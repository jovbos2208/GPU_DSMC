#pragma once

// results.json for DSMC runs: same schema (vleo-results/1) and quantities as
// TPMC; the statistics block describes the block averaging and the
// diagnostics block the DSMC resolution and collision statistics.

#include <string>

#include "vleo/dsmc/Dsmc.hpp"

namespace vleo {

std::string dsmc_results_json(const DsmcResult& r, const SimulationConfig& cfg, const std::string& timestamp_utc);

}  // namespace vleo
