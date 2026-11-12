#pragma once

// Result files of a TPMC run (docs/tpmc.md, section "Output").
//   results.json        central result: the six main quantities with standard
//                       errors and confidence intervals, reference values,
//                       statistics, diagnostics, provenance (always written)
//   results.csv         one row per scalar quantity (output.format: csv)
//   surface_fields.csv  per-triangle force, heat and impingement rate
//                       (output.save_surface_fields: true)

#include <string>

#include "vleo/geometry/Geometry.hpp"
#include "vleo/io/Config.hpp"
#include "vleo/tpmc/Tpmc.hpp"

namespace vleo {

inline constexpr const char* kResultsSchema = "vleo-results/1";

std::string results_json(const TpmcResult& r, const SimulationConfig& cfg, const std::string& timestamp_utc);
std::string results_csv(const TpmcResult& r);
std::string surface_fields_csv(const TpmcResult& r, const Geometry& g);

}  // namespace vleo
