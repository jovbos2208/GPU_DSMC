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
#include "vleo/io/JsonWriter.hpp"
#include "vleo/tpmc/Tpmc.hpp"

namespace vleo {

inline constexpr const char* kResultsSchema = "vleo-results/1";

std::string results_json(const TpmcResult& r, const SimulationConfig& cfg, const std::string& timestamp_utc);
std::string results_csv(const TpmcResult& r);
std::string surface_fields_csv(const TpmcResult& r, const Geometry& g);

// Building blocks shared with the DSMC result writer (same schema).
void write_simulation_and_reference(JsonWriter& w, const SimulationConfig& cfg);
void write_estimate_blocks(JsonWriter& w, const TpmcEstimates& e);  // results, std_error, confidence_interval
void write_provenance(JsonWriter& w, const std::string& timestamp_utc);
std::string estimates_csv(const TpmcEstimates& e);
void write_profile(JsonWriter& w, const std::map<std::string, Profiler::Entry>& profile);
std::string surface_fields_csv(const SurfaceFields& f, const Geometry& g);

}  // namespace vleo
