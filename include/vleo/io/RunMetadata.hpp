#pragma once

// Run metadata: everything required to reproduce and audit a run (build
// provenance, runtime environment, resolved configuration, derived values).

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "vleo/io/Config.hpp"
#include "vleo/io/JsonWriter.hpp"
#include "vleo/io/PlatformConfig.hpp"

namespace vleo {

inline constexpr const char* kRunMetadataSchema = "vleo-run-metadata/1";

struct RuntimeInfo {
  std::string timestamp_utc;          // ISO 8601, e.g. 2026-11-12T10:00:00Z
  std::vector<std::string> command_line;
  std::string working_directory;
  std::string execution_space;        // Kokkos default execution space at runtime
  int concurrency = 0;                // Kokkos::DefaultExecutionSpace().concurrency()
  int device_id = -1;                 // Kokkos::device_id(); -1 for host backends
};

struct RunMetadataInput {
  const SimulationConfig* config = nullptr;     // required
  const PlatformConfig* platform = nullptr;     // optional
  RuntimeInfo runtime;
  std::string status;                           // e.g. "configured", "completed"
  std::vector<std::string> notes;               // human-readable remarks / warnings
  std::vector<std::string> output_files;        // files written next to the metadata
  // Additional top-level sections (e.g. "geometry", "inflow"): key and a
  // callback that writes exactly one JSON value.
  std::vector<std::pair<std::string, std::function<void(JsonWriter&)>>> sections;
};

std::string make_run_metadata_json(const RunMetadataInput& in);

std::string utc_timestamp_now();

}  // namespace vleo
