#pragma once

// Human-readable run log `simulation.out` (rank 0 only).
//
// Written while the run progresses (flushed after every entry, so it can be
// followed with `tail -f`): run and build information, free-stream
// conditions, surface model, geometry, domain and inflow, the numerical
// parameters (DSMC: time step, cells, particle weight W = physical molecules
// per simulator particle; TPMC: histories and the event-rate factor), the
// progress of the run with an estimate of the remaining time, and finally the
// results with standard errors, diagnostics, warnings and the output files.
// The machine-readable data stay in results.json / run_metadata.json.

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/io/Config.hpp"
#include "vleo/io/RunMetadata.hpp"
#include "vleo/particles/Inflow.hpp"
#include "vleo/tpmc/Tpmc.hpp"

namespace vleo {

class RunLog {
 public:
  explicit RunLog(const std::filesystem::path& file);

  // Run, conditions, surface, geometry, domain and inflow.
  void header(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
              const RuntimeInfo& rt, const std::filesystem::path& out_dir);
  void dsmc_parameters(const SimulationConfig& cfg, const InflowModel& inflow, const DsmcSettings& s,
                       const CellSizePilotReport& pilot);
  void tpmc_parameters(const SimulationConfig& cfg, const InflowModel& inflow);

  void progress(const DsmcProgress& p, double timestep_s);
  void progress(const TpmcProgress& p, const std::string& monitored_name);

  void dsmc_summary(const DsmcResult& r);
  void tpmc_summary(const TpmcResult& r);
  void message(const std::string& kind, const std::string& text);  // "warning", "error", "note"
  void finish(const std::string& status, const std::vector<std::string>& files);

 private:
  void section(const std::string& title);
  void kv(const std::string& key, const std::string& value);
  void estimates(const TpmcEstimates& e, double critical_value);
  void flush() { out_.flush(); }

  std::ofstream out_;
  bool progress_header_ = false;
  double last_logged_fraction_ = -1.0, last_logged_wall_s_ = 0.0;
  double final_wall_s_ = 0.0;
};

}  // namespace vleo
