#pragma once

// Time-dependent DSMC with elastic VHS collisions (No-Time-Counter scheme,
// Bird 1994), using the same geometry, inflow, GSI and tally definitions as
// TPMC. One time step (docs/dsmc.md section 1):
//
//   1. inject   free-stream molecules through the domain faces (count per
//               channel = rate dt / W, fractional part by a random number);
//               each enters at a random time within the step
//   2. move     ballistic motion for dt with surface interactions (Maxwell /
//               CLL), multiple reflections and domain exit (open boundary) or
//               specular reflection at the box walls (closed test boxes)
//   3. sample   (sampling phase only) momentum / energy transferred to the
//               body in this step, surface fields, cell populations
//   4. compact  remove particles that left or failed (order preserving)
//   5. bin      cell index, histogram + exclusive prefix sum, fill, and a
//               per-cell sort by particle id (deterministic order)
//   6. collide  NTC per cell: N_cand = N (N-1) W (sigma g)_max dt / (2 V_eff)
//               + remainder; uniform pairs without self-pairing; accept with
//               sigma g / (sigma g)_max; isotropic VHS scattering
//
// Reproducibility: all random numbers are bound to (cell, step), (particle,
// event) or (step, channel); particle order is deterministic (prefix-sum
// compaction, id-sorted cells) and the global tallies are reduced, not
// accumulated atomically. A run is therefore bitwise reproducible for a given
// backend and configuration.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vleo/core/Profiler.hpp"
#include "vleo/dsmc/CollisionGrid.hpp"
#include "vleo/dsmc/Vhs.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/io/Config.hpp"
#include "vleo/particles/Inflow.hpp"
#include "vleo/particles/ParticleStorage.hpp"
#include "vleo/tally/Statistics.hpp"
#include "vleo/tally/TimeSeries.hpp"
#include "vleo/tpmc/Tpmc.hpp"

namespace vleo {

enum class DsmcBoundary {
  Open,      // free-stream inflow through all faces, outflow removes particles
  Specular,  // closed box with specular walls, no inflow (homogeneous test cases)
};

// Resolved numerical parameters of a DSMC run.
struct DsmcSettings {
  double timestep_s = 0.0;
  double cell_size_m = 0.0;            // requested edge length (actual cells divide the box exactly)
  double particle_weight = 0.0;        // physical molecules per simulator particle W
  std::uint64_t warmup_steps = 0;
  std::uint64_t sample_steps = 0;
  std::uint32_t blocks = 20;
  std::uint64_t max_particles = 0;
  std::uint32_t max_interactions_per_step = 1000;
  DsmcBoundary boundary = DsmcBoundary::Open;
  bool initial_fill = true;            // start from free-stream gas (else empty domain)
  bool profile = false;                // phase timings (also VLEO_PROFILE=1)
  GridOptions grid;

  // Free-stream reference values used for the automatic choices.
  double lambda_inf_m = 0.0;
  double tau_inf_s = 0.0;
  double c_mp_inf_mps = 0.0;
  bool timestep_automatic = false, cell_size_automatic = false, warmup_automatic = false;
};

// Pilot runs for the automatic cell size (docs/dsmc.md section 3.1).
struct CellSizePilotOptions {
  // target: max cell edge <= lambda_fraction * min local lambda. Calibrated on
  // the M3.17 cell-size study (docs/dsmc.md section 3.1): the smallest local
  // lambda lies in the thin cold layer at the wall; C_D converged to 0.24 % at
  // dx = 1.7 lambda_min and was +3.6 % at 3.4 lambda_min.
  double lambda_fraction = 1.5;
  std::uint32_t particles_per_cell = 10;
  std::uint64_t sample_steps = 200;
  int max_pilots = 2;
  double min_gas_fraction = 0.25;       // cells used for the local lambda (cut cells below are noisy)
};

struct CellSizePilotReport {
  bool performed = false;
  int pilots = 0;
  double initial_cell_size_m = 0.0;     // free-stream rule min(lambda_inf / 2, L_max / 16)
  double min_local_lambda_m = 0.0;      // smallest local mean free path seen by the pilots
  double cell_size_m = 0.0;             // chosen cell size
  bool limited_by_max_cells = false;
  std::uint64_t steps = 0;
  double wall_time_s = 0.0;
};

// Mixture reference values (M4.2): with x_s the mole fractions and
// nu_s = sum_q n_q <sigma g>_sq the collision frequency of species s,
//   lambda_inf = sum_s x_s c_mean_s / nu_s,   tau_inf = 1 / sum_s x_s nu_s.
// Automatic rules (docs/dsmc.md section 3):
//   cell size  = min(lambda_inf / 2, L_max / 16)     (L_max: largest domain extent)
//   time step  = min(tau_inf / 10, dx_min / (4 (|V_inf| + 3 c_mp)))
//   W          = n_inf V_cell / particles_per_cell_target
//   warm-up    = 3 * (domain diagonal / max(|V_inf|, c_mp)) / dt   steps
DsmcSettings resolve_dsmc_settings(const SimulationConfig& cfg, const Domain& domain,
                                   std::optional<double> cell_size_override_m = std::nullopt);

struct DsmcCounters {
  std::uint64_t steps = 0;
  std::uint64_t injected = 0;
  std::uint64_t exited = 0;
  std::uint64_t truncated = 0;         // more than max_interactions_per_step in one step
  std::uint64_t lost_inside_body = 0;
  std::uint64_t invalid = 0;
  std::uint64_t removed_inside_at_fill = 0;
  std::uint64_t surface_interactions = 0;
  std::uint64_t collision_candidates = 0;
  std::uint64_t collisions = 0;
  std::uint64_t sigma_g_max_updates = 0;
};

// Per-species diagnostics of the sampling phase (M4.3).
struct DsmcSpeciesDiagnostics {
  std::string name;
  double mean_particles = 0.0;
  std::uint64_t injected = 0;          // whole run
  Vec3 force_N;                        // contribution of this species to the force on the body
  double heat_rate_W = 0.0;            // contribution to the heat rate
  std::vector<std::uint64_t> collisions_with;  // accepted collisions with species q (whole run)
};

struct DsmcDiagnostics {
  // resolution (M3.13)
  double cell_dx_m = 0.0, cell_dy_m = 0.0, cell_dz_m = 0.0;
  double timestep_s = 0.0;
  double lambda_inf_m = 0.0, tau_inf_s = 0.0;
  double cell_over_lambda_inf = 0.0;   // max cell edge / lambda_inf
  double dt_over_tau_inf = 0.0;
  double max_cfl = 0.0;                // max |v| dt / min cell edge over sampled steps
  double min_local_lambda_m = 0.0;     // from the sampled density: lambda_inf n_inf / n_cell
  double min_local_lambda_resolved_m = 0.0;  // same, cells with gas fraction >= 0.25 only
  double fraction_cells_dx_gt_lambda = 0.0;  // gas cells with max edge > local lambda
  double mean_particles_per_gas_cell = 0.0;
  double fraction_cells_below_5 = 0.0; // gas cells with < 5 particles on average
  double particles_mean = 0.0, particles_min = 0.0, particles_max = 0.0;
  // collisions (M3.8)
  double acceptance_ratio = 0.0;       // accepted / candidates
  double collisions_per_particle_per_step = 0.0;
  double free_stream_collisions_per_particle_per_step = 0.0;  // dt / tau_inf
};

struct DsmcResult {
  TpmcEstimates estimates;                 // same quantities as TPMC, SE from block averaging
  std::array<SeriesStatistics, kHistoryQuantities> series;  // per quantity (rates, unscaled)
  DsmcSettings settings;
  DsmcCounters counters;
  DsmcDiagnostics diagnostics;
  GridReport grid;
  double critical_value = 0.0;
  double wall_time_s = 0.0;
  std::optional<SurfaceFields> surface;
  std::vector<DsmcSpeciesDiagnostics> species;
  std::map<std::string, Profiler::Entry> profile;  // phase timings (profiling enabled only)
  double particle_steps = 0.0;                     // sum over steps of the particle number
  std::vector<std::string> warnings;
  CellSizePilotReport pilot;               // set by refine_cell_size_with_pilot
};

struct DsmcProgress {
  std::uint64_t step;
  std::uint64_t total_steps;
  bool sampling;
  std::size_t particles;
  double elapsed_s;
};
using DsmcProgressCallback = std::function<void(const DsmcProgress&)>;

class DsmcSimulation {
 public:
  DsmcSimulation(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                 const DsmcSettings& settings);
  ~DsmcSimulation();
  DsmcSimulation(const DsmcSimulation&) = delete;
  DsmcSimulation& operator=(const DsmcSimulation&) = delete;

  // Initial state: free-stream gas in the domain outside the body.
  void fill_freestream();
  // Replaces all particles (tests): velocities in the body frame, species index 0..
  void set_particles(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                     const std::vector<std::uint32_t>& species, const std::vector<double>& e_rot = {},
                     const std::vector<double>& e_vib = {});

  // Advances one time step; `sample` adds the step to the sampling statistics.
  void step(bool sample);

  std::size_t num_particles() const;  // active particles
  // Removes ended particles now (the time step compacts lazily; tests call
  // this before inspecting the particle arrays).
  void compact();
  const DsmcCounters& counters() const;
  const CollisionGrid& grid() const;
  const DsmcSettings& settings() const;
  std::uint64_t sampled_steps() const;

  // Host copies of the current particle state (tests, diagnostics).
  std::vector<Vec3> velocities() const;
  std::vector<Vec3> positions() const;
  std::vector<std::uint64_t> particle_ids() const;
  // Per-cell (offset, count) and particle list of the last binning (tests).
  void binning(std::vector<int>& offsets, std::vector<int>& list) const;
  std::vector<std::uint32_t> species() const;
  std::vector<double> rotational_energies() const;
  std::vector<double> vibrational_energies() const;
  // Accepted collisions per species pair (p <= q), whole run.
  std::uint64_t pair_collisions(int p, int q) const;

  // Statistics of the sampling phase.
  DsmcResult result() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Complete run: warm-up, sampling, statistics.
DsmcResult run_dsmc(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                    const DsmcSettings& settings, const DsmcProgressCallback& progress = {});

// Automatic cell size from the local mean free path (M5, open item of M3.17):
// starting from the free-stream rule, short pilot runs with few particles per
// cell measure the smallest local mean free path lambda_loc = lambda_inf
// n_inf / n_cell; the cell size is reduced to lambda_fraction * lambda_loc
// (at most max_pilots pilots, a second pilot re-measures on the finer grid).
// Returns settings resolved for the chosen cell size; no-op for fixed cells.
DsmcSettings refine_cell_size_with_pilot(const SimulationConfig& cfg, const Geometry& geometry,
                                         const InflowModel& inflow, const DsmcSettings& settings,
                                         CellSizePilotReport& report, const CellSizePilotOptions& options = {});

}  // namespace vleo
