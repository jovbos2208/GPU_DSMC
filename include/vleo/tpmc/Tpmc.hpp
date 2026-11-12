#pragma once

// Steady-state TPMC solver: forces, moments, aerodynamic coefficients and heat
// transfer with statistical uncertainties (docs/tpmc.md).
//
// Event weight (M2.6). The domain boundary emits free-stream molecules at the
// total rate R [1/s] (Inflow.hpp). N simulated histories are an i.i.d. sample
// of these molecules, so each history represents the rate gamma = R / N and
//     F = sum_e gamma dp_e = R * mean_h(dP_h),   dP_h = sum_{e in h} dp_e,
// and likewise for the moment M and the heat rate Q. N is the number of
// histories actually run, which makes an early convergence stop exact.

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vleo/geometry/Geometry.hpp"
#include "vleo/gsi/Scatter.hpp"
#include "vleo/io/Config.hpp"
#include "vleo/particles/Inflow.hpp"
#include "vleo/particles/Transport.hpp"
#include "vleo/tally/Statistics.hpp"

namespace vleo {

// Surface interaction for the transport kernel: per-material GSI parameters.
struct GsiInteraction {
  Kokkos::View<const SurfaceProperties*> properties;  // indexed by material id
  Kokkos::View<const std::uint32_t*> material;         // per triangle

  KOKKOS_INLINE_FUNCTION Vec3 operator()(const SurfaceEvent& e, RandomStream& rng) const {
    const SurfaceProperties& sp = properties(material(e.triangle));
    return scatter_particle(e.velocity_in, sp.wall_velocity, e.normal_gas, e.mass_kg, 0.0, sp, rng).velocity_out;
  }
  KOKKOS_INLINE_FUNCTION Vec3 wall_velocity(int triangle) const { return properties(material(triangle)).wall_velocity; }
};

SurfaceProperties make_surface_properties(const SurfaceSection& s);

struct TpmcEstimates {
  std::array<Estimate, 3> force_N;
  std::array<Estimate, 3> moment_Nm;
  Estimate heat_rate_W, heat_rate_positive_W, heat_rate_negative_W;
  Estimate drag_N, lift_N, side_N;  // force components along e_D, e_L, e_Y
  Estimate C_D, C_L, C_Y;
  std::array<Estimate, 3> C_M;
};

// Per-triangle results (input triangle order is available via source_index).
struct SurfaceFields {
  std::vector<Vec3> force_N;
  std::vector<double> heat_rate_W;
  std::vector<double> heat_rate_positive_W;
  std::vector<double> heat_rate_negative_W;
  std::vector<double> impingement_rate_per_s;
};

struct TpmcResult {
  TpmcEstimates estimates;
  MomentArray moments;  // per-history moments of all tallied quantities
  TransportCounters counters;
  std::uint64_t histories = 0;
  std::uint64_t batches = 0;
  double inflow_rate_per_s = 0.0;
  double event_rate_factor_per_s = 0.0;  // gamma = R / N
  double critical_value = 0.0;           // z for the configured confidence level
  bool converged = false;                // convergence criterion reached (false if none configured)
  double wall_time_s = 0.0;
  std::optional<SurfaceFields> surface;
  std::vector<std::string> warnings;
};

struct TpmcProgress {
  std::uint64_t histories;
  std::uint64_t target;
  double elapsed_s;
  Estimate monitored;  // C_D, or the convergence quantity if configured
};

using TpmcProgressCallback = std::function<void(const TpmcProgress&)>;

// Runs the TPMC simulation. The geometry must be uploaded (Kokkos initialised).
// `histories_override` (if > 0) replaces tpmc.particle_histories.
TpmcResult run_tpmc(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                    std::uint64_t histories_override = 0, const TpmcProgressCallback& progress = {});

// Estimates from merged moments (also used for the convergence check).
TpmcEstimates make_estimates(const MomentArray& m, double inflow_rate, const SimulationConfig& cfg, double z);

const Estimate& select(const TpmcEstimates& e, ConvergenceQuantity q);

}  // namespace vleo
