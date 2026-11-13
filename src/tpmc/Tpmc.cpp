#include "vleo/tpmc/Tpmc.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

namespace vleo {

SurfaceProperties make_surface_properties(const SurfaceSection& s) {
  SurfaceProperties p;
  p.model = s.model;
  p.temperature_K = s.temperature_K;
  if (s.maxwell) p.diffuse_fraction = s.maxwell->diffuse_fraction;
  if (s.cll) {
    p.alpha_n = s.cll->alpha_n;
    p.sigma_t = s.cll->sigma_t;
    p.alpha_rot = s.cll->alpha_rot;
    p.alpha_vib = s.cll->alpha_vib;
  }
  p.wall_velocity = Vec3();
  return p;
}

std::vector<SpeciesModes> make_species_modes(const GasSection& gas) {
  std::vector<SpeciesModes> v;
  for (const auto& sp : gas.species) {
    SpeciesModes m;
    m.mass = sp.mass_kg;
    m.rot_dof = sp.modes.rot_dof;
    m.rot_relax_prob = sp.modes.rot_collision_number > 0 ? 1.0 / sp.modes.rot_collision_number : 0.0;
    m.vib = sp.modes.vib_dof > 0 ? 1 : 0;
    m.theta_v = sp.modes.vib_temperature_K;
    m.vib_relax_prob = sp.modes.vib_collision_number > 0 ? 1.0 / sp.modes.vib_collision_number : 0.0;
    m.omega = sp.vhs.omega;
    v.push_back(m);
  }
  return v;
}

TpmcEstimates make_estimates(const MomentArray& m, double R, const SimulationConfig& cfg, double z) {
  TpmcEstimates e;
  auto est = [&](int q, double scale) { return make_estimate(m[static_cast<std::size_t>(q)], R * scale, z); };
  for (int k = 0; k < 3; ++k) {
    e.force_N[static_cast<std::size_t>(k)] = est(kQPx + k, 1.0);
    e.moment_Nm[static_cast<std::size_t>(k)] = est(kQLx + k, 1.0);
  }
  e.heat_rate_W = est(kQE, 1.0);
  e.heat_rate_positive_W = est(kQEPos, 1.0);
  e.heat_rate_negative_W = est(kQENeg, 1.0);
  e.drag_N = est(kQPDrag, 1.0);
  e.lift_N = est(kQPLift, 1.0);
  e.side_N = est(kQPSide, 1.0);
  const double qa = cfg.derived.dynamic_pressure_Pa * cfg.reference.area_m2;
  const double qal = qa * cfg.reference.length_m;
  e.C_D = est(kQPDrag, 1.0 / qa);
  e.C_L = est(kQPLift, 1.0 / qa);
  e.C_Y = est(kQPSide, 1.0 / qa);
  for (int k = 0; k < 3; ++k) e.C_M[static_cast<std::size_t>(k)] = est(kQLx + k, 1.0 / qal);
  return e;
}

const Estimate& select(const TpmcEstimates& e, ConvergenceQuantity q) {
  switch (q) {
    case ConvergenceQuantity::CD: return e.C_D;
    case ConvergenceQuantity::CL: return e.C_L;
    case ConvergenceQuantity::CY: return e.C_Y;
    case ConvergenceQuantity::CMx: return e.C_M[0];
    case ConvergenceQuantity::CMy: return e.C_M[1];
    case ConvergenceQuantity::CMz: return e.C_M[2];
    case ConvergenceQuantity::HeatRate: return e.heat_rate_W;
  }
  return e.C_D;
}

namespace {

template <class T>
Kokkos::View<const T*> to_device(const std::vector<T>& v, const char* label) {
  Kokkos::View<T*> d(std::string(label), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

bool reached(const Estimate& e, const ConvergenceCriterion& c) {
  if (c.relative_error) return e.std_error > 0.0 && e.std_error <= *c.relative_error * std::fabs(e.value);
  return e.std_error > 0.0 && e.std_error <= *c.absolute_error;
}

}  // namespace

TpmcResult run_tpmc(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                    std::uint64_t histories_override, const TpmcProgressCallback& progress) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::uint64_t target = histories_override > 0 ? histories_override : cfg.tpmc.particle_histories;
  const std::size_t batch = static_cast<std::size_t>(std::min<std::uint64_t>(cfg.tpmc.batch_size, target));

  // --- Device setup ---------------------------------------------------------
  std::vector<SurfaceProperties> props(geometry.host.material_names.size(), make_surface_properties(cfg.surface));
  GsiInteraction gsi;
  gsi.properties = to_device(props, "surface_properties");
  gsi.material = geometry.device.mesh.material;

  std::vector<double> masses;
  for (const auto& s : cfg.gas.species) masses.push_back(s.mass_kg);

  TransportSetup setup;
  setup.geometry = geometry.device;
  setup.domain = inflow.domain;
  setup.inflow = upload(inflow);
  setup.seed = cfg.simulation.random_seed;
  setup.max_interactions = cfg.tpmc.max_reflections;
  setup.weight = inflow.total_rate_per_s / static_cast<double>(target);
  setup.species_mass = to_device(masses, "species_mass");
  setup.species_modes = to_device(make_species_modes(cfg.gas), "species_modes");
  setup.tally.history = Kokkos::View<double**, Kokkos::LayoutLeft>("history_tallies", batch, kHistoryTallies);
  setup.tally.moment_center = cfg.reference.moment_center_m;
  if (cfg.output.save_surface_fields)
    setup.tally.surface =
        Kokkos::View<double**, Kokkos::LayoutLeft>("surface_tallies", geometry.host.size(), kSurfaceTallies);

  ParticleStorage particles(batch);
  const AeroAxes& axes = cfg.derived.aero_axes;

  TpmcResult r;
  r.inflow_rate_per_s = inflow.total_rate_per_s;
  r.critical_value = critical_value(cfg.statistics.confidence_level);
  const auto& conv = cfg.tpmc.convergence;

  // --- Batches --------------------------------------------------------------
  std::uint64_t done = 0;
  while (done < target) {
    const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(batch, target - done));
    const TransportCounters c = run_histories(setup, gsi, particles, done, n);
    r.counters += c;
    const MomentArray bm = reduce_history_moments(setup.tally.history, n, axes);
    for (std::size_t k = 0; k < r.moments.size(); ++k) r.moments[k].merge(bm[k]);
    done += n;
    ++r.batches;

    const TpmcEstimates e = make_estimates(r.moments, r.inflow_rate_per_s, cfg, r.critical_value);
    const Estimate& monitored = conv ? select(e, conv->quantity) : e.C_D;
    if (progress) {
      const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      progress({done, target, el, monitored});
    }
    if (conv && done >= conv->min_histories && done >= 2 && reached(monitored, *conv)) {
      r.converged = true;
      break;
    }
  }
  Kokkos::fence();
  r.histories = done;
  r.event_rate_factor_per_s = r.inflow_rate_per_s / static_cast<double>(done);
  r.estimates = make_estimates(r.moments, r.inflow_rate_per_s, cfg, r.critical_value);

  // --- Surface fields (same events, scaled by gamma = R / N) ------------------
  if (cfg.output.save_surface_fields) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), setup.tally.surface);
    SurfaceFields f;
    const double g = r.event_rate_factor_per_s;
    for (std::size_t t = 0; t < geometry.host.size(); ++t) {
      f.force_N.push_back(g * Vec3(h(t, kSurfPx), h(t, kSurfPy), h(t, kSurfPz)));
      f.heat_rate_W.push_back(g * h(t, kSurfE));
      f.heat_rate_positive_W.push_back(g * h(t, kSurfEPos));
      f.heat_rate_negative_W.push_back(g * h(t, kSurfENeg));
      f.impingement_rate_per_s.push_back(g * h(t, kSurfHits));
    }
    r.surface = std::move(f);
  }

  // --- Quality warnings -------------------------------------------------------
  const TransportCounters& c = r.counters;
  if (c.truncated > 0) {
    std::ostringstream os;
    os << c.truncated << " histories reached tpmc.max_reflections = " << cfg.tpmc.max_reflections
       << " and were truncated (their remaining interactions are missing)";
    r.warnings.push_back(os.str());
  }
  if (c.lost_inside_body + c.invalid > 0) {
    std::ostringstream os;
    os << c.lost_inside_body << " histories lost inside the body and " << c.invalid
       << " invalid histories (numerical failures; please report)";
    r.warnings.push_back(os.str());
  }
  if (!c.balanced()) r.warnings.push_back("particle balance violated");
  if (conv && !r.converged)
    r.warnings.push_back("convergence criterion not reached within tpmc.particle_histories");

  r.wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace vleo
