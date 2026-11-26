#include "RunLog.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "vleo/core/BuildInfo.hpp"
#include "vleo/core/PhysicalConstants.hpp"

namespace vleo {

namespace {

constexpr double kRadToDeg = 57.29577951308232;

std::string num(double v, int precision = 6) {
  std::ostringstream s;
  s << std::setprecision(precision) << v;
  return s.str();
}

std::string vec(const Vec3& v, int precision = 6) {
  return "(" + num(v.x, precision) + ", " + num(v.y, precision) + ", " + num(v.z, precision) + ")";
}

std::string count(std::uint64_t v) {  // 12345678 -> 12 345 678
  std::string d = std::to_string(v), r;
  for (std::size_t i = 0; i < d.size(); ++i) {
    if (i > 0 && (d.size() - i) % 3 == 0) r += ' ';
    r += d[i];
  }
  return r;
}

std::string mode_name(ParameterMode m) { return m == ParameterMode::Automatic ? "automatic" : "fixed"; }

std::string duration(double s) {
  if (!(s >= 0.0) || !std::isfinite(s)) return "-";
  std::ostringstream o;
  if (s < 60.0) {
    o << std::fixed << std::setprecision(1) << s << " s";
  } else {
    const auto t = static_cast<long long>(std::llround(s));
    o << t / 3600 << ":" << std::setw(2) << std::setfill('0') << (t / 60) % 60 << ":" << std::setw(2) << t % 60;
  }
  return o.str();
}

const char* const kInflowFaceNames[6] = {"-x", "+x", "-y", "+y", "-z", "+z"};

}  // namespace

RunLog::RunLog(const std::filesystem::path& file) : out_(file, std::ios::trunc) {
  if (!out_) throw std::runtime_error("cannot write '" + file.string() + "'");
}

void RunLog::section(const std::string& title) {
  out_ << "\n" << title << "\n" << std::string(title.size(), '-') << "\n";
}

void RunLog::kv(const std::string& key, const std::string& value) {
  out_ << "  " << std::left << std::setw(34) << key << std::right << ": " << value << "\n";
}

void RunLog::header(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                    const RuntimeInfo& rt, const std::filesystem::path& out_dir) {
  const BuildInfo& b = build_info();
  const std::string rule(80, '=');
  out_ << rule << "\n vleo_solver " << b.configure.solver_version << " - simulation log\n" << rule << "\n";

  section("Run");
  kv("name", cfg.simulation.name.empty() ? "-" : cfg.simulation.name);
  kv("solver", std::string(to_string(cfg.simulation.mode)));
  kv("started (UTC)", rt.timestamp_utc);
  std::string cmd;
  for (const auto& a : rt.command_line) cmd += (cmd.empty() ? "" : " ") + a;
  kv("command line", cmd);
  kv("working directory", rt.working_directory);
  kv("configuration", cfg.source_file.string());
  kv("output directory", std::filesystem::absolute(out_dir).lexically_normal().string());
  kv("random seed", std::to_string(cfg.simulation.random_seed));
  const std::string git = b.git.available ? std::string(b.git.describe) : "unknown";
  kv("build", git + ", " + std::string(b.configure.build_type) + ", Kokkos " + b.kokkos_version);
  std::string backend = std::string(b.configure.backend) + " / " + rt.execution_space;
  if (rt.device_id >= 0) backend += ", device " + std::to_string(rt.device_id);
  kv("backend", backend);
  if (rt.mpi_ranks > 1) {
    std::string d;
    for (int id : rt.rank_device_ids) d += (d.empty() ? "" : ", ") + std::to_string(id);
    kv("MPI ranks", std::to_string(rt.mpi_ranks) + (rt.device_id >= 0 ? " (devices " + d + ")" : ""));
  }

  // --- free stream ---------------------------------------------------------
  const GasSection& g = cfg.gas;
  const DerivedQuantities& dq = cfg.derived;
  axes_ = dq.aero_axes;
  section("Free stream (body frame)");
  for (std::size_t i = 0; i < g.species.size(); ++i) {
    const SpeciesFraction& s = g.species[i];
    std::ostringstream o;
    o << std::left << std::setw(4) << s.name << std::right << " x = " << std::setw(9) << num(s.mole_fraction, 6)
      << "   n = " << num(s.mole_fraction * g.number_density_m3, 6) << " m^-3   m = " << num(s.mass_kg, 6)
      << " kg";
    if (s.modes.rot_dof > 0) o << "   rot";
    if (s.modes.vib_dof > 0) o << " + vib (theta_v " << num(s.modes.vib_temperature_K, 5) << " K)";
    kv(i == 0 ? "species" : "", o.str());
  }
  kv("number density n_inf", num(g.number_density_m3) + " m^-3");
  kv("mass density rho_inf", num(dq.mass_density_kg_m3) + " kg/m^3");
  kv("mean molecular mass", num(dq.mean_molecular_mass_kg) + " kg");
  kv("temperature T_inf", num(g.translational_temperature_K) + " K");
  kv("bulk velocity V_inf", vec(g.bulk_velocity_body_mps) + " m/s");
  kv("speed |V_inf|", num(dq.freestream_speed_mps) + " m/s");
  if (dq.freestream_speed_mps > 0.0) {
    const Vec3 d = (-1.0 / dq.freestream_speed_mps) * g.bulk_velocity_body_mps;  // flight direction
    const double alpha = std::atan2(d.z, d.x) * kRadToDeg + 0.0;  // + 0.0: no "-0"
    const double beta = std::asin(std::clamp(d.y, -1.0, 1.0)) * kRadToDeg + 0.0;
    kv("flight direction (alpha, beta)", num(alpha, 6) + " deg, " + num(beta, 6) + " deg");
  }
  if (dq.mean_molecular_mass_kg > 0.0 && g.translational_temperature_K > 0.0) {
    const double c_mp = std::sqrt(2.0 * constants::k_B * g.translational_temperature_K / dq.mean_molecular_mass_kg);
    kv("speed ratio |V_inf| / c_mp", num(dq.freestream_speed_mps / c_mp, 6) + "   (c_mp = " + num(c_mp, 6) + " m/s)");
  }
  kv("dynamic pressure q_inf", num(dq.dynamic_pressure_Pa) + " Pa");
  try {
    const double lambda = resolve_dsmc_settings(cfg, inflow.domain).lambda_inf_m;
    std::string v = num(lambda) + " m";
    if (cfg.reference.length_m > 0.0) v += "   (Kn = lambda / L_ref = " + num(lambda / cfg.reference.length_m) + ")";
    kv("mean free path lambda_inf (VHS)", v);
  } catch (const std::exception&) {
    // no collision data (e.g. TPMC with species without VHS parameters)
  }

  // --- surface -----------------------------------------------------------------
  const SurfaceSection& sf = cfg.surface;
  section("Gas-surface interaction");
  kv("model", std::string(to_string(sf.model)));
  if (sf.maxwell) kv("diffuse fraction", num(sf.maxwell->diffuse_fraction));
  if (sf.cll) {
    kv("alpha_n / sigma_t", num(sf.cll->alpha_n) + " / " + num(sf.cll->sigma_t));
    kv("alpha_rot / alpha_vib", num(sf.cll->alpha_rot) + " / " + num(sf.cll->alpha_vib));
  }
  kv("wall temperature T_w", num(sf.temperature_K) + " K");
  if (!sf.material_file.empty()) kv("material file", sf.material_file.string());

  // --- geometry ----------------------------------------------------------------
  const MeshReport& m = geometry.report;
  section("Geometry");
  kv("file", cfg.geometry.resolved_file.string());
  kv("units (scale to m)", std::string(to_string(cfg.geometry.units)) + " (" + num(dq.geometry_scale_to_m) + ")");
  std::string tri = count(m.triangles);
  if (m.degenerate_removed > 0) tri += " (" + count(m.degenerate_removed) + " degenerate removed)";
  kv("triangles", tri);
  kv("vertices / edges", count(m.vertices) + " / " + count(m.edges));
  double volume = 0.0;
  for (const auto& c : m.components) volume += c.signed_volume_m3;
  kv("components", std::to_string(m.components.size()) + (m.closed ? ", all closed" : ", open component(s)"));
  kv("surface area", num(m.total_area_m2) + " m^2");
  if (m.closed) kv("enclosed volume", num(volume) + " m^3");
  kv("bounding box min", vec(m.bbox.lo) + " m");
  kv("bounding box max", vec(m.bbox.hi) + " m");
  kv("extent", vec(m.bbox.hi - m.bbox.lo) + " m");
  if (!geometry.host.material_names.empty()) {
    std::string mats;
    for (const auto& n : geometry.host.material_names) mats += (mats.empty() ? "" : ", ") + n;
    kv("materials", mats);
  }
  kv("BVH nodes / depth", count(geometry.bvh.stats.nodes) + " / " + std::to_string(geometry.bvh.stats.max_depth));
  for (const auto& w : m.warnings) kv("warning", w);

  section("Reference values");
  kv("reference area A_ref", num(cfg.reference.area_m2) + " m^2");
  kv("reference length L_ref", num(cfg.reference.length_m) + " m");
  kv("moment centre", vec(cfg.reference.moment_center_m) + " m");
  kv("drag axis e_D", vec(dq.aero_axes.drag));
  kv("lift axis e_L", vec(dq.aero_axes.lift) + (dq.lift_axis_was_projected ? "  (projected)" : ""));
  kv("side axis e_Y", vec(dq.aero_axes.side));

  // --- domain and inflow ----------------------------------------------------------
  const Aabb& box = inflow.domain.box;
  const Vec3 e = box.hi - box.lo;
  section("Domain and inflow");
  kv("box min", vec(box.lo) + " m");
  kv("box max", vec(box.hi) + " m");
  kv("extent", vec(e) + " m");
  kv("volume", num(e.x * e.y * e.z) + " m^3");
  kv("total inflow rate", num(inflow.total_rate_per_s) + " molecules/s");
  double face_rate[6] = {};
  for (const auto& c : inflow.channels) face_rate[c.face] += c.rate_per_s;
  for (int f = 0; f < 6; ++f)
    if (face_rate[f] > 0.0)
      kv(std::string("  face ") + kInflowFaceNames[f],
         num(face_rate[f]) + " /s (" + num(100.0 * face_rate[f] / inflow.total_rate_per_s, 4) + " %)");
  flush();
}

void RunLog::dsmc_parameters(const SimulationConfig& cfg, const InflowModel& inflow, const DsmcSettings& s,
                             const CellSizePilotReport& pilot) {
  const DSMCSection& d = cfg.dsmc;
  section("DSMC parameters");
  kv("collision model", std::string(to_string(d.collision_model)));
  kv("mean free path lambda_inf", num(s.lambda_inf_m) + " m");
  kv("mean collision time tau_inf", num(s.tau_inf_s) + " s");
  kv("most probable speed c_mp", num(s.c_mp_inf_mps) + " m/s");
  kv("time step dt", num(s.timestep_s) + " s (" + mode_name(d.timestep_mode) + ", dt / tau_inf = " +
                         num(s.timestep_s / s.tau_inf_s, 4) + ")");
  kv("cell size", num(s.cell_size_m) + " m (" + mode_name(d.cell_size_mode) +
                      ", dx / lambda_inf = " + num(s.cell_size_m / s.lambda_inf_m, 4) + ")");
  if (pilot.performed) {
    kv("cell-size pilot", std::to_string(pilot.pilots) + " run(s), " + duration(pilot.wall_time_s) +
                              ": min local lambda " + num(pilot.min_local_lambda_m) + " m, " +
                              num(pilot.initial_cell_size_m) + " -> " + num(pilot.cell_size_m) + " m" +
                              (pilot.limited_by_max_cells ? " (limited by max cells)" : ""));
  }
  kv("domain boundary", std::string(to_string(d.boundary)));
  for (std::size_t f = 0; f < 6; ++f) {
    if (!d.faces[f]) continue;
    const DomainFace& fc = *d.faces[f];
    std::string v(to_string(fc.type));
    if (fc.type == FaceType::Wall) v += ", diffuse, T = " + num(fc.temperature_K) + " K, u = " + vec(fc.velocity_mps) + " m/s";
    if (fc.type == FaceType::Inflow)
      v += ", n = " + num(fc.number_density_m3) + " m^-3, T = " + num(fc.temperature_K) + " K, u = " +
           vec(fc.velocity_mps) + " m/s" + (fc.fill_depth_m > 0.0 ? ", fill " + num(fc.fill_depth_m) + " m" : "");
    kv(std::string("  face ") + std::string(kFaceNames[f]), v);
  }
  kv("particles per cell (target)", std::to_string(d.particles_per_cell_target));
  kv("MACROPARTICLE FACTOR W (F_N)", num(s.particle_weight, 8) + " real molecules per simulator particle");
  const Vec3 e = inflow.domain.box.extent();
  // as make_device_grid: the cells divide the box exactly
  std::uint64_t cells = 1;
  for (int k = 0; k < 3; ++k)
    cells *= static_cast<std::uint64_t>(std::max(1.0, std::ceil(e[k] / s.cell_size_m * (1.0 - 1e-12))));
  kv("cells", count(cells));
  kv("simulator particles (free stream)",
     "~" + count(static_cast<std::uint64_t>(std::llround(cfg.gas.number_density_m3 * e.x * e.y * e.z /
                                                         s.particle_weight))) +
         " (domain volume x n_inf / W)");
  kv("injected per step", "~" + num(inflow.total_rate_per_s * s.timestep_s / s.particle_weight, 6));
  kv("max particles", s.max_particles > 0 ? count(s.max_particles) : "automatic");
  const std::uint64_t total = s.warmup_steps + s.sample_steps;
  kv("warm-up steps", count(s.warmup_steps) + (s.warmup_automatic ? " (automatic)" : ""));
  kv("sampling steps / blocks", count(s.sample_steps) + " / " + std::to_string(s.blocks));
  kv("simulated time", num(static_cast<double>(total) * s.timestep_s) + " s (" + count(total) + " steps, sampling " +
                           num(static_cast<double>(s.sample_steps) * s.timestep_s) + " s)");
  kv("load balancing (MPI)", s.load_balance ? "on (end of warm-up)" : "off");
  flush();
}

void RunLog::tpmc_parameters(const SimulationConfig& cfg, const InflowModel& inflow) {
  const TPMCSection& t = cfg.tpmc;
  section("TPMC parameters");
  kv("particle histories", count(t.particle_histories));
  kv("batch size", count(t.batch_size));
  kv("max reflections per history", std::to_string(t.max_reflections));
  const double gamma = inflow.total_rate_per_s / static_cast<double>(t.particle_histories);
  kv("weight per history gamma = R / N", num(gamma, 8) + " real molecules/s per test particle");
  kv("", "(TPMC has no time step: each history stands for gamma molecules per second)");
  if (t.convergence) {
    const ConvergenceCriterion& c = *t.convergence;
    std::string v = std::string(to_string(c.quantity)) + ": ";
    v += c.relative_error ? "relative SE <= " + num(*c.relative_error) : "absolute SE <= " + num(*c.absolute_error);
    v += ", min " + count(c.min_histories) + " histories";
    kv("convergence criterion", v);
  } else {
    kv("convergence criterion", "none (fixed number of histories)");
  }
  kv("confidence level", num(cfg.statistics.confidence_level));
  flush();
}

void RunLog::progress(const DsmcProgress& p, double timestep_s) {
  if (!progress_header_) {
    section("Progress");
    out_ << "  " << std::setw(23) << "step" << std::setw(9) << "done" << "  " << std::left << std::setw(9) << "phase"
         << std::right << std::setw(14) << "particles" << std::setw(14) << "t_sim [s]" << std::setw(12) << "wall"
         << std::setw(12) << "remaining" << "  time (UTC)\n";
    progress_header_ = true;
  }
  const double frac = static_cast<double>(p.step) / static_cast<double>(std::max<std::uint64_t>(1, p.total_steps));
  const double remaining = frac > 0.0 ? p.elapsed_s * (1.0 - frac) / frac : -1.0;
  std::ostringstream pct;
  pct << std::fixed << std::setprecision(1) << 100.0 * frac << " %";
  out_ << "  " << std::setw(23) << (count(p.step) + " / " + count(p.total_steps)) << std::setw(9) << pct.str() << "  "
       << std::left << std::setw(9) << (p.sampling ? "sampling" : "warm-up") << std::right << std::setw(14)
       << count(p.particles) << std::setw(14) << num(static_cast<double>(p.step) * timestep_s, 5) << std::setw(12)
       << duration(p.elapsed_s) << std::setw(12) << duration(remaining) << "  " << utc_timestamp_now() << "\n";
  final_wall_s_ = p.elapsed_s;
  flush();
}

void RunLog::progress(const TpmcProgress& p, const std::string& monitored_name) {
  const double frac = static_cast<double>(p.histories) / static_cast<double>(std::max<std::uint64_t>(1, p.target));
  // one line per 1 % or 30 s (TPMC reports every batch)
  const bool last = p.histories >= p.target;
  if (!last && frac - last_logged_fraction_ < 0.01 && p.elapsed_s - last_logged_wall_s_ < 30.0) return;
  last_logged_fraction_ = frac;
  last_logged_wall_s_ = p.elapsed_s;
  if (!progress_header_) {
    section("Progress");
    out_ << "  " << std::setw(27) << "histories" << std::setw(9) << "done" << std::setw(28)
         << (monitored_name + " +- SE") << std::setw(12) << "wall" << std::setw(12) << "remaining"
         << "  time (UTC)\n";
    progress_header_ = true;
  }
  const double remaining = frac > 0.0 ? p.elapsed_s * (1.0 - frac) / frac : -1.0;
  std::ostringstream pct;
  pct << std::fixed << std::setprecision(1) << 100.0 * frac << " %";
  out_ << "  " << std::setw(27) << (count(p.histories) + " / " + count(p.target)) << std::setw(9) << pct.str()
       << std::setw(28) << (num(p.monitored.value, 7) + " +- " + num(p.monitored.std_error, 3)) << std::setw(12)
       << duration(p.elapsed_s) << std::setw(12) << duration(remaining) << "  " << utc_timestamp_now() << "\n";
  final_wall_s_ = p.elapsed_s;
  flush();
}

void RunLog::estimates(const TpmcEstimates& e, double critical_value, const AeroAxes& axes) {
  section("Results (SE = standard error, CI at z = " + num(critical_value, 4) + ")");
  out_ << "  aerodynamic frame: e_D along V_inf, e_L lift axis, e_Y = e_L x e_D (axes under Reference values)\n"
          "  body frame: F_x .. M_z and C_Mx .. C_Mz\n";
  out_ << "  " << std::left << std::setw(14) << "quantity" << std::right << std::setw(16) << "value" << std::setw(14)
       << "SE" << std::setw(10) << "SE/|v|" << std::setw(16) << "CI low" << std::setw(16) << "CI high" << "\n";
  auto row = [&](const std::string& name, const Estimate& x) {
    const double rel = x.value != 0.0 ? std::fabs(x.std_error / x.value) : 0.0;
    std::ostringstream r;
    r << std::fixed << std::setprecision(2) << 100.0 * rel << " %";
    out_ << "  " << std::left << std::setw(14) << name << std::right << std::setw(16) << num(x.value, 8)
         << std::setw(14) << num(x.std_error, 4) << std::setw(10) << r.str() << std::setw(16) << num(x.ci_low, 8)
         << std::setw(16) << num(x.ci_high, 8) << "\n";
  };
  row("C_D", e.C_D);
  row("C_L", e.C_L);
  row("C_Y", e.C_Y);
  // moment coefficients about the aerodynamic axes: projection of the body
  // components (SE without the covariance of the components, which is not tallied)
  auto project = [&](const Vec3& a) {
    Estimate x;
    x.value = a.x * e.C_M[0].value + a.y * e.C_M[1].value + a.z * e.C_M[2].value;
    x.std_error = std::sqrt(a.x * a.x * e.C_M[0].std_error * e.C_M[0].std_error +
                            a.y * a.y * e.C_M[1].std_error * e.C_M[1].std_error +
                            a.z * a.z * e.C_M[2].std_error * e.C_M[2].std_error);
    x.ci_low = x.value - critical_value * x.std_error;
    x.ci_high = x.value + critical_value * x.std_error;
    return x;
  };
  row("C_M,D (e_D)", project(axes.drag));
  row("C_M,Y (e_Y)", project(axes.side));
  row("C_M,L (e_L)", project(axes.lift));
  row("C_Mx", e.C_M[0]);
  row("C_My", e.C_M[1]);
  row("C_Mz", e.C_M[2]);
  row("drag [N]", e.drag_N);
  row("lift [N]", e.lift_N);
  row("side [N]", e.side_N);
  row("F_x [N]", e.force_N[0]);
  row("F_y [N]", e.force_N[1]);
  row("F_z [N]", e.force_N[2]);
  row("M_x [N m]", e.moment_Nm[0]);
  row("M_y [N m]", e.moment_Nm[1]);
  row("M_z [N m]", e.moment_Nm[2]);
  row("Q [W]", e.heat_rate_W);
  row("Q+ [W]", e.heat_rate_positive_W);
  row("Q- [W]", e.heat_rate_negative_W);
}

void RunLog::dsmc_summary(const DsmcResult& r) {
  estimates(r.estimates, r.critical_value, axes_);
  const DsmcDiagnostics& d = r.diagnostics;
  const GridReport& g = r.grid;
  const DsmcCounters& c = r.counters;
  section("DSMC diagnostics");
  kv("MACROPARTICLE FACTOR W (F_N)", num(r.settings.particle_weight, 8));
  kv("cells (gas / cut / solid)",
     count(static_cast<std::uint64_t>(g.cells)) + " (" + count(static_cast<std::uint64_t>(g.gas_cells)) + " / " +
         count(static_cast<std::uint64_t>(g.cut_cells)) + " / " + count(static_cast<std::uint64_t>(g.solid_cells)) +
         ")");
  kv("cell edges dx, dy, dz", num(d.cell_dx_m) + ", " + num(d.cell_dy_m) + ", " + num(d.cell_dz_m) + " m");
  kv("gas volume", num(g.gas_volume_m3) + " of " + num(g.domain_volume_m3) + " m^3");
  kv("particles mean / min / max", num(d.particles_mean, 8) + " / " + num(d.particles_min, 8) + " / " +
                                       num(d.particles_max, 8));
  kv("mean particles per gas cell", num(d.mean_particles_per_gas_cell, 4) + "   (cells < 5: " +
                                        num(100.0 * d.fraction_cells_below_5, 3) + " %)");
  kv("max cell / lambda_inf", num(d.cell_over_lambda_inf, 4));
  kv("dt / tau_inf", num(d.dt_over_tau_inf, 4));
  kv("max CFL", num(d.max_cfl, 4));
  kv("min local lambda (all / f>=0.25)", num(d.min_local_lambda_m) + " / " + num(d.min_local_lambda_resolved_m) + " m");
  kv("gas cells with dx > local lambda", num(100.0 * d.fraction_cells_dx_gt_lambda, 4) + " %");
  kv("collisions per particle and step", num(d.collisions_per_particle_per_step, 4) + " (free stream " +
                                             num(d.free_stream_collisions_per_particle_per_step, 4) + ")");
  kv("acceptance ratio", num(d.acceptance_ratio, 4));
  kv("steps", count(c.steps));
  kv("injected / exited", count(c.injected) + " / " + count(c.exited));
  kv("surface interactions", count(c.surface_interactions));
  kv("collisions (candidates)", count(c.collisions) + " (" + count(c.collision_candidates) + ")");
  kv("truncated / lost / invalid", count(c.truncated) + " / " + count(c.lost_inside_body) + " / " + count(c.invalid));
  if (r.species.size() > 1) {
    for (const auto& s : r.species)
      kv("species " + s.name, "mean " + num(s.mean_particles, 7) + " particles, injected " + count(s.injected) +
                                  ", F = " + vec(s.force_N, 5) + " N, Q = " + num(s.heat_rate_W, 5) + " W");
  }
  const DsmcDecomposition& dc = r.decomposition;
  if (dc.ranks > 1) {
    kv("MPI ranks", std::to_string(dc.ranks) + (dc.rebalanced ? ", rebalanced" : ""));
    kv("imbalance before / after", num(dc.imbalance_before, 4) + " / " + num(dc.imbalance_after, 4));
    kv("migrated particles", count(dc.migrated));
  }
  final_wall_s_ = r.wall_time_s;
  flush();
}

void RunLog::tpmc_summary(const TpmcResult& r) {
  estimates(r.estimates, r.critical_value, axes_);
  const TransportCounters& c = r.counters;
  section("TPMC diagnostics");
  kv("histories / batches", count(r.histories) + " / " + count(r.batches));
  kv("weight per history gamma", num(r.event_rate_factor_per_s, 8) + " molecules/s");
  kv("convergence criterion reached", r.converged ? "yes" : "no / not configured");
  kv("histories with surface hits", count(c.histories_with_hits) + " (" +
                                        num(100.0 * static_cast<double>(c.histories_with_hits) /
                                                static_cast<double>(std::max<std::uint64_t>(1, r.histories)),
                                            4) +
                                        " %)");
  kv("surface interactions", count(c.surface_interactions) + " (max " + count(c.max_interactions) + " per history)");
  kv("truncated / lost / invalid", count(c.truncated) + " / " + count(c.lost_inside_body) + " / " + count(c.invalid));
  if (r.ranks > 1) kv("MPI ranks", std::to_string(r.ranks));
  final_wall_s_ = r.wall_time_s;
  flush();
}

void RunLog::message(const std::string& kind, const std::string& text) {
  out_ << kind << ": " << text << "\n";
  flush();
}

void RunLog::finish(const std::string& status, const std::vector<std::string>& files) {
  section("End");
  kv("status", status);
  kv("solver wall time", duration(final_wall_s_) + " (" + num(final_wall_s_) + " s)");
  kv("finished (UTC)", utc_timestamp_now());
  std::string f;
  for (const auto& n : files) f += (f.empty() ? "" : ", ") + n;
  kv("output files", f);
  flush();
}

}  // namespace vleo
