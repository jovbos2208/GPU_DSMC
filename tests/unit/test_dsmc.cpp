// M3.1, M3.8, M3.11-M3.16 and Gate M3: DSMC time loop, homogeneous gas,
// collision rate, relaxation, conservation, free-molecular limit vs TPMC,
// reproducibility; block averaging of correlated series.
//
// Tolerances (fixed before the tests were run, docs/dsmc.md section 7):
//   energy conservation in a closed specular box: 1e-10 relative over the run
//   collision count vs. equilibrium theory: 5 sqrt(K) + 0.5 % (NTC bound updates)
//   temperatures / distributions: 5 standard errors, KS sqrt(n) D < 1.95
//   DSMC vs TPMC (free-molecular): 5 sqrt(SE_a^2 + SE_b^2)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <sstream>

#include "vleo/core/Species.hpp"
#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/io/ConfigParser.hpp"
#include "vleo/tally/TimeSeries.hpp"
#include "vleo/tpmc/Tpmc.hpp"

using namespace vleo;
namespace fs = std::filesystem;
using Catch::Matchers::WithinRel;

#ifndef VLEO_TEST_SCALE
#define VLEO_TEST_SCALE 1
#endif

namespace {

struct Case {
  SimulationConfig cfg;
  Geometry geometry;
  InflowModel inflow;
};

Case make_case(const TriangleSoup& soup, const std::string& yaml_body) {
  static int counter = 0;
  const fs::path dir = fs::temp_directory_path() / "vleo_test_dsmc";
  fs::create_directories(dir);
  const fs::path stl = dir / ("body" + std::to_string(counter++) + ".stl");
  write_stl_ascii(stl, soup);
  Case c;
  c.cfg = parse_simulation_config("geometry: {file: " + stl.string() + "}\n" + yaml_body, dir);
  c.geometry = make_geometry(soup);
  const Domain dom = make_domain(c.cfg.domain, c.geometry.host.bbox, 1e3 * c.geometry.device.surface_offset_m);
  c.inflow = InflowModel::create(dom, c.cfg.gas);
  return c;
}

// Closed 1 cm box of argon (specular walls) with a negligible specular body at the centre.
Case argon_box(double n, double T, int ppc) {
  std::ostringstream y;
  y << "simulation: {name: box, mode: dsmc, random_seed: 17}\n"
    << "domain: {box_min_m: [-0.005, -0.005, -0.005], box_max_m: [0.005, 0.005, 0.005]}\n"
    << "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
    << "gas: {number_density_m3: " << n << ", translational_temperature_K: " << T
    << ", bulk_velocity_body_mps: [-1.0e-3, 0, 0], species: {Ar: 1.0}}\n"
    << "surface: {model: maxwell, temperature_K: " << T << ", maxwell: {diffuse_fraction: 0}}\n"
    << "dsmc: {cell_size_mode: fixed, cell_size_m: 0.0025, particles_per_cell_target: " << ppc << "}\n";
  return make_case(primitives::box({-5e-5, -5e-5, -5e-5}, {5e-5, 5e-5, 5e-5}), y.str());
}

double kinetic_energy(const std::vector<Vec3>& v, double m) {
  double e = 0.0;
  for (const Vec3& x : v) e += 0.5 * m * norm2(x);
  return e;
}

double ks_normal(std::vector<double> x, double sigma) {
  std::sort(x.begin(), x.end());
  const double n = static_cast<double>(x.size());
  double d = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double f = 0.5 * std::erfc(-x[i] / (sigma * std::sqrt(2.0)));
    d = std::max({d, (static_cast<double>(i) + 1.0) / n - f, f - static_cast<double>(i) / n});
  }
  return std::sqrt(n) * d;
}

}  // namespace

TEST_CASE("Homogeneous equilibrium gas: collision rate, energy conservation, Maxwellian", "[dsmc][physics]") {
  const double n = 1e21, T = 300.0;
  Case c = argon_box(n, T, 50);
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  s.boundary = DsmcBoundary::Specular;
  s.timestep_s = 0.1 * s.tau_inf_s;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
  sim.fill_freestream();
  const SpeciesInfo ar = *find_species("Ar");
  const std::size_t n_particles = sim.num_particles();
  const double N = static_cast<double>(n_particles);
  const double E0 = kinetic_energy(sim.velocities(), ar.mass_kg);
  REQUIRE(N > 2000);

  for (int k = 0; k < 100; ++k) sim.step(false);  // let the (sigma g)_max bounds settle
  const std::uint64_t acc0 = sim.counters().collisions;
  const int steps = 2000 / VLEO_TEST_SCALE;
  for (int k = 0; k < steps; ++k) sim.step(true);
  const double K = static_cast<double>(sim.counters().collisions - acc0);

  // Expected: sum over cells of N_c (N_c - 1) / 2 * W <sigma g> dt / V_c with
  // E[N_c (N_c - 1)] = N (N - 1) / M for N particles spread over M equal cells.
  const DeviceGrid& g = sim.grid().grid;
  const double M = g.num_cells();
  const double sg = vhs_mean_sigma_g(ar.mass_kg, ar.vhs, ar.mass_kg, ar.vhs, T);
  const double expected = steps * 0.5 * N * (N - 1.0) / M * s.particle_weight * sg * s.timestep_s / g.cell_volume();
  INFO("collisions " << K << ", expected " << expected);
  CHECK(std::fabs(K - expected) < 5.0 * std::sqrt(expected) + 0.005 * expected);
  CHECK(sim.num_particles() == n_particles);  // closed box: no particle created or lost
  const std::vector<Vec3> v = sim.velocities();
  CHECK_THAT(kinetic_energy(v, ar.mass_kg), WithinRel(E0, 1e-10));

  const double sigma = std::sqrt(constants::k_B * T / ar.mass_kg);
  for (int k = 0; k < 3; ++k) {
    std::vector<double> comp;
    for (const Vec3& x : v) comp.push_back(x[k]);
    CHECK(ks_normal(comp, sigma) < 1.95);
  }
  const DsmcResult r = sim.result();
  CHECK(r.counters.collision_candidates >= r.counters.collisions);
  CHECK(r.diagnostics.acceptance_ratio > 0.0);
  CHECK(r.diagnostics.acceptance_ratio < 1.0);
  // collisions per particle per step vs. dt / tau
  CHECK_THAT(r.diagnostics.collisions_per_particle_per_step, WithinRel(s.timestep_s / s.tau_inf_s, 0.03));
}

TEST_CASE("Relaxation of an anisotropic velocity distribution to equilibrium", "[dsmc][physics]") {
  const double n = 1e21, T = 300.0;
  Case c = argon_box(n, T, 200);
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  s.boundary = DsmcBoundary::Specular;
  s.timestep_s = 0.1 * s.tau_inf_s;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
  sim.fill_freestream();
  // Replace the velocities: T_x = 2 T, T_y = T_z = T / 2 (same total energy).
  const SpeciesInfo ar = *find_species("Ar");
  const double sig = std::sqrt(constants::k_B * T / ar.mass_kg);
  std::mt19937_64 gen(11);
  std::normal_distribution<double> nd;
  std::vector<Vec3> pos = sim.positions(), vel(pos.size());
  for (auto& x : vel) x = Vec3(std::sqrt(2.0) * sig * nd(gen), sig / std::sqrt(2.0) * nd(gen), sig / std::sqrt(2.0) * nd(gen));
  sim.set_particles(pos, vel, std::vector<std::uint32_t>(pos.size(), 0u));
  const double E0 = kinetic_energy(vel, ar.mass_kg);

  for (int k = 0; k < 300; ++k) sim.step(false);  // 30 collision times
  const std::vector<Vec3> v = sim.velocities();
  CHECK_THAT(kinetic_energy(v, ar.mass_kg), WithinRel(E0, 1e-10));
  // Energy is conserved exactly, so compare the axes with their own mean
  // (the sampled initial energy fluctuates around the nominal T).
  const double N = static_cast<double>(v.size());
  double tk[3] = {0, 0, 0};
  for (const Vec3& x : v)
    for (int k = 0; k < 3; ++k) tk[k] += x[k] * x[k];
  const double t_mean = (tk[0] + tk[1] + tk[2]) / 3.0;
  for (int k = 0; k < 3; ++k) {
    INFO("axis " << k << ": T_k / T_mean = " << tk[k] / t_mean << " (initially " << (k == 0 ? 2.0 : 0.5) << ")");
    CHECK(std::fabs(tk[k] / t_mean - 1.0) < 5.0 * std::sqrt(2.0 / N));
  }
  // ... and the components are Maxwellian at that temperature.
  const double sig_mean = std::sqrt(t_mean / N);
  for (int k = 0; k < 3; ++k) {
    std::vector<double> comp;
    for (const Vec3& x : v) comp.push_back(x[k]);
    CHECK(ks_normal(comp, sig_mean) < 1.95);
  }
}

TEST_CASE("Free-molecular limit: DSMC agrees statistically with TPMC", "[dsmc][physics][gate]") {
  const std::string body =
      "simulation: {name: fm, mode: dsmc, random_seed: 23}\n"
      "domain: {margin_m: 0.2}\n"
      "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
      "gas: {number_density_m3: 1.0e12, translational_temperature_K: 900, "
      "bulk_velocity_body_mps: [-7500, 600, -900], species: {He: 1.0}}\n"
      "surface: {model: cll, temperature_K: 300, cll: {alpha_n: 0.85, sigma_t: 0.65}}\n"
      "dsmc: {particles_per_cell_target: 8, warmup_steps: 300, blocks: 20}\n"
      "tpmc: {particle_histories: 4000000, batch_size: 1000000}\n";
  Case c = make_case(read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl"), body);
  c.cfg.dsmc.sample_steps = 6000 / VLEO_TEST_SCALE;
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  CHECK(s.lambda_inf_m > 1e3);  // free-molecular
  const DsmcResult d = run_dsmc(c.cfg, c.geometry, c.inflow, s);
  CHECK(d.counters.lost_inside_body + d.counters.invalid == 0);
  CHECK(d.counters.collisions < d.counters.injected / 1000);

  SimulationConfig tcfg = c.cfg;
  tcfg.simulation.mode = SolverMode::TPMC;
  const TpmcResult t = run_tpmc(tcfg, c.geometry, c.inflow, 4000000 / VLEO_TEST_SCALE);
  auto same = [](const Estimate& a, const Estimate& b, const char* what, bool sharp) {
    INFO(what << ": DSMC " << a.value << " +- " << a.std_error << ", TPMC " << b.value << " +- " << b.std_error);
    CHECK(std::fabs(a.value - b.value) < 5.0 * std::hypot(a.std_error, b.std_error));
    // the comparison must be sharp for the large quantities
    if (sharp) CHECK(a.std_error < 0.01 * std::fabs(b.value));
  };
  same(d.estimates.C_D, t.estimates.C_D, "C_D", true);
  same(d.estimates.C_L, t.estimates.C_L, "C_L", false);
  same(d.estimates.C_Y, t.estimates.C_Y, "C_Y", false);
  same(d.estimates.heat_rate_W, t.estimates.heat_rate_W, "Q", true);
}

TEST_CASE("A monatomic DSMC run is bitwise reproducible", "[dsmc][gate]") {
  const std::string body =
      "simulation: {name: rep, mode: dsmc, random_seed: 5}\n"
      "domain: {margin_m: 0.3}\n"
      "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
      "gas: {number_density_m3: 1.0e20, translational_temperature_K: 300, "
      "bulk_velocity_body_mps: [-2000, 0, 300], species: {Ar: 1.0}}\n"
      "surface: {model: maxwell, temperature_K: 300, maxwell: {diffuse_fraction: 0.8}}\n"
      "dsmc: {cell_size_mode: fixed, cell_size_m: 0.075, particles_per_cell_target: 10, warmup_steps: 20, "
      "sample_steps: 40, blocks: 4}\n"
      "output: {save_surface_fields: true}\n";
  Case c = make_case(primitives::box({-0.3, -0.3, -0.3}, {0.3, 0.3, 0.3}), body);
  const DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  const DsmcResult a = run_dsmc(c.cfg, c.geometry, c.inflow, s);
  const DsmcResult b = run_dsmc(c.cfg, c.geometry, c.inflow, s);
  CHECK(a.counters.collisions > 1000);  // collisional
  CHECK(a.counters.collisions == b.counters.collisions);
  CHECK(a.counters.injected == b.counters.injected);
  CHECK(a.estimates.C_D.value == b.estimates.C_D.value);
  CHECK(a.estimates.heat_rate_W.value == b.estimates.heat_rate_W.value);
  CHECK(a.estimates.C_M[1].value == b.estimates.C_M[1].value);
  // surface fields sum to the global result (same events)
  REQUIRE(a.surface);
  Vec3 F;
  for (const Vec3& f : a.surface->force_N) F += f;
  CHECK(approx_equal(F, Vec3(a.estimates.force_N[0].value, a.estimates.force_N[1].value, a.estimates.force_N[2].value),
                     0.0, 1e-9));
}

TEST_CASE("Resolution diagnostics and automatic parameters", "[dsmc]") {
  const std::string body =
      "simulation: {name: res, mode: dsmc, random_seed: 1}\n"
      "domain: {margin_m: 0.5}\n"
      "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
      "gas: {number_density_m3: 1.0e21, translational_temperature_K: 300, "
      "bulk_velocity_body_mps: [-1500, 0, 0], species: {Ar: 1.0}}\n"
      "surface: {model: maxwell, temperature_K: 300, maxwell: {diffuse_fraction: 1}}\n"
      "dsmc: {cell_size_mode: fixed, cell_size_m: 0.1, particles_per_cell_target: 4, warmup_steps: 5, "
      "sample_steps: 10, blocks: 2}\n";
  Case c = make_case(primitives::box({-0.2, -0.2, -0.2}, {0.2, 0.2, 0.2}), body);
  const DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  CHECK(!s.cell_size_automatic);
  CHECK(s.timestep_automatic);
  CHECK(s.timestep_s <= 0.1 * s.tau_inf_s * (1 + 1e-12));
  const DsmcResult r = run_dsmc(c.cfg, c.geometry, c.inflow, s);
  // cells (0.1 m) are far coarser than lambda (~1.4 mm): must be reported
  CHECK(r.diagnostics.cell_over_lambda_inf > 10.0);
  CHECK(r.diagnostics.fraction_cells_dx_gt_lambda > 0.9);
  CHECK(r.diagnostics.mean_particles_per_gas_cell < 10.0);
  bool warned = false;
  for (const auto& w : r.warnings) warned = warned || w.find("local mean free path") != std::string::npos;
  CHECK(warned);
}

TEST_CASE("Block averaging of correlated series", "[statistics]") {
  // AR(1): x_t = phi x_{t-1} + e_t has tau_int = (1 + phi) / (2 (1 - phi)).
  std::mt19937_64 gen(3);
  std::normal_distribution<double> nd;
  const double phi = 0.9;
  const std::size_t N = 400000;
  std::vector<double> x(N);
  double v = 0.0;
  for (std::size_t i = 0; i < N; ++i) x[i] = v = phi * v + nd(gen);
  const SeriesStatistics s = analyze_series(x, 40);
  const double tau = (1 + phi) / (2 * (1 - phi));
  const double se_true = std::sqrt(s.variance * 2 * tau / static_cast<double>(N));
  CHECK_THAT(s.tau_int, WithinRel(tau, 0.15));
  CHECK_THAT(s.std_error_blocks, WithinRel(se_true, 0.35));  // 40 blocks: ~11 % scatter of the estimate
  CHECK(s.std_error_naive < 0.3 * s.std_error_blocks);       // naive SE ignores the correlation
  CHECK(s.blocks_long_enough);

  std::vector<double> u(100000);
  for (auto& y : u) y = nd(gen);
  const SeriesStatistics w = analyze_series(u, 20);
  CHECK(w.tau_int < 0.6);
  CHECK_THAT(w.std_error_blocks, WithinRel(w.std_error_naive, 0.5));
}
