// M4.2-M4.4: DSMC gas mixtures (species-pair collisions, mixture injection,
// species diagnostics), homogeneous O/N/He reference cases.
//
// Tolerances (fixed before the tests were run, docs/dsmc.md section 7):
//   energy conservation in a closed specular box: 1e-10 relative
//   pair collision counts vs. equilibrium theory: 5 sqrt(K) + 1 %
//   species temperatures, injection shares: 5 standard errors

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <random>
#include <sstream>

#include "vleo/core/Species.hpp"
#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/io/ConfigParser.hpp"

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
  const fs::path dir = fs::temp_directory_path() / "vleo_test_mixture";
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

// Closed 1 cm box (specular walls) with a negligible specular body.
Case mixture_box(double n, double T, int ppc, const std::string& species) {
  std::ostringstream y;
  y << "simulation: {name: mix, mode: dsmc, random_seed: 29}\n"
    << "domain: {box_min_m: [-0.005, -0.005, -0.005], box_max_m: [0.005, 0.005, 0.005]}\n"
    << "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
    << "gas: {number_density_m3: " << n << ", translational_temperature_K: " << T
    << ", bulk_velocity_body_mps: [-1.0e-3, 0, 0], species: {" << species << "}}\n"
    << "surface: {model: maxwell, temperature_K: " << T << ", maxwell: {diffuse_fraction: 0}}\n"
    << "dsmc: {cell_size_mode: fixed, cell_size_m: 0.0025, particles_per_cell_target: " << ppc << "}\n";
  return make_case(primitives::box({-5e-5, -5e-5, -5e-5}, {5e-5, 5e-5, 5e-5}), y.str());
}

double kinetic_energy(const std::vector<Vec3>& v, const std::vector<std::uint32_t>& sp, const SimulationConfig& cfg) {
  double e = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) e += 0.5 * cfg.gas.species[sp[i]].mass_kg * norm2(v[i]);
  return e;
}

// Temperature of species q from its particles (3/2 k T = <1/2 m |v|^2>, gas at rest).
double species_temperature(const std::vector<Vec3>& v, const std::vector<std::uint32_t>& sp, std::uint32_t q,
                           double m, double& n_out) {
  double e = 0.0, n = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i)
    if (sp[i] == q) {
      e += 0.5 * m * norm2(v[i]);
      n += 1.0;
    }
  n_out = n;
  return 2.0 * e / (3.0 * n * constants::k_B);
}

}  // namespace

TEST_CASE("O/N/He mixture at equilibrium: pair collision rates, energy, common temperature", "[mixture][physics]") {
  const double n = 1e21, T = 400.0;
  Case c = mixture_box(n, T, 80, "O: 0.7, N: 0.2, He: 0.1");
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  s.boundary = DsmcBoundary::Specular;
  s.timestep_s = 0.1 * s.tau_inf_s;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
  sim.fill_freestream();
  const std::vector<std::uint32_t> sp = sim.species();
  const double E0 = kinetic_energy(sim.velocities(), sp, c.cfg);
  double Ns[3] = {0, 0, 0};
  for (auto q : sp) Ns[q] += 1.0;
  REQUIRE(Ns[2] > 200);

  for (int k = 0; k < 100; ++k) sim.step(false);
  std::uint64_t K0[3][3];
  for (int p = 0; p < 3; ++p)
    for (int q = p; q < 3; ++q) K0[p][q] = sim.pair_collisions(p, q);
  const int steps = 2000 / VLEO_TEST_SCALE;
  for (int k = 0; k < steps; ++k) sim.step(true);

  // Expected accepted collisions of pair (p, q) per step: sum over cells of
  // N_p N_q (p != q) or N_p (N_p - 1) / 2 (p == q) times W <sigma g>_pq dt / V_c,
  // with the cell sums of N_p N_q equal to N_p N_q / M (uniform gas, M cells).
  const DeviceGrid& g = sim.grid().grid;
  const double M = g.num_cells();
  for (int p = 0; p < 3; ++p) {
    for (int q = p; q < 3; ++q) {
      const auto& a = c.cfg.gas.species[static_cast<std::size_t>(p)];
      const auto& b = c.cfg.gas.species[static_cast<std::size_t>(q)];
      const double sg = vhs_mean_sigma_g(a.mass_kg, a.vhs, b.mass_kg, b.vhs, T);
      const double pairs = p == q ? 0.5 * Ns[p] * (Ns[p] - 1.0) / M : Ns[p] * Ns[q] / M;
      const double expected = steps * pairs * s.particle_weight * sg * s.timestep_s / g.cell_volume();
      const double K = static_cast<double>(sim.pair_collisions(p, q) - K0[p][q]);
      INFO(a.name << "-" << b.name << ": collisions " << K << ", expected " << expected);
      CHECK(std::fabs(K - expected) < 5.0 * std::sqrt(expected) + 0.01 * expected);
    }
  }
  const std::vector<Vec3> v = sim.velocities();
  CHECK_THAT(kinetic_energy(v, sp, c.cfg), WithinRel(E0, 1e-10));
  for (std::uint32_t q = 0; q < 3; ++q) {
    double nq = 0.0;
    const double Tq = species_temperature(v, sp, q, c.cfg.gas.species[q].mass_kg, nq);
    INFO("species " << q << ": T = " << Tq);
    CHECK(std::fabs(Tq / T - 1.0) < 5.0 * std::sqrt(2.0 / (3.0 * nq)));
  }
  const DsmcResult r = sim.result();
  REQUIRE(r.species.size() == 3);
  CHECK(r.species[0].name == "O");
  CHECK(r.species[2].collisions_with[0] == sim.pair_collisions(0, 2));
  CHECK_THAT(r.species[0].mean_particles, WithinRel(Ns[0], 1e-12));  // closed box: constant
}

TEST_CASE("Two-temperature He/Ar mixture relaxes to the common temperature", "[mixture][physics]") {
  const double n = 1e21;
  Case c = mixture_box(n, 300.0, 120, "He: 0.5, Ar: 0.5");
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  s.boundary = DsmcBoundary::Specular;
  s.timestep_s = 0.1 * s.tau_inf_s;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
  sim.fill_freestream();
  // He at 1200 K, Ar at 300 K (equal numbers in expectation).
  std::vector<Vec3> pos = sim.positions();
  std::vector<std::uint32_t> sp = sim.species();
  std::vector<Vec3> vel(pos.size());
  std::mt19937_64 gen(5);
  std::normal_distribution<double> nd;
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double T0 = sp[i] == 0 ? 1200.0 : 300.0;
    const double sig = std::sqrt(constants::k_B * T0 / c.cfg.gas.species[sp[i]].mass_kg);
    vel[i] = sig * Vec3(nd(gen), nd(gen), nd(gen));
  }
  sim.set_particles(pos, vel, sp);
  const double E0 = kinetic_energy(vel, sp, c.cfg);
  for (int k = 0; k < 600; ++k) sim.step(false);  // 60 mean collision times
  const std::vector<Vec3> v = sim.velocities();
  CHECK_THAT(kinetic_energy(v, sp, c.cfg), WithinRel(E0, 1e-10));
  double n_he = 0.0, n_ar = 0.0;
  const double T_he = species_temperature(v, sp, 0, c.cfg.gas.species[0].mass_kg, n_he);
  const double T_ar = species_temperature(v, sp, 1, c.cfg.gas.species[1].mass_kg, n_ar);
  const double T_eq = 2.0 * E0 / (3.0 * (n_he + n_ar) * constants::k_B);
  INFO("T_He = " << T_he << ", T_Ar = " << T_ar << ", T_eq = " << T_eq);
  CHECK(std::fabs(T_he / T_eq - 1.0) < 5.0 * std::sqrt(2.0 / (3.0 * n_he)));
  CHECK(std::fabs(T_ar / T_eq - 1.0) < 5.0 * std::sqrt(2.0 / (3.0 * n_ar)));
}

TEST_CASE("Mixture injection follows the species flux shares", "[mixture]") {
  // Open domain, free-molecular density: the injected numbers per species
  // must follow sum over faces of Gamma_s A.
  const std::string body =
      "simulation: {name: inj, mode: dsmc, random_seed: 41}\n"
      "domain: {margin_m: 0.3}\n"
      "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
      "gas: {number_density_m3: 1.0e12, translational_temperature_K: 900, "
      "bulk_velocity_body_mps: [-7500, 300, 0], species: {O: 0.75, N: 0.2, He: 0.05}}\n"
      "surface: {model: maxwell, temperature_K: 300, maxwell: {diffuse_fraction: 1}}\n"
      "dsmc: {particles_per_cell_target: 10, warmup_steps: 50, sample_steps: 400, blocks: 10}\n";
  Case c = make_case(primitives::box({-0.25, -0.25, -0.25}, {0.25, 0.25, 0.25}), body);
  const DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  const DsmcResult r = run_dsmc(c.cfg, c.geometry, c.inflow, s);
  std::vector<double> rate(3, 0.0);
  for (const auto& ch : c.inflow.channels) rate[ch.species] += ch.rate_per_s;
  double total_rate = 0.0, total_inj = 0.0;
  for (int q = 0; q < 3; ++q) {
    total_rate += rate[static_cast<std::size_t>(q)];
    total_inj += static_cast<double>(r.species[static_cast<std::size_t>(q)].injected);
  }
  for (int q = 0; q < 3; ++q) {
    const double p = rate[static_cast<std::size_t>(q)] / total_rate;
    const double obs = static_cast<double>(r.species[static_cast<std::size_t>(q)].injected) / total_inj;
    INFO(r.species[static_cast<std::size_t>(q)].name << ": injected share " << obs << ", expected " << p);
    CHECK(std::fabs(obs - p) < 5.0 * std::sqrt(p * (1 - p) / total_inj));
  }
  // species contributions add up to the totals
  double Q = 0.0;
  Vec3 F;
  for (const auto& sd : r.species) {
    Q += sd.heat_rate_W;
    F += sd.force_N;
  }
  CHECK_THAT(Q, WithinRel(r.estimates.heat_rate_W.value, 1e-9));
  CHECK_THAT(F.x, WithinRel(r.estimates.force_N[0].value, 1e-9));
  CHECK(r.counters.lost_inside_body + r.counters.invalid == 0);
}
