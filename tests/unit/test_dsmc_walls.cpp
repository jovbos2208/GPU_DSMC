// Face boundaries of the closed DSMC box (dsmc.faces) and the transport
// verification cases built on them (docs/verification.md):
//   periodic / specular / wall kinematics,
//   wall pressure n k T and zero heat flux in equilibrium,
//   free-molecular Couette and Fourier flow against the exact solutions,
//   near-continuum Couette and Fourier flow: viscosity and thermal
//   conductivity of argon (VHS and VSS) against Chapman-Enskog.
//
// Tolerances (fixed before the tests were run):
//   kinematics: exact (1e-12 relative)
//   equilibrium and free-molecular cases: 5 standard errors + 0.5 % (collisions
//     at Kn ~ 1000 and the finite number of wall hits per step)
//   transport coefficients: 5 standard errors + 2 % (first-order Chapman-Enskog
//     differs from the exact VHS / VSS values by 0.2-0.4 %; the remaining margin
//     covers the finite Kn = 0.05 and dx = lambda / 3, dt = 0.2 tau)
//   wall tallies vs. the momentum and heat flux through planes in the gas: 5 combined
//     standard errors

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <random>
#include <sstream>

#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/io/ConfigParser.hpp"
#include "vleo/tally/TimeSeries.hpp"

using namespace vleo;
namespace fs = std::filesystem;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

#ifndef VLEO_TEST_SCALE
#define VLEO_TEST_SCALE 1
#endif

namespace {

constexpr double kB = constants::k_B;

struct Case {
  SimulationConfig cfg;
  Geometry geometry;
  InflowModel inflow;
};

// Argon box x [0, lx], y [-h/2, h/2], z [0, lz] with the given dsmc.faces block.
// The solver needs a body: a specular 0.2 um cube in a corner of the box.
Case argon_channel(double n, double T, double lx, double h, double lz, double cell, int ppc, const std::string& faces,
                   const std::string& model = "VHS") {
  static int counter = 0;
  const fs::path dir = fs::temp_directory_path() / "vleo_test_dsmc_walls";
  fs::create_directories(dir);
  const fs::path stl = dir / ("body" + std::to_string(counter++) + ".stl");
  const Vec3 c0(0.05 * lx, -0.45 * h, 0.05 * lz);
  const TriangleSoup soup = primitives::box(c0 - Vec3(1e-7, 1e-7, 1e-7), c0 + Vec3(1e-7, 1e-7, 1e-7));
  write_stl_ascii(stl, soup);
  std::ostringstream y;
  y.precision(17);
  y << "geometry: {file: " << stl.string() << "}\n"
    << "simulation: {name: channel, mode: dsmc, random_seed: 23}\n"
    << "domain: {box_min_m: [0, " << -0.5 * h << ", 0], box_max_m: [" << lx << ", " << 0.5 * h << ", " << lz << "]}\n"
    << "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
    << "gas: {number_density_m3: " << n << ", translational_temperature_K: " << T
    << ", bulk_velocity_body_mps: [-1.0e-3, 0, 0], species: {Ar: 1.0}}\n"
    << "surface: {model: maxwell, temperature_K: " << T << ", maxwell: {diffuse_fraction: 0}}\n"
    << "dsmc:\n  boundary: specular\n  collision_model: " << model << "\n  cell_size_mode: fixed\n  cell_size_m: " << cell
    << "\n  particles_per_cell_target: " << ppc << "\n  faces:\n"
    << faces;
  Case c;
  c.cfg = parse_simulation_config(y.str(), dir);
  c.geometry = make_geometry(soup);
  const Domain dom = make_domain(c.cfg.domain, c.geometry.host.bbox, 1e3 * c.geometry.device.surface_offset_m);
  c.inflow = InflowModel::create(dom, c.cfg.gas);
  return c;
}

std::string wall(const char* face, double T, double ux = 0.0) {
  std::ostringstream s;
  s.precision(17);
  s << "    " << face << ": {type: wall, temperature_K: " << T << ", velocity_mps: [" << ux << ", 0, 0]}\n";
  return s.str();
}

DsmcWallResult wall_result(const DsmcResult& r, const std::string& face) {
  for (const auto& w : r.walls)
    if (w.face == face) return w;
  throw std::runtime_error("no wall result for " + face);
}

// First-order Chapman-Enskog viscosity of a VHS / VSS gas (Bird 1994, eqs. 4.62
// and 4.64): mu = 5 (alpha + 1)(alpha + 2) sqrt(pi m k T_ref) /
// (4 alpha (5 - 2 omega)(7 - 2 omega) pi d^2) (T / T_ref)^omega.
double viscosity_ce(const SpeciesInfo& s, bool vss, double T) {
  const VhsParameters& p = s.vhs;
  const double a = vss ? p.alpha : 1.0;
  const double d = vss ? p.d_vss_m : p.d_ref_m;
  const double w = p.omega;
  return 5.0 * (a + 1.0) * (a + 2.0) * std::sqrt(constants::pi * s.mass_kg * kB * p.T_ref_K) /
         (4.0 * a * (5.0 - 2.0 * w) * (7.0 - 2.0 * w) * constants::pi * d * d) * std::pow(T / p.T_ref_K, w);
}
// Monatomic gas, first order: kappa = 15 k mu / (4 m) (Pr = 2/3).
double conductivity_ce(const SpeciesInfo& s, bool vss, double T) {
  return 15.0 * kB / (4.0 * s.mass_kg) * viscosity_ce(s, vss, T);
}

double c_mean(double m, double T) { return std::sqrt(8.0 * kB * T / (constants::pi * m)); }

// Least-squares slope of y(x).
double slope(const std::vector<double>& x, const std::vector<double>& y) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

double mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}
double std_error(const std::vector<double>& v) {
  const double m = mean(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / (static_cast<double>(v.size()) - 1.0) / static_cast<double>(v.size()));
}

// Profiles and fluxes in the gas, accumulated per block from host copies of the
// particles after every step: velocity and temperature in bins along y, and the
// x-momentum and energy carried through planes in the core (|y| <= 0.3 H) by the
// particles that crossed them during the step. A particle flies with the velocity
// it had after the collisions of the previous step, so the crossings are the
// exact kinetic fluxes of the scheme. (The box average sum m v_x v_y / V of the
// post-collision velocities is not: the collisions relax the stress by
// ~ dt / tau_mu per step and the free flight rebuilds it, so it is biased low by
// ~ dt / (2 tau_mu), 6 % at dt = 0.2 tau.)
struct GasProbe {
  int bins, planes = 21;
  double h, area, weight, mass, dt;
  struct Block {
    std::vector<double> n, ux, e;  // per bin: count, sum v_x, sum |v|^2
    double px = 0, en = 0;         // sum over crossings of m v_x sign(dy), m |v|^2 / 2 sign(dy)
    int profiles = 0, steps = 0;
  };
  std::vector<Block> blocks;
  std::vector<Vec3> pos0, vel0;
  std::vector<std::uint64_t> id0;
  GasProbe(int nb, double h_, double a, double w, double m, double dt_, int nblocks)
      : bins(nb), h(h_), area(a), weight(w), mass(m), dt(dt_) {
    blocks.resize(static_cast<std::size_t>(nblocks));
    for (auto& b : blocks) {
      b.n.assign(static_cast<std::size_t>(nb), 0.0);
      b.ux.assign(static_cast<std::size_t>(nb), 0.0);
      b.e.assign(static_cast<std::size_t>(nb), 0.0);
    }
  }
  void begin(const DsmcSimulation& sim) {
    pos0 = sim.positions();
    vel0 = sim.velocities();
    id0 = sim.particle_ids();
  }
  // Call after every step of the sampling phase.
  void after_step(const DsmcSimulation& sim, int block, bool profile) {
    Block& b = blocks[static_cast<std::size_t>(block)];
    const auto pos = sim.positions();
    const auto vel = sim.velocities();
    const auto ids = sim.particle_ids();
    std::uint64_t max_id = 0;
    for (auto id : ids) max_id = std::max(max_id, id);
    std::vector<std::int64_t> slot(max_id + 1, -1);
    for (std::size_t i = 0; i < ids.size(); ++i) slot[ids[i]] = static_cast<std::int64_t>(i);
    const double y_lo = -0.3 * h, dy_p = 0.6 * h / (planes - 1);
    for (std::size_t i = 0; i < id0.size(); ++i) {
      if (id0[i] > max_id || slot[id0[i]] < 0) continue;
      const double ya = pos0[i].y, yb = pos[static_cast<std::size_t>(slot[id0[i]])].y;
      // planes strictly between ya and yb (no wall is reachable from the core within one step)
      const double lo = std::min(ya, yb), hi = std::max(ya, yb);
      int j0 = static_cast<int>(std::ceil((lo - y_lo) / dy_p));
      int j1 = static_cast<int>(std::floor((hi - y_lo) / dy_p));
      j0 = std::max(j0, 0);
      j1 = std::min(j1, planes - 1);
      if (j1 < j0) continue;
      const double sgn = yb > ya ? 1.0 : -1.0;
      const double k = static_cast<double>(j1 - j0 + 1);
      b.px += k * sgn * mass * vel0[i].x;
      b.en += k * sgn * 0.5 * mass * norm2(vel0[i]);
    }
    ++b.steps;
    if (profile) {
      for (std::size_t i = 0; i < pos.size(); ++i) {
        const Vec3& v = vel[i];
        int k = static_cast<int>((pos[i].y / h + 0.5) * bins);
        k = std::clamp(k, 0, bins - 1);
        b.n[static_cast<std::size_t>(k)] += 1.0;
        b.ux[static_cast<std::size_t>(k)] += v.x;
        b.e[static_cast<std::size_t>(k)] += norm2(v);
      }
      ++b.profiles;
    }
    pos0 = pos;
    vel0 = vel;
    id0 = ids;
  }
  // Flux of x-momentum (P_xy) and energy (q_y) in +y through the core planes.
  double pxy(const Block& b) const { return weight * b.px / (area * dt * planes * b.steps); }
  double qy(const Block& b) const { return weight * b.en / (area * dt * planes * b.steps); }
  double bin_y(int k) const { return (static_cast<double>(k) + 0.5) / bins * h - 0.5 * h; }
  // Mean velocity and translational temperature of bin k in block `b`.
  double ux(const Block& b, int k) const { return b.ux[static_cast<std::size_t>(k)] / b.n[static_cast<std::size_t>(k)]; }
  double temperature(const Block& b, int k) const {
    const double nk = b.n[static_cast<std::size_t>(k)];
    const double u = ux(b, k);
    return mass * (b.e[static_cast<std::size_t>(k)] / nk - u * u) / (3.0 * kB);
  }
};

}  // namespace

TEST_CASE("Box faces: periodic re-entry, specular reflection and diffuse walls", "[dsmc][faces]") {
  // Rarefied argon (lambda ~ 13 m in a 1 cm box): no collisions in one step.
  const double h = 0.01;
  Case c = argon_channel(1e17, 300.0, h, h, h, 0.0025, 10,
                         "    x_min: periodic\n    x_max: periodic\n    z_min: specular\n" + wall("y_max", 500.0, 40.0));
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  REQUIRE(s.faces[0].type == DsmcFace::Periodic);
  REQUIRE(s.faces[1].type == DsmcFace::Periodic);
  REQUIRE(s.faces[2].type == DsmcFace::Default);  // y_min: specular box wall
  REQUIRE(s.faces[3].type == DsmcFace::Wall);
  REQUIRE(s.faces[4].type == DsmcFace::Specular);
  REQUIRE(s.faces[5].type == DsmcFace::Default);
  s.timestep_s = 1e-6;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);

  // 1: crosses x_max, 2: crosses x_min, 3: hits y_min (specular), 4..: hit the y_max wall
  std::vector<Vec3> pos{{0.0095, 0.0, 0.002}, {0.0005, 0.001, 0.007}, {0.006, -0.0045, 0.006}};
  std::vector<Vec3> vel{{1000.0, 0.0, 0.0}, {-1000.0, 0.0, 0.0}, {0.0, -1000.0, 0.0}};
  for (int i = 0; i < 200; ++i) {
    pos.emplace_back(0.001 + 4e-5 * i, 0.0045, 0.001 + 4e-5 * i);
    vel.emplace_back(0.0, 1000.0, 0.0);
  }
  sim.set_particles(pos, vel, std::vector<std::uint32_t>(pos.size(), 0u));
  sim.step(false);
  sim.compact();
  REQUIRE(sim.num_particles() == pos.size());
  const auto ids = sim.particle_ids();
  const auto p1 = sim.positions();
  const auto v1 = sim.velocities();
  double vy_wall = 0, vx_wall = 0;
  int n_wall = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto id = ids[i];
    if (id == 0) {  // 9.5 mm + 1 mm -> 0.5 mm
      CHECK_THAT(p1[i].x, WithinAbs(0.0005, 1e-12));
      CHECK(v1[i].x == 1000.0);
    } else if (id == 1) {  // 0.5 mm - 1 mm -> 9.5 mm
      CHECK_THAT(p1[i].x, WithinAbs(0.0095, 1e-12));
      CHECK(v1[i].x == -1000.0);
    } else if (id == 2) {  // specular at y = -5 mm: -4.5 -> -5 -> -4.5 mm
      CHECK_THAT(p1[i].y, WithinAbs(-0.0045, 1e-12));
      CHECK(v1[i].y == 1000.0);
    } else {
      CHECK(v1[i].y < 0.0);  // re-emitted into the gas
      CHECK(p1[i].y <= 0.005);
      vy_wall += v1[i].y;
      vx_wall += v1[i].x;
      ++n_wall;
    }
  }
  REQUIRE(n_wall == 200);
  // Diffuse at 500 K moving with 40 m/s: <v_x> = 40 m/s, <v_y> = -sqrt(pi k T / (2 m)).
  const double m = find_species("Ar")->mass_kg;
  const double s1 = std::sqrt(kB * 500.0 / m);
  CHECK(std::fabs(vx_wall / n_wall - 40.0) < 5.0 * s1 / std::sqrt(200.0));
  CHECK(std::fabs(-vy_wall / n_wall - std::sqrt(constants::pi * kB * 500.0 / (2.0 * m))) <
        5.0 * 0.6 * s1 / std::sqrt(200.0));
  CHECK(sim.counters().exited == 0);
  CHECK(sim.counters().invalid == 0);
}

TEST_CASE("Rejected: face boundaries", "[dsmc][faces][config]") {
  const double h = 0.01;
  CHECK_THROWS(argon_channel(1e17, 300.0, h, h, h, 0.0025, 10, "    x_min: periodic\n"));  // unpaired
  CHECK_THROWS(argon_channel(1e17, 300.0, h, h, h, 0.0025, 10, "    y_min: wall\n"));      // wall needs a map
  CHECK_THROWS(argon_channel(1e17, 300.0, h, h, h, 0.0025, 10,
                             "    y_min: {type: wall, temperature_K: 300, velocity_mps: [0, 5, 0]}\n"));  // normal
  CHECK_THROWS(argon_channel(1e17, 300.0, h, h, h, 0.0025, 10,
                             "    y_min: {type: periodic, temperature_K: 300}\n"));
  CHECK_THROWS(argon_channel(1e17, 300.0, h, h, h, 0.0025, 10, "    top: specular\n"));
}

TEST_CASE("Diffuse walls in equilibrium: pressure n k T, no shear, no heat flux", "[dsmc][faces][physics]") {
  const double T = 300.0, n = 1e21, L = 0.01;  // lambda ~ 1.4 mm
  std::string faces;
  for (const char* f : {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}) faces += wall(f, T);
  Case c = argon_channel(n, T, L, L, L, 0.0025, 100, faces);
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  s.timestep_s = 0.2 * s.tau_inf_s;
  s.blocks = 10;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
  sim.fill_freestream();
  for (int k = 0; k < 200; ++k) sim.step(false);
  for (int k = 0; k < 4000 / VLEO_TEST_SCALE; ++k) sim.step(true);
  const DsmcResult r = sim.result();
  REQUIRE(r.walls.size() == 6);
  const double n_box = static_cast<double>(sim.num_particles()) * s.particle_weight / (L * L * L);
  const double p = n_box * kB * T;
  const double m = find_species("Ar")->mass_kg;
  const double heat_scale = 0.25 * n_box * c_mean(m, T) * 2.0 * kB * T * L * L;  // one-way energy flux
  for (const DsmcWallResult& w : r.walls) {
    const int axis = w.face[0] - 'x';
    const double sign = w.face[3] == 'a' ? 1.0 : -1.0;  // gas pushes every wall outward
    INFO(w.face << ": F = (" << w.force_N[0].value << ", " << w.force_N[1].value << ", " << w.force_N[2].value
                << ") N, p A = " << p * w.area_m2 << " N, Q = " << w.heat_rate_W.value << " W");
    const Estimate& fn = w.force_N[static_cast<std::size_t>(axis)];
    CHECK(std::fabs(sign * fn.value - p * w.area_m2) < 5.0 * fn.std_error + 0.005 * p * w.area_m2);
    for (int k = 1; k < 3; ++k) {
      const Estimate& ft = w.force_N[static_cast<std::size_t>((axis + k) % 3)];
      CHECK(std::fabs(ft.value) < 5.0 * ft.std_error + 0.005 * p * w.area_m2);
    }
    CHECK(std::fabs(w.heat_rate_W.value) < 5.0 * w.heat_rate_W.std_error + 0.005 * heat_scale);
  }
}

TEST_CASE("Free-molecular Couette and Fourier flow between diffuse walls", "[dsmc][faces][physics]") {
  // Kn ~ 1300: wall-to-wall flight without collisions.
  const double n = 1e17, h = 0.01, lx = 0.005;
  const SpeciesInfo ar = *find_species("Ar");
  const double m = ar.mass_kg;
  const int steps = 6000 / VLEO_TEST_SCALE;

  SECTION("Couette: walls at -U and +U, T = 273 K") {
    const double T = 273.0, U = 100.0;
    Case c = argon_channel(n, T, lx, h, lx, 0.001, 200,
                           "    x_min: periodic\n    x_max: periodic\n    z_min: periodic\n    z_max: periodic\n" +
                               wall("y_min", T, -U) + wall("y_max", T, U));
    DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
    s.timestep_s = 2e-6;
    s.blocks = 10;
    DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
    sim.fill_freestream();
    for (int k = 0; k < 200; ++k) sim.step(false);
    for (int k = 0; k < steps; ++k) sim.step(true);
    const DsmcResult r = sim.result();
    const double nb = static_cast<double>(sim.num_particles()) * s.particle_weight / (lx * h * lx);
    const double A = lx * lx;
    // Exact: one-way flux n c_mean / 4, each hit carries 2 m U in x and 2 m U^2 of energy (wall frame).
    const double tau = 0.5 * nb * m * c_mean(m, T) * U;
    const double q = 0.5 * nb * m * c_mean(m, T) * U * U;
    for (const char* f : {"y_min", "y_max"}) {
      const DsmcWallResult w = wall_result(r, f);
      const double dir = f[3] == 'a' ? -1.0 : 1.0;  // upper wall is dragged to -x, lower to +x
      INFO(f << ": F_x = " << w.force_N[0].value << " +- " << w.force_N[0].std_error << ", exact " << dir * tau * A
             << "; Q = " << w.heat_rate_W.value << " +- " << w.heat_rate_W.std_error << ", exact " << q * A);
      CHECK(std::fabs(w.force_N[0].value - dir * tau * A) < 5.0 * w.force_N[0].std_error + 0.005 * tau * A);
      CHECK(std::fabs(w.heat_rate_W.value - q * A) < 5.0 * w.heat_rate_W.std_error + 0.005 * q * A);
    }
    CHECK(r.counters.collisions < sim.counters().steps * 50);  // essentially collisionless
  }

  SECTION("Fourier: walls at 400 K and 250 K") {
    const double T1 = 400.0, T2 = 250.0;
    Case c = argon_channel(n, 0.5 * (T1 + T2), lx, h, lx, 0.001, 200, wall("y_min", T1) + wall("y_max", T2));
    DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
    s.timestep_s = 2e-6;
    s.blocks = 10;
    DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
    sim.fill_freestream();
    for (int k = 0; k < 300; ++k) sim.step(false);
    for (int k = 0; k < steps; ++k) sim.step(true);
    const DsmcResult r = sim.result();
    const double nb = static_cast<double>(sim.num_particles()) * s.particle_weight / (lx * h * lx);
    const double A = lx * lx;
    // Exact: half-Maxwellians from both walls with equal fluxes G; n = 2 G (1/c1 + 1/c2),
    // q = 2 k G (T1 - T2).
    const double c1 = c_mean(m, T1), c2 = c_mean(m, T2);
    const double G = nb * c1 * c2 / (2.0 * (c1 + c2));
    const double q = 2.0 * kB * G * (T1 - T2);
    const DsmcWallResult hot = wall_result(r, "y_min");
    const DsmcWallResult cold = wall_result(r, "y_max");
    INFO("Q hot " << hot.heat_rate_W.value << ", cold " << cold.heat_rate_W.value << ", exact +-" << q * A);
    CHECK(std::fabs(cold.heat_rate_W.value - q * A) < 5.0 * cold.heat_rate_W.std_error + 0.005 * q * A);
    CHECK(std::fabs(hot.heat_rate_W.value + q * A) < 5.0 * hot.heat_rate_W.std_error + 0.005 * q * A);
  }
}

TEST_CASE("Near-continuum Couette and Fourier flow: viscosity and conductivity of argon", "[dsmc][faces][physics][transport][.]") {
  // Kn = lambda / H = 0.05, dx = lambda / 3, dt = 0.2 tau. The shear stress and
  // the heat flux come from the wall tallies (results.json walls), the gradients
  // from a fit in the core |y| < 0.3 H, outside the Knudsen layers. The fluxes
  // through planes in the core must equal the wall values (conservation).
  // Started from the linear profiles, warm-up ~ 5 diffusion times.
  const SpeciesInfo ar = *find_species("Ar");
  const double m = ar.mass_kg;
  const bool vss = GENERATE(false, true);
  const char* model = vss ? "VSS" : "VHS";
  const double T0 = 273.0, n = 1e21;
  // lambda_inf of the configuration (resolve with a dummy box first)
  Case probe = argon_channel(n, T0, 0.01, 0.01, 0.01, 0.001, 10, "    x_min: specular\n", model);
  const double lambda = resolve_dsmc_settings(probe.cfg, probe.inflow.domain).lambda_inf_m;
  const double h = 20.0 * lambda, dx = lambda / 3.0, lx = 2.0 * dx;
  const int nblocks = 10;
  const int bins = 60;
  const int blocks_steps = 2000 / VLEO_TEST_SCALE;  // per block

  SECTION("Couette: walls at -100 and +100 m/s") {
    const double U = 100.0;
    Case c = argon_channel(n, T0, lx, h, lx, dx, 100,
                           "    x_min: periodic\n    x_max: periodic\n    z_min: periodic\n    z_max: periodic\n" +
                               wall("y_min", T0, -U) + wall("y_max", T0, U),
                           model);
    DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
    s.timestep_s = 0.2 * s.tau_inf_s;
    s.blocks = nblocks;
    DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
    // linear velocity profile, uniform density and temperature
    const auto N = static_cast<std::size_t>(std::llround(n * lx * h * lx / s.particle_weight));
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> nor(0.0, std::sqrt(kB * T0 / m));
    std::vector<Vec3> pos, vel;
    for (std::size_t i = 0; i < N; ++i) {
      const double y = (uni(rng) - 0.5) * h;
      pos.emplace_back(uni(rng) * lx, y, uni(rng) * lx);
      vel.emplace_back(0.9 * U * 2.0 * y / h + nor(rng), nor(rng), nor(rng));
    }
    sim.set_particles(pos, vel, std::vector<std::uint32_t>(N, 0u));
    for (int k = 0; k < 2000; ++k) sim.step(false);
    GasProbe g(bins, h, lx * lx, s.particle_weight, m, s.timestep_s, nblocks);
    g.begin(sim);
    for (int b = 0; b < nblocks; ++b)
      for (int k = 0; k < blocks_steps; ++k) {
        sim.step(true);
        g.after_step(sim, b, k % 2 == 0);
      }
    std::vector<double> dudy_b, pxy_b, T_b;
    for (const auto& bl : g.blocks) {
      std::vector<double> ys, us, Ts;
      for (int k = 0; k < bins; ++k) {
        const double y = g.bin_y(k);
        if (std::fabs(y) > 0.3 * h) continue;
        ys.push_back(y);
        us.push_back(g.ux(bl, k));
        Ts.push_back(g.temperature(bl, k));
      }
      dudy_b.push_back(slope(ys, us));
      pxy_b.push_back(g.pxy(bl));
      T_b.push_back(mean(Ts));
    }
    // shear stress at the upper wall: tau = -F_x / A = mu du/dy
    const DsmcResult r = sim.result();
    const DsmcWallResult top = wall_result(r, "y_max");
    const double A = lx * lx;
    const double tau = -top.force_N[0].value / A, tau_se = top.force_N[0].std_error / A;
    const double dudy = mean(dudy_b), dudy_se = std_error(dudy_b);
    const double mu = tau / dudy;
    const double se = mu * std::hypot(tau_se / tau, dudy_se / dudy);
    const double mu_ce = viscosity_ce(ar, vss, mean(T_b));
    INFO(model << ": mu = " << mu << " +- " << se << " Pa s, Chapman-Enskog " << mu_ce << " ("
               << 100.0 * (mu / mu_ce - 1.0) << " %), core T " << mean(T_b) << " K, Kn " << lambda / h);
    CHECK(std::fabs(mu - mu_ce) < 5.0 * se + 0.02 * mu_ce);
    // momentum conservation: the flux through the core planes equals the wall shear
    const double pxy = mean(pxy_b), pxy_se = std_error(pxy_b);
    INFO("wall shear " << tau << " +- " << tau_se << " Pa, plane flux " << -pxy << " +- " << pxy_se << " Pa");
    CHECK(std::fabs(tau + pxy) < 5.0 * std::hypot(tau_se, pxy_se));
  }

  SECTION("Fourier: walls at 323 K and 223 K") {
    const double T1 = 323.0, T2 = 223.0;
    Case c = argon_channel(n, T0, lx, h, lx, dx, 100, wall("y_min", T1) + wall("y_max", T2), model);
    DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
    s.timestep_s = 0.2 * s.tau_inf_s;
    s.blocks = nblocks;
    DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
    // linear temperature, density ~ 1 / T (uniform pressure)
    const auto N = static_cast<std::size_t>(std::llround(n * lx * h * lx / s.particle_weight));
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> nor(0.0, 1.0);
    auto T_lin = [&](double y) { return T1 + (T2 - T1) * (y / h + 0.5); };
    std::vector<Vec3> pos, vel;
    while (pos.size() < N) {
      const double y = (uni(rng) - 0.5) * h;
      if (uni(rng) > T2 / T_lin(y)) continue;
      const double sd = std::sqrt(kB * T_lin(y) / m);
      pos.emplace_back(uni(rng) * lx, y, uni(rng) * lx);
      vel.emplace_back(sd * nor(rng), sd * nor(rng), sd * nor(rng));
    }
    sim.set_particles(pos, vel, std::vector<std::uint32_t>(N, 0u));
    for (int k = 0; k < 2000; ++k) sim.step(false);
    GasProbe g(bins, h, lx * lx, s.particle_weight, m, s.timestep_s, nblocks);
    g.begin(sim);
    for (int b = 0; b < nblocks; ++b)
      for (int k = 0; k < blocks_steps; ++k) {
        sim.step(true);
        g.after_step(sim, b, k % 2 == 0);
      }
    // kappa ~ T^omega: kappa dT/dy = kappa_ref T_ref^-omega / (1 + omega) d(T^(1 + omega))/dy
    const double w = ar.vhs.omega;
    const double kref = conductivity_ce(ar, vss, ar.vhs.T_ref_K);
    std::vector<double> qce_b, qy_b;
    for (const auto& bl : g.blocks) {
      std::vector<double> ys, th;
      for (int k = 0; k < bins; ++k) {
        const double y = g.bin_y(k);
        if (std::fabs(y) > 0.3 * h) continue;
        ys.push_back(y);
        th.push_back(std::pow(g.temperature(bl, k), 1.0 + w));
      }
      qce_b.push_back(-kref * std::pow(ar.vhs.T_ref_K, -w) / (1.0 + w) * slope(ys, th));
      qy_b.push_back(g.qy(bl));
    }
    // heat flux into the cold wall
    const DsmcResult r = sim.result();
    const DsmcWallResult cold = wall_result(r, "y_max");
    const double A = lx * lx;
    const double q = cold.heat_rate_W.value / A, q_se = cold.heat_rate_W.std_error / A;
    const double qce = mean(qce_b), qce_se = std_error(qce_b);
    const double ratio = q / qce;
    const double se = ratio * std::hypot(q_se / q, qce_se / qce);
    INFO(model << ": kappa / kappa_CE = " << ratio << " +- " << se << ", Kn " << lambda / h);
    CHECK(std::fabs(ratio - 1.0) < 5.0 * se + 0.02);
    // energy conservation: the flux through the core planes equals the wall heat flux
    const double qy = mean(qy_b), qy_se = std_error(qy_b);
    INFO("wall heat flux " << q << " +- " << q_se << " W/m2, plane flux " << qy << " +- " << qy_se << " W/m2");
    CHECK(std::fabs(q - qy) < 5.0 * std::hypot(q_se, qy_se));
  }
}
