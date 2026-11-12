// M2.5-M2.11, M2.13-M2.18: TPMC solver physics and statistics.
//
// Tolerances (fixed before the tests were run, docs/tpmc.md section 7):
//   statistical agreement with an analytic value: |x - x_ref| < 5 SE
//   two independent simulations:                  |a - b| < 5 sqrt(SE_a^2 + SE_b^2)
//   identities that are exact up to rounding:     1e-12 relative to the scale

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/io/ConfigParser.hpp"
#include "vleo/tpmc/Results.hpp"
#include "vleo/tpmc/Tpmc.hpp"

using namespace vleo;
namespace fs = std::filesystem;

namespace {

constexpr double kSqrtPi = 1.77245385090551602730;

std::string vec(const Vec3& v) {
  std::ostringstream os;
  os.precision(17);
  os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return os.str();
}

struct Problem {
  SimulationConfig cfg;
  Geometry geometry;
  InflowModel inflow;
};

// Builds a complete problem from a mesh and YAML fragments.
Problem make_problem(const TriangleSoup& soup, const std::string& surface_yaml, const Vec3& u, double T_gas,
                     const std::string& species_yaml = "    O: 1.0\n", const std::string& extra = "",
                     bool require_closed = true) {
  static int counter = 0;
  const fs::path dir = fs::temp_directory_path() / "vleo_test_tpmc";
  fs::create_directories(dir);
  const fs::path stl = dir / ("body" + std::to_string(counter++) + ".stl");
  write_stl_ascii(stl, soup);
  std::ostringstream y;
  y << "simulation:\n  name: test\n  mode: tpmc\n  random_seed: 4711\n"
    << "geometry:\n  file: " << stl.string() << "\n  require_closed: " << (require_closed ? "true" : "false") << "\n"
    << "domain:\n  margin_m: 0.1\n"
    << "reference:\n  area_m2: 1.0\n  length_m: 1.0\n  moment_center_m: [0, 0, 0]\n  lift_axis_body: [1, 0, 1]\n"  // projected normal to the flow
    << "gas:\n  number_density_m3: 1.0e14\n  translational_temperature_K: " << T_gas
    << "\n  bulk_velocity_body_mps: " << vec(u) << "\n  species:\n"
    << species_yaml << surface_yaml << extra;
  Problem p;
  p.cfg = parse_simulation_config(y.str(), dir);
  MeshOptions mo;
  mo.require_closed = require_closed;
  p.geometry = make_geometry(soup, mo);
  const Domain dom = make_domain(p.cfg.domain, p.geometry.host.bbox, 1e3 * p.geometry.device.surface_offset_m);
  p.inflow = InflowModel::create(dom, p.cfg.gas);
  return p;
}

std::string maxwell_yaml(double fd, double Tw = 300.0) {
  std::ostringstream os;
  os << "surface:\n  model: maxwell\n  temperature_K: " << Tw << "\n  maxwell:\n    diffuse_fraction: " << fd << "\n";
  return os.str();
}

std::string cll_yaml(double an, double st, double Tw = 300.0) {
  std::ostringstream os;
  os << "surface:\n  model: cll\n  temperature_K: " << Tw << "\n  cll:\n    alpha_n: " << an << "\n    sigma_t: " << st
     << "\n";
  return os.str();
}

// Debug builds (serial, unoptimised) run 1/VLEO_TEST_SCALE of the histories;
// the 5-sigma criteria hold for any sample size.
#ifndef VLEO_TEST_SCALE
#define VLEO_TEST_SCALE 1
#endif

std::string tpmc_yaml(std::uint64_t histories, std::uint64_t batch, bool surface_fields = false) {
  histories /= VLEO_TEST_SCALE;
  batch /= VLEO_TEST_SCALE;
  std::ostringstream os;
  os << "tpmc:\n  particle_histories: " << histories << "\n  batch_size: " << batch << "\n  max_reflections: 1000\n";
  if (surface_fields) os << "output:\n  save_surface_fields: true\n";
  return os.str();
}

// |x - ref| <= 5 SE + floor. The floor (1e-12 of the problem scale) only
// matters for quantities that are exactly zero in every history (e.g. the
// tangential force on a specular plate), where SE = 0.
void check_stat(const Estimate& e, double ref, const char* what, double floor = 0.0) {
  INFO(what << ": simulated " << e.value << " +- " << e.std_error << ", reference " << ref);
  CHECK(std::fabs(e.value - ref) <= 5.0 * e.std_error + floor);
}

void check_same(const Estimate& a, const Estimate& b, const char* what) {
  INFO(what << ": " << a.value << " +- " << a.std_error << " vs " << b.value << " +- " << b.std_error);
  CHECK(std::fabs(a.value - b.value) < 5.0 * std::sqrt(a.std_error * a.std_error + b.std_error * b.std_error));
}

// ---------------------------------------------------------------------------
// Free-molecular flat plate (one side with outward normal n), per species.
// Derivation in docs/tpmc.md section 6. Returns the force on the plate side
// (momentum transferred to the body) and the heat rate, per unit area.
// ---------------------------------------------------------------------------
struct SideFlux {
  double gamma;  // number flux
  double P_i;    // incident normal momentum flux
  Vec3 tau_i;    // incident tangential momentum flux (vector)
  double E_i;    // incident energy flux
  double Ecn2;   // n c^3 J3 / sqrt(pi) = Gamma <c_n^2>_flux
  double ut2;    // |u_t|^2
  double c2;     // c^2 = 2 k T / m
};

SideFlux side_flux(const Vec3& n, const Vec3& u, double nd, double T, double m) {
  SideFlux f;
  const double c = std::sqrt(2 * constants::k_B * T / m);
  const double s = dot(u, -n) / c;
  const Vec3 ut = reject_from_unit(u, n);
  const double A = 0.5 * kSqrtPi * std::erfc(-s), B = 0.5 * std::exp(-s * s);
  const double J2 = s * B + 0.5 * A * (1 + 2 * s * s);
  const double J3 = B * (s * s + 1) + A * s * (s * s + 1.5);
  f.gamma = number_flux(nd, c, s);
  f.P_i = nd * m * c * c / kSqrtPi * J2;
  f.tau_i = m * f.gamma * ut;
  f.Ecn2 = nd * c * c * c / kSqrtPi * J3;
  f.ut2 = norm2(ut);
  f.c2 = c * c;
  f.E_i = 0.5 * m * (f.Ecn2 + f.gamma * (f.ut2 + c * c));
  return f;
}

struct PlateResult {
  Vec3 force;
  double heat;
};

// Maxwell with diffuse fraction fd at wall temperature Tw, per unit area.
PlateResult plate_maxwell(const Vec3& n, const Vec3& u, double nd, double T, double m, double fd, double Tw) {
  const SideFlux f = side_flux(n, u, nd, T, m);
  const double sw = std::sqrt(constants::k_B * Tw / m);
  // diffuse: re-emitted normal momentum flux m Gamma <c_rn> = m Gamma sw sqrt(pi/2), energy Gamma 2 k Tw
  const Vec3 F_diff = -(f.P_i + m * f.gamma * sw * std::sqrt(constants::pi / 2)) * n + f.tau_i;
  const double Q_diff = f.E_i - f.gamma * 2 * constants::k_B * Tw;
  const Vec3 F_spec = -2.0 * f.P_i * n;  // tangential momentum conserved, normal reversed
  return {fd * F_diff + (1 - fd) * F_spec, fd * Q_diff};
}

// CLL: tangential force and heat (the mean normal re-emission speed has no closed form).
PlateResult plate_cll(const Vec3& n, const Vec3& u, double nd, double T, double m, double an, double st, double Tw) {
  const SideFlux f = side_flux(n, u, nd, T, m);
  const double sw2 = constants::k_B * Tw / m;
  const double E_r = 0.5 * m *
                     ((1 - st) * (1 - st) * f.gamma * (f.ut2 + f.c2) + 2 * sw2 * st * (2 - st) * f.gamma +
                      (1 - an) * f.Ecn2 + 2 * an * sw2 * f.gamma);
  return {st * f.tau_i, f.E_i - E_r};  // force: tangential part only
}

TriangleSoup unit_plate() {
  TriangleSoup s;
  primitives::add_rectangle(s, {-0.5, -0.5, 0}, {1, 0, 0}, {0, 1, 0});  // z = 0, normal +z, area 1
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST_CASE("Flat plate: Maxwell against free-molecular theory", "[tpmc][physics][plate]") {
  const auto [fd, alpha_deg] = GENERATE(table<double, double>({{1.0, 30.0}, {0.0, 30.0}, {0.6, 60.0}, {1.0, 90.0}}));
  INFO("f_d = " << fd << ", angle of attack " << alpha_deg << " deg");
  const double a = alpha_deg * constants::pi / 180.0;
  const double V = 7500.0, T = 900.0, Tw = 300.0;
  const Vec3 u(-V * std::cos(a), 0.0, -V * std::sin(a));  // hits the +z side
  Problem p = make_problem(unit_plate(), maxwell_yaml(fd, Tw), u, T, "    O: 0.7\n    He: 0.3\n",
                           tpmc_yaml(4000000, 1000000), false);
  const TpmcResult r = run_tpmc(p.cfg, p.geometry, p.inflow);
  REQUIRE(r.counters.balanced());
  REQUIRE(r.counters.lost_inside_body + r.counters.invalid == 0);

  Vec3 F;
  double Q = 0;
  for (const auto& sp : p.cfg.gas.species) {
    const double nd = p.cfg.gas.number_density_m3 * sp.mole_fraction;
    for (const Vec3& n : {Vec3(0, 0, 1), Vec3(0, 0, -1)}) {
      const PlateResult pr = plate_maxwell(n, u, nd, T, sp.mass_kg, fd, Tw);
      F += pr.force;
      Q += pr.heat;
    }
  }
  const auto& e = r.estimates;
  const double floor = 1e-12 * norm(F);
  check_stat(e.force_N[0], F.x, "F_x", floor);
  check_stat(e.force_N[1], F.y, "F_y", floor);
  check_stat(e.force_N[2], F.z, "F_z", floor);
  CHECK(e.force_N[2].std_error < 0.01 * std::fabs(F.z));  // the comparison is sharp
  if (fd > 0)
    check_stat(e.heat_rate_W, Q, "Q");
  else
    CHECK(std::fabs(e.heat_rate_W.value) < 1e-12 * e.heat_rate_positive_W.value + 1e-30);
}

TEST_CASE("Flat plate: CLL tangential force and heat against free-molecular theory", "[tpmc][physics][plate]") {
  const auto [an, st] = GENERATE(table<double, double>({{0.85, 0.65}, {0.3, 1.4}, {1.0, 1.0}}));
  INFO("alpha_n = " << an << ", sigma_t = " << st);
  const double a = 25.0 * constants::pi / 180.0;
  const double V = 7500.0, T = 900.0, Tw = 300.0;
  const Vec3 u(-V * std::cos(a), 0.0, -V * std::sin(a));
  Problem p = make_problem(unit_plate(), cll_yaml(an, st, Tw), u, T, "    O: 0.7\n    He: 0.3\n",
                           tpmc_yaml(4000000, 1000000), false);
  const TpmcResult r = run_tpmc(p.cfg, p.geometry, p.inflow);
  Vec3 F;
  double Q = 0;
  for (const auto& sp : p.cfg.gas.species) {
    const double nd = p.cfg.gas.number_density_m3 * sp.mole_fraction;
    for (const Vec3& n : {Vec3(0, 0, 1), Vec3(0, 0, -1)}) {
      const PlateResult pr = plate_cll(n, u, nd, T, sp.mass_kg, an, st, Tw);
      F += pr.force;
      Q += pr.heat;
    }
  }
  check_stat(r.estimates.force_N[0], F.x, "F_x (tangential)");
  check_stat(r.estimates.force_N[1], F.y, "F_y", 1e-12 * norm(F));
  check_stat(r.estimates.heat_rate_W, Q, "Q");
  CHECK(r.estimates.heat_rate_W.std_error < 0.01 * std::fabs(Q));
}

TEST_CASE("Specular limit: no heat transfer, CLL(0,0) identical to Maxwell(0)", "[tpmc][physics]") {
  const TriangleSoup sat = read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl");
  const Vec3 u(-7800, 300, -500);
  Problem pm = make_problem(sat, maxwell_yaml(0.0), u, 900.0, "    O: 0.8\n    He: 0.2\n", tpmc_yaml(500000, 100000));
  Problem pc = make_problem(sat, cll_yaml(0.0, 0.0), u, 900.0, "    O: 0.8\n    He: 0.2\n", tpmc_yaml(500000, 100000));
  const TpmcResult rm = run_tpmc(pm.cfg, pm.geometry, pm.inflow);
  const TpmcResult rc = run_tpmc(pc.cfg, pc.geometry, pc.inflow);
  // Energy is conserved in every specular reflection (up to rounding).
  CHECK(std::fabs(rm.estimates.heat_rate_W.value) < 1e-12 * rm.estimates.heat_rate_positive_W.value + 1e-30);
  CHECK(rm.counters.surface_interactions > 10000);
  // Same histories (specular needs no random numbers): identical results up to rounding.
  for (int k = 0; k < 3; ++k) {
    const double a = rm.estimates.force_N[static_cast<std::size_t>(k)].value;
    const double b = rc.estimates.force_N[static_cast<std::size_t>(k)].value;
    CHECK(std::fabs(a - b) <= 1e-12 * norm(Vec3(rm.estimates.force_N[0].value, 0, 0)));
  }
  CHECK(rm.counters.surface_interactions == rc.counters.surface_interactions);
}

TEST_CASE("Fully diffuse limit: CLL(1,1) statistically equals Maxwell(1)", "[tpmc][physics]") {
  const TriangleSoup sat = read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl");
  const Vec3 u(-7800, 0, -300);
  Problem pm = make_problem(sat, maxwell_yaml(1.0), u, 900.0, "    O: 1.0\n", tpmc_yaml(1000000, 250000));
  Problem pc = make_problem(sat, cll_yaml(1.0, 1.0), u, 900.0, "    O: 1.0\n", tpmc_yaml(1000000, 250000));
  pc.cfg.simulation.random_seed = 99;  // independent sample
  const TpmcResult rm = run_tpmc(pm.cfg, pm.geometry, pm.inflow);
  const TpmcResult rc = run_tpmc(pc.cfg, pc.geometry, pc.inflow);
  check_same(rm.estimates.C_D, rc.estimates.C_D, "C_D");
  check_same(rm.estimates.C_L, rc.estimates.C_L, "C_L");
  check_same(rm.estimates.heat_rate_W, rc.estimates.heat_rate_W, "Q");
  for (std::size_t k = 0; k < 3; ++k) check_same(rm.estimates.C_M[k], rc.estimates.C_M[k], "C_M");
}

TEST_CASE("Thermal equilibrium: no net heat flux for any accommodation", "[tpmc][physics]") {
  const int model = GENERATE(0, 1, 2);
  // Concave geometry (narrow gap) to include multiple reflections.
  TriangleSoup soup = primitives::box({-0.5, -0.5, -0.5}, {0.0, 0.5, 0.5});
  primitives::add_box(soup, {0.02, -0.5, -0.5}, {0.5, 0.5, 0.5});
  const std::string surf = model == 0 ? maxwell_yaml(1.0, 400.0)
                           : model == 1 ? maxwell_yaml(0.4, 400.0)
                                        : cll_yaml(0.5, 0.8, 400.0);
  INFO("model " << model);
  // Gas at rest (a tiny drift defines the axes; its effect on Q is O(u^2)).
  Problem p = make_problem(soup, surf, Vec3(-1e-3, 0, 0), 400.0, "    O: 0.6\n    He: 0.4\n", tpmc_yaml(2000000, 500000));
  const TpmcResult r = run_tpmc(p.cfg, p.geometry, p.inflow);
  const Estimate& q = r.estimates.heat_rate_W;
  INFO("Q = " << q.value << " +- " << q.std_error << ", incident part " << r.estimates.heat_rate_positive_W.value);
  CHECK(std::fabs(q.value) < 5.0 * q.std_error);
  // The test is sensitive: the uncertainty is small compared to the exchanged energy.
  CHECK(q.std_error < 0.02 * r.estimates.heat_rate_positive_W.value);
  CHECK(r.counters.max_interactions > 5);
}

TEST_CASE("Symmetric body in symmetric flow: transverse forces and moments vanish", "[tpmc][physics]") {
  const TriangleSoup sat = read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl");
  Problem p = make_problem(sat, cll_yaml(0.85, 0.65), Vec3(-7800, 0, 0), 900.0, "    O: 0.8\n    N: 0.15\n    He: 0.05\n",
                           tpmc_yaml(2000000, 500000));
  const TpmcResult r = run_tpmc(p.cfg, p.geometry, p.inflow);
  const auto& e = r.estimates;
  check_stat(e.force_N[1], 0.0, "F_y");
  check_stat(e.force_N[2], 0.0, "F_z");
  check_stat(e.C_L, 0.0, "C_L");
  check_stat(e.C_Y, 0.0, "C_Y");
  for (std::size_t k = 0; k < 3; ++k) check_stat(e.C_M[k], 0.0, "C_M");
  CHECK(e.C_D.value > 0.0);
  CHECK(e.C_D.std_error < 0.01 * e.C_D.value);
}

TEST_CASE("Surface fields sum to the global balance; batching and reproducibility", "[tpmc][tally]") {
  const TriangleSoup sat = read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl");
  const Vec3 u(-7600, 400, -900);
  Problem p = make_problem(sat, maxwell_yaml(0.9), u, 900.0, "    O: 0.9\n    He: 0.1\n", tpmc_yaml(600000, 200000, true));
  const TpmcResult a = run_tpmc(p.cfg, p.geometry, p.inflow);
  REQUIRE(a.surface);
  Vec3 F;
  double Q = 0;
  for (std::size_t t = 0; t < p.geometry.host.size(); ++t) {
    F += a.surface->force_N[t];
    Q += a.surface->heat_rate_W[t];
  }
  const Vec3 Fg(a.estimates.force_N[0].value, a.estimates.force_N[1].value, a.estimates.force_N[2].value);
  CHECK(approx_equal(F, Fg, 0.0, 1e-10));
  CHECK_THAT(Q, Catch::Matchers::WithinRel(a.estimates.heat_rate_W.value, 1e-10));
  CHECK(a.batches == 3);

  // Same run again: bitwise identical. Different batch size: identical histories, sums equal up to rounding.
  const TpmcResult b = run_tpmc(p.cfg, p.geometry, p.inflow);
  CHECK(b.estimates.C_D.value == a.estimates.C_D.value);
  CHECK(b.estimates.heat_rate_W.value == a.estimates.heat_rate_W.value);
  p.cfg.tpmc.batch_size = 150000;
  const TpmcResult c = run_tpmc(p.cfg, p.geometry, p.inflow);
  CHECK(c.batches == (VLEO_TEST_SCALE == 1 ? 4u : (600000u / VLEO_TEST_SCALE + 149999u) / 150000u));
  CHECK_THAT(c.estimates.C_D.value, Catch::Matchers::WithinRel(a.estimates.C_D.value, 1e-12));
  CHECK_THAT(c.estimates.C_D.std_error, Catch::Matchers::WithinRel(a.estimates.C_D.std_error, 1e-9));
  CHECK(c.counters.surface_interactions == a.counters.surface_interactions);

  // Coefficients are the forces normalised with q A (and L for moments).
  const double qa = p.cfg.derived.dynamic_pressure_Pa * p.cfg.reference.area_m2;
  CHECK_THAT(a.estimates.C_D.value, Catch::Matchers::WithinRel(dot(Fg, p.cfg.derived.aero_axes.drag) / qa, 1e-12));
  CHECK_THAT(a.estimates.C_L.value, Catch::Matchers::WithinRel(dot(Fg, p.cfg.derived.aero_axes.lift) / qa, 1e-12));

  // Output writers.
  const std::string json = results_json(a, p.cfg, "2026-01-01T00:00:00Z");
  for (const std::string& key : std::vector<std::string>{
           "\"force_N\"", "\"moment_Nm\"", "\"C_D\"", "\"C_L\"", "\"C_M\"", "\"heat_rate_W\"", "\"std_error\"",
           "\"confidence_interval\"", "\"histories\": " + std::to_string(600000 / VLEO_TEST_SCALE)})
    CHECK(json.find(key) != std::string::npos);
  const std::string csv = results_csv(a);
  CHECK(csv.find("C_D,-,") != std::string::npos);
  const std::string sf = surface_fields_csv(a, p.geometry);
  CHECK(static_cast<std::size_t>(std::count(sf.begin(), sf.end(), '\n')) == p.geometry.host.size() + 1);
}

TEST_CASE("Convergence stop on the standard error of a target quantity", "[tpmc][statistics]") {
  const TriangleSoup sat = read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl");
  Problem p = make_problem(sat, maxwell_yaml(1.0), Vec3(-7800, 0, 0), 900.0, "    O: 1.0\n",
                           "tpmc:\n  particle_histories: 50000000\n  batch_size: 20000\n"
                           "  convergence:\n    quantity: C_D\n    relative_error: 0.01\n    min_histories: 100000\n");
  const TpmcResult r = run_tpmc(p.cfg, p.geometry, p.inflow);
  CHECK(r.converged);
  CHECK(r.histories >= 100000);
  CHECK(r.histories < 50000000);
  CHECK(r.histories % 20000 == 0);
  CHECK(r.estimates.C_D.std_error <= 0.01 * r.estimates.C_D.value);
  // One batch earlier the criterion was not yet met (or min_histories not reached).
  CHECK(r.warnings.empty());

  // Unreachable target: runs to particle_histories and warns.
  p.cfg.tpmc.particle_histories = 60000;
  p.cfg.tpmc.convergence->relative_error = 1e-6;
  p.cfg.tpmc.convergence->min_histories = 0;
  const TpmcResult s = run_tpmc(p.cfg, p.geometry, p.inflow);
  CHECK(!s.converged);
  CHECK(s.histories == 60000);
  CHECK(!s.warnings.empty());
}

TEST_CASE("Statistics helpers", "[statistics]") {
  CHECK_THAT(critical_value(0.95), Catch::Matchers::WithinAbs(1.959963984540054, 1e-12));
  CHECK_THAT(critical_value(0.99), Catch::Matchers::WithinAbs(2.5758293035489004, 1e-12));
  CHECK_THAT(normal_quantile(0.5), Catch::Matchers::WithinAbs(0.0, 1e-15));
  CHECK_THAT(normal_quantile(1e-10), Catch::Matchers::WithinAbs(-6.361340902404056, 1e-10));
  // Chan merge equals the one-pass result.
  Moments a, b, all;
  auto add = [](Moments& m, double x) {
    Moments one;
    one.n = 1;
    one.mean = x;
    m.merge(one);
  };
  for (int i = 0; i < 1000; ++i) {
    const double x = (i % 7 == 0) ? 1e3 + i : 0.0;
    add(i < 400 ? a : b, x);
    add(all, x);
  }
  a.merge(b);
  CHECK(a.n == 1000);
  CHECK_THAT(a.mean, Catch::Matchers::WithinRel(all.mean, 1e-14));
  CHECK_THAT(a.m2, Catch::Matchers::WithinRel(all.m2, 1e-12));
  const Estimate e = make_estimate(a, 2.0, 1.96);
  CHECK_THAT(e.value, Catch::Matchers::WithinRel(2.0 * a.mean, 1e-15));
  CHECK_THAT(e.ci_high - e.ci_low, Catch::Matchers::WithinRel(2 * 1.96 * e.std_error, 1e-14));
}
