// Binary diffusion (docs/verification.md, stage 1): decay of a cosine
// concentration mode in a closed specular box.
//
// An equimolar mixture of A and B at uniform density and temperature with
// x_A(x) = 1/2 + a0 cos(pi x / L) relaxes as a(t) = a0 exp(-D12 pi^2 t / L^2):
// with specular walls the cosine is an eigenmode of Fick's equation, and the
// molar-average velocity stays zero. The decay rate gives D12, compared with
// the first Chapman-Enskog approximation for the VHS / VSS pair of the solver
// (diameter, omega, T_ref and alpha averaged, vleo/dsmc/Vhs.hpp):
//   D12 = 3 (alpha + 1) / (8 (5 - 2 omega)) sqrt(2 pi k T / m_r) / (n pi d^2) (T / T_ref)^(omega - 1/2)
// (hard spheres: 3 / (8 n d^2) sqrt(k T / (2 pi m_r))).
//
// Tolerances (fixed before the tests were run):
//   O / N (masses 16 / 14, same collision data: higher Chapman-Enskog orders < 1 %):
//     5 SE + 2 %
//   He / Ar (mass ratio 0.1: the higher orders raise D12 by a few %): the measured
//     value must lie between 0.97 and 1.12 times the first approximation.
// Box length study on the W7700 (DIFF_L, 16 runs each, docs/verification.md):
//   L / lambda      15              30              60
//   O/N VHS     0.998 +- 0.020  0.999 +- 0.007  1.001 +- 0.006
//   He/Ar VHS   0.958 +- 0.028  1.072 +- 0.016  1.080 +- 0.008
//   He/Ar VSS   0.687 +- 0.025  1.038 +- 0.011  1.088 +- 0.007
// He has a mean free path several times the mixture lambda, so its finite-k lambda
// error is large at small L; at the default L = 30 lambda He/Ar VSS is still ~5 % low.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <sstream>

#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/io/ConfigParser.hpp"

using namespace vleo;
namespace fs = std::filesystem;

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

// Closed specular box x [0, L], y, z [0, w] of an equimolar A/B mixture.
Case mixture_box(const std::string& a, const std::string& b, double n, double T, double L, double w, double cell,
                 int ppc, const std::string& model, std::uint64_t seed) {
  static int counter = 0;
  const fs::path dir = fs::temp_directory_path() / "vleo_test_dsmc_diffusion";
  fs::create_directories(dir);
  const fs::path stl = dir / ("body" + std::to_string(counter++) + ".stl");
  const Vec3 c0(0.01 * L, 0.05 * w, 0.05 * w);
  const TriangleSoup soup = primitives::box(c0 - Vec3(1e-7, 1e-7, 1e-7), c0 + Vec3(1e-7, 1e-7, 1e-7));
  write_stl_ascii(stl, soup);
  std::ostringstream y;
  y.precision(17);
  y << "geometry: {file: " << stl.string() << "}\n"
    << "simulation: {name: diffusion, mode: dsmc, random_seed: " << seed << "}\n"
    << "domain: {box_min_m: [0, 0, 0], box_max_m: [" << L << ", " << w << ", " << w << "]}\n"
    << "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
    << "gas: {number_density_m3: " << n << ", translational_temperature_K: " << T
    << ", bulk_velocity_body_mps: [-1.0e-3, 0, 0], species: {" << a << ": 0.5, " << b << ": 0.5}}\n"
    << "surface: {model: maxwell, temperature_K: " << T << ", maxwell: {diffuse_fraction: 0}}\n"
    << "dsmc: {boundary: specular, collision_model: " << model << ", cell_size_mode: fixed, cell_size_m: " << cell
    << ", particles_per_cell_target: " << ppc << "}\n";
  Case c;
  c.cfg = parse_simulation_config(y.str(), dir);
  c.geometry = make_geometry(soup);
  const Domain dom = make_domain(c.cfg.domain, c.geometry.host.bbox, 1e3 * c.geometry.device.surface_offset_m);
  c.inflow = InflowModel::create(dom, c.cfg.gas);
  return c;
}

// First Chapman-Enskog approximation of D12 for the solver's VHS / VSS pair.
double diffusion_ce(const SpeciesInfo& p, const SpeciesInfo& q, bool vss, double n, double T) {
  const double d = vss ? 0.5 * (p.vhs.d_vss_m + q.vhs.d_vss_m) : 0.5 * (p.vhs.d_ref_m + q.vhs.d_ref_m);
  const double w = 0.5 * (p.vhs.omega + q.vhs.omega);
  const double Tref = 0.5 * (p.vhs.T_ref_K + q.vhs.T_ref_K);
  const double a = vss ? 0.5 * (p.vhs.alpha + q.vhs.alpha) : 1.0;
  const double mr = p.mass_kg * q.mass_kg / (p.mass_kg + q.mass_kg);
  return 3.0 * (a + 1.0) / (8.0 * (5.0 - 2.0 * w)) * std::sqrt(2.0 * constants::pi * kB * T / mr) /
         (n * constants::pi * d * d) * std::pow(T / Tref, w - 0.5);
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

// One run: returns the measured D12 from the decay of the cosine mode.
double measure_d12(const std::string& a, const std::string& b, const std::string& model, std::uint64_t seed,
                   double& lambda_out, double& d_ce_out) {
  const SpeciesInfo pa = *find_species(a), pb = *find_species(b);
  const double n = 1e21, T = 273.0;
  // mixture mean free path (solver definition) from a probe box
  Case probe = mixture_box(a, b, n, T, 0.01, 0.01, 0.001, 10, model, seed);
  const DsmcSettings ps = resolve_dsmc_settings(probe.cfg, probe.inflow.domain);
  const double lambda = ps.lambda_inf_m;
  // box length in mixture mean free paths; DIFF_L overrides it for the k lambda study (docs/verification.md)
  const double L = (std::getenv("DIFF_L") ? std::atof(std::getenv("DIFF_L")) : 30.0) * lambda, dx = lambda / 3.0,
               w = 2.0 * dx;
  Case c = mixture_box(a, b, n, T, L, w, dx, 100, model, seed);
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  s.timestep_s = 0.2 * s.tau_inf_s;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);

  const double a0 = 0.4;
  const auto N = static_cast<std::size_t>(std::llround(n * L * w * w / s.particle_weight));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::normal_distribution<double> nor(0.0, 1.0);
  std::vector<Vec3> pos, vel;
  std::vector<std::uint32_t> sp;
  for (std::size_t i = 0; i < N; ++i) {
    const double x = uni(rng) * L;
    const bool is_a = uni(rng) < 0.5 + a0 * std::cos(constants::pi * x / L);
    const double sd = std::sqrt(kB * T / (is_a ? pa.mass_kg : pb.mass_kg));
    pos.emplace_back(x, uni(rng) * w, uni(rng) * w);
    vel.emplace_back(sd * nor(rng), sd * nor(rng), sd * nor(rng));
    sp.push_back(is_a ? 0u : 1u);
  }
  sim.set_particles(pos, vel, sp);

  // D ~ 0.5 lambda c_mean: decay time L^2 / (pi^2 D) ~ 180 tau. Fit ln a(t) from
  // 20 tau (the initial Maxwellians relax to the diffusive state within a few
  // tau) until a has fallen to 0.15 a0 (beyond, noise dominates the logarithm).
  const double D_ce = diffusion_ce(pa, pb, model == "VSS", n, T);
  const double t_decay = L * L / (constants::pi * constants::pi * D_ce);
  const double t_end = t_decay * std::log(1.0 / 0.15);  // until a ~ 0.15 a0
  const int steps = static_cast<int>(std::ceil(t_end / s.timestep_s));
  std::vector<double> ts, la;
  for (int k = 1; k <= steps; ++k) {
    sim.step(false);
    if (k % 5 != 0) continue;
    const double t = k * s.timestep_s;
    if (t < 20.0 * s.tau_inf_s) continue;
    const auto x = sim.positions();
    const auto spc = sim.species();
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
      if (spc[i] == 0u) sum += std::cos(constants::pi * x[i].x / L);
    const double amp = 2.0 * sum / static_cast<double>(x.size());
    if (!(amp > 0.0)) continue;
    ts.push_back(t);
    la.push_back(std::log(amp));
  }
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double m = static_cast<double>(ts.size());
  for (std::size_t i = 0; i < ts.size(); ++i) {
    sx += ts[i];
    sy += la[i];
    sxx += ts[i] * ts[i];
    sxy += ts[i] * la[i];
  }
  const double rate = -(m * sxy - sx * sy) / (m * sxx - sx * sx);
  lambda_out = lambda;
  d_ce_out = D_ce;
  return rate * L * L / (constants::pi * constants::pi);
}

}  // namespace

TEST_CASE("Binary diffusion: decay of a concentration mode vs. Chapman-Enskog", "[dsmc][physics][diffusion]") {
  const auto [a, b, model, lo, hi, strict] = GENERATE(table<std::string, std::string, std::string, double, double, bool>(
      {{"O", "N", "VHS", 0.0, 0.0, true}, {"He", "Ar", "VHS", 0.97, 1.12, false}, {"He", "Ar", "VSS", 0.97, 1.12, false}}));
  // independent runs (seeds) for the standard error; DIFF_RUNS overrides the number for studies
  const int runs = std::getenv("DIFF_RUNS") ? std::atoi(std::getenv("DIFF_RUNS"))
                                            : (6 / VLEO_TEST_SCALE > 1 ? 6 / VLEO_TEST_SCALE : 2);
  std::vector<double> d;
  double lambda = 0.0, d_ce = 0.0;
  for (int r = 0; r < runs; ++r) d.push_back(measure_d12(a, b, model, 101 + static_cast<std::uint64_t>(r), lambda, d_ce));
  const double dm = mean(d), se = std_error(d);
  INFO(a << "/" << b << " " << model << ": D12 = " << dm << " +- " << se << " m2/s, Chapman-Enskog (1st) " << d_ce
         << ", ratio " << dm / d_ce << " +- " << se / d_ce << ", lambda " << lambda << " m, " << runs << " runs");
  if (strict) {
    CHECK(std::fabs(dm - d_ce) < 5.0 * se + 0.02 * d_ce);
  } else {
    CHECK(dm / d_ce > lo);
    CHECK(dm / d_ce < hi);
  }
}
