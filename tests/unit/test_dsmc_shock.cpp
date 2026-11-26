// Normal shock in argon (docs/verification.md, stage 1): a one-dimensional
// box between two inflow faces (dsmc.faces type inflow) with the upstream and
// the Rankine-Hugoniot downstream state, started from the jump (fill_depth_m).
//
// Checks (tolerances fixed before the tests were run):
//   inflow faces: number flux and velocity moments of the injected molecules,
//     region fill: particle numbers and states   5 standard errors
//   Ma 1.4: plateaus vs. Rankine-Hugoniot          1 % (n, u), 1.5 % (T)
//           shock thickness (tanh fit, delta = 2 w) vs. Navier-Stokes with the same mu(T)
//             and kappa = 15 k mu / (4 m): 1 <= delta / delta_NS <= 1.2 (5 SE margin below).
//             Navier-Stokes underestimates the thickness of a shock, increasingly with
//             the Mach number; recorded DSMC / NS: 1.05 +- 0.03 (Ma 1.2, plateaus not
//             fully reached in 60 lambda1), 1.10 +- 0.02 (Ma 1.4), 1.27 +- 0.01 (Ma 2.0).
//           conservation of the mass flux through the shock: 1 %

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <sstream>

#include "vleo/core/PhysicalConstants.hpp"
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

constexpr double kB = constants::k_B;
constexpr double kGamma = 5.0 / 3.0;

struct Case {
  SimulationConfig cfg;
  Geometry geometry;
  InflowModel inflow;
};

// Argon box x [-L/2, L/2], y, z [0, w] with the given dsmc.faces block; a
// specular 0.2 um marker cube in a corner (the solver needs a body).
Case argon_tube(double n, double T, double u, double L, double w, double cell, int ppc, const std::string& faces) {
  static int counter = 0;
  const fs::path dir = fs::temp_directory_path() / "vleo_test_dsmc_shock";
  fs::create_directories(dir);
  const fs::path stl = dir / ("body" + std::to_string(counter++) + ".stl");
  const Vec3 c0(-0.49 * L, 0.05 * w, 0.05 * w);
  const TriangleSoup soup = primitives::box(c0 - Vec3(1e-7, 1e-7, 1e-7), c0 + Vec3(1e-7, 1e-7, 1e-7));
  write_stl_ascii(stl, soup);
  std::ostringstream y;
  y.precision(17);
  y << "geometry: {file: " << stl.string() << "}\n"
    << "simulation: {name: tube, mode: dsmc, random_seed: 31}\n"
    << "domain: {box_min_m: [" << -0.5 * L << ", 0, 0], box_max_m: [" << 0.5 * L << ", " << w << ", " << w << "]}\n"
    << "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
    << "gas: {number_density_m3: " << n << ", translational_temperature_K: " << T << ", bulk_velocity_body_mps: [" << u
    << ", 0, 0], species: {Ar: 1.0}}\n"
    << "surface: {model: maxwell, temperature_K: " << T << ", maxwell: {diffuse_fraction: 0}}\n"
    << "dsmc:\n  boundary: specular\n  cell_size_mode: fixed\n  cell_size_m: " << cell
    << "\n  particles_per_cell_target: " << ppc << "\n  faces:\n"
    << faces;
  Case c;
  c.cfg = parse_simulation_config(y.str(), dir);
  c.geometry = make_geometry(soup);
  const Domain dom = make_domain(c.cfg.domain, c.geometry.host.bbox, 1e3 * c.geometry.device.surface_offset_m);
  c.inflow = InflowModel::create(dom, c.cfg.gas);
  return c;
}

std::string inflow_face(const char* face, double n, double T, double u, double fill) {
  std::ostringstream s;
  s.precision(17);
  s << "    " << face << ": {type: inflow, number_density_m3: " << n << ", temperature_K: " << T
    << ", velocity_mps: [" << u << ", 0, 0], fill_depth_m: " << fill << "}\n";
  return s.str();
}

// Rankine-Hugoniot jump of a monatomic gas: (n2/n1, u2/u1, T2/T1).
std::array<double, 3> rankine_hugoniot(double M) {
  const double g = kGamma, M2 = M * M;
  const double r = (g + 1.0) * M2 / ((g - 1.0) * M2 + 2.0);
  const double t = (2.0 * g * M2 - (g - 1.0)) * ((g - 1.0) * M2 + 2.0) / ((g + 1.0) * (g + 1.0) * M2);
  return {r, 1.0 / r, t};
}

// First-order Chapman-Enskog viscosity of VHS argon.
double viscosity(const SpeciesInfo& s, double T) {
  const VhsParameters& p = s.vhs;
  const double w = p.omega, d = p.d_ref_m;
  return 15.0 * std::sqrt(constants::pi * s.mass_kg * kB * p.T_ref_K) /
         (2.0 * constants::pi * d * d * (5.0 - 2.0 * w) * (7.0 - 2.0 * w)) * std::pow(T / p.T_ref_K, w);
}

// Navier-Stokes shock structure (shock frame, x along the flow). With the mass
// flux m0 = rho1 u1, momentum flux P and energy flux E:
//   4/3 mu u' = m0 u + rho R T - P,   kappa T' = m0 c_v T - m0 u^2 / 2 + P u - E.
// The profile is the trajectory from the upstream node into the downstream
// saddle; it is integrated backwards from the saddle along its stable
// eigenvector. Returns the number density on the grid xs (centre: n = (n1+n2)/2).
std::vector<double> navier_stokes_density(const SpeciesInfo& ar, double n1, double T1, double u1, double M,
                                          const std::vector<double>& xs) {
  const double m = ar.mass_kg, R = kB / m, cv = 1.5 * R;
  const auto rh = rankine_hugoniot(M);
  const double rho1 = n1 * m, m0 = rho1 * u1;
  const double P = m0 * u1 + rho1 * R * T1, E = m0 * (2.5 * R * T1 + 0.5 * u1 * u1);
  auto rhs = [&](double u, double T, double& du, double& dT) {
    const double mu = viscosity(ar, T), kappa = 15.0 * kB / (4.0 * m) * mu;
    du = 0.75 / mu * (m0 * u + m0 * R * T / u - P);
    dT = (m0 * cv * T - 0.5 * m0 * u * u + P * u - E) / kappa;
  };
  const double u2 = u1 * rh[1], T2 = T1 * rh[2];
  // Jacobian at the downstream point
  const double hu = 1e-6 * u2, hT = 1e-6 * T2;
  double a[2][2], f0u, f0T, fu, fT;
  rhs(u2, T2, f0u, f0T);
  rhs(u2 + hu, T2, fu, fT);
  a[0][0] = (fu - f0u) / hu;
  a[1][0] = (fT - f0T) / hu;
  rhs(u2, T2 + hT, fu, fT);
  a[0][1] = (fu - f0u) / hT;
  a[1][1] = (fT - f0T) / hT;
  const double tr = a[0][0] + a[1][1], det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
  const double lam = 0.5 * tr - std::sqrt(0.25 * tr * tr - det);  // negative eigenvalue (saddle)
  double eu = a[0][1], eT = lam - a[0][0];                         // eigenvector
  if (std::fabs(eu) < 1e-300) {
    eu = lam - a[1][1];
    eT = a[1][0];
  }
  const double nrm = std::hypot(eu / u2, eT / T2);
  eu /= nrm;
  eT /= nrm;
  if (eu < 0.0) {  // step towards the upstream state (larger u)
    eu = -eu;
    eT = -eT;
  }
  // integrate backwards in x from the saddle, starting on its stable eigenvector
  std::vector<double> x{0.0}, n{n1 * rh[0]};
  double u = u2 + 1e-6 * eu, T = T2 + 1e-6 * eT;
  double xc = 0.0;
  const double h = -2e-7;  // 0.2 um steps (the shock is ~ 1 cm thick)
  for (int k = 0; k < 2000000 && u < u1 * (1.0 - 1e-6); ++k) {
    double k1u, k1T, k2u, k2T, k3u, k3T, k4u, k4T;
    rhs(u, T, k1u, k1T);
    rhs(u + 0.5 * h * k1u, T + 0.5 * h * k1T, k2u, k2T);
    rhs(u + 0.5 * h * k2u, T + 0.5 * h * k2T, k3u, k3T);
    rhs(u + h * k3u, T + h * k3T, k4u, k4T);
    u += h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u);
    T += h / 6.0 * (k1T + 2 * k2T + 2 * k3T + k4T);
    xc += h;
    x.push_back(xc);
    n.push_back(m0 / (m * u));
  }
  std::reverse(x.begin(), x.end());
  std::reverse(n.begin(), n.end());
  // centre at n = (n1 + n2) / 2
  const double nm = 0.5 * n1 * (1.0 + rh[0]);
  double x0 = 0.0;
  for (std::size_t i = 1; i < n.size(); ++i)
    if ((n[i - 1] - nm) * (n[i] - nm) <= 0.0) {
      x0 = x[i - 1] + (nm - n[i - 1]) / (n[i] - n[i - 1]) * (x[i] - x[i - 1]);
      break;
    }
  std::vector<double> out;
  for (double xq : xs) {
    const double xx = xq + x0;
    if (xx <= x.front()) {
      out.push_back(n.front());
      continue;
    }
    if (xx >= x.back()) {
      out.push_back(n.back());
      continue;
    }
    const auto it = std::upper_bound(x.begin(), x.end(), xx);
    const std::size_t i = static_cast<std::size_t>(it - x.begin());
    out.push_back(n[i - 1] + (xx - x[i - 1]) / (x[i] - x[i - 1]) * (n[i] - n[i - 1]));
  }
  return out;
}

// Shock thickness delta = (n2 - n1) / max dn/dx of the least-squares fit
// n = (n1 + n2)/2 + (n2 - n1)/2 tanh((x - x0) / w) over |x| < range, so
// delta = 2 w. A fit instead of the largest local slope: the maximum of a
// noisy slope is biased high (thickness biased low). The same estimator is
// applied to the Navier-Stokes profile.
double thickness(const std::vector<double>& xs, const std::vector<double>& n, double n1, double n2, double range) {
  const double a = 0.5 * (n1 + n2), b = 0.5 * (n2 - n1);
  auto sse = [&](double x0, double w) {
    double e = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      if (std::fabs(xs[i]) > range) continue;
      const double r = n[i] - (a + b * std::tanh((xs[i] - x0) / w));
      e += r * r;
    }
    return e;
  };
  // coordinate search on (x0, w), shrinking steps
  double x0 = 0.0, w = 0.2 * range, dx0 = 0.1 * range, dw = 0.1 * range;
  double best = sse(x0, w);
  for (int it = 0; it < 400 && (dx0 > 1e-6 * range || dw > 1e-6 * range); ++it) {
    bool moved = false;
    for (const auto& [px, pw] : {std::pair{dx0, 0.0}, {-dx0, 0.0}, {0.0, dw}, {0.0, -dw}}) {
      const double nx = x0 + px, nw = w + pw;
      if (nw <= 0.0) continue;
      const double e = sse(nx, nw);
      if (e < best) {
        best = e;
        x0 = nx;
        w = nw;
        moved = true;
      }
    }
    if (!moved) {
      dx0 *= 0.5;
      dw *= 0.5;
    }
  }
  return 2.0 * w;
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

}  // namespace

TEST_CASE("Inflow faces: injected flux and state, region fill", "[dsmc][faces]") {
  // Empty argon tube (no fill), inflow at x_min with 300 K, 500 m/s, and at
  // x_max with 600 K, at rest: count and moments of the entering molecules.
  const SpeciesInfo ar = *find_species("Ar");
  const double m = ar.mass_kg;
  const double L = 0.02, w = 0.005, n = 1e18;  // lambda ~ 1.3 m: no collisions
  Case c = argon_tube(n, 300.0, 500.0, L, w, 0.0025, 200,
                      inflow_face("x_min", n, 300.0, 500.0, 0.0) + inflow_face("x_max", 2.0 * n, 600.0, 0.0, 0.0));
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  REQUIRE(s.faces[0].type == DsmcFace::Inflow);
  REQUIRE(s.faces[1].type == DsmcFace::Inflow);
  s.timestep_s = 1e-7;  // particles move < 0.1 mm per step
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
  const int steps = 1000;
  for (int k = 0; k < steps; ++k) sim.step(false);
  // Expected number entering through each face: n c / (2 sqrt pi) F(s) A dt steps / W
  const double A = w * w;
  auto entering = [&](double nn, double T, double un) {
    const double cth = std::sqrt(2.0 * kB * T / m);
    return number_flux(nn, cth, un / cth) * A * s.timestep_s * steps / s.particle_weight;
  };
  const double e_lo = entering(n, 300.0, 500.0), e_hi = entering(2.0 * n, 600.0, 0.0);
  const double injected = static_cast<double>(sim.counters().injected);
  INFO("injected " << injected << ", expected " << e_lo + e_hi);
  CHECK(std::fabs(injected - (e_lo + e_hi)) < 5.0 * std::sqrt(e_lo + e_hi));
  // Velocities of the molecules that entered through x_min (v_x > 0 near x_min):
  // tangential mean 0 and variance k T / m; through x_max: v_x < 0.
  sim.compact();
  const auto pos = sim.positions();
  const auto vel = sim.velocities();
  // (the residence time depends on v_x only, so the tangential moments are unbiased;
  // specular side faces flip the sign of v_y / v_z but keep their squares)
  double vz_lo = 0, vz2_lo = 0, cnt_lo = 0, vz2_hi = 0, cnt_hi = 0;
  for (std::size_t i = 0; i < pos.size(); ++i) {
    if (vel[i].x > 0.0) {
      vz_lo += vel[i].z;
      vz2_lo += vel[i].z * vel[i].z;
      cnt_lo += 1;
    } else {
      vz2_hi += vel[i].z * vel[i].z;
      cnt_hi += 1;
    }
  }
  REQUIRE(cnt_lo > 1000);
  REQUIRE(cnt_hi > 1000);
  CHECK(std::fabs(vz_lo / cnt_lo) < 5.0 * std::sqrt(kB * 300.0 / m / cnt_lo));
  CHECK_THAT(vz2_lo / cnt_lo, WithinRel(kB * 300.0 / m, 5.0 * std::sqrt(2.0 / cnt_lo)));
  CHECK_THAT(vz2_hi / cnt_hi, WithinRel(kB * 600.0 / m, 5.0 * std::sqrt(2.0 / cnt_hi)));

  // Region fill: x_min slab 5 mm with (n, 300 K, 500 m/s), x_max slab 15 mm with (2n, 600 K, 0).
  Case f = argon_tube(n, 300.0, 500.0, L, w, 0.0025, 20,
                      inflow_face("x_min", n, 300.0, 500.0, 0.005) + inflow_face("x_max", 2.0 * n, 600.0, 0.0, 0.015));
  DsmcSettings sf = resolve_dsmc_settings(f.cfg, f.inflow.domain);
  DsmcSimulation fill(f.cfg, f.geometry, f.inflow, sf);
  fill.fill_freestream();
  const auto fp = fill.positions();
  const auto fv = fill.velocities();
  double n_lo = 0, n_hi = 0, ux_lo = 0, e_hi2 = 0;
  for (std::size_t i = 0; i < fp.size(); ++i) {
    if (fp[i].x < -0.005) {
      n_lo += 1;
      ux_lo += fv[i].x;
    } else {
      n_hi += 1;
      e_hi2 += norm2(fv[i]);
    }
  }
  const double N_lo = n * 0.005 * A / sf.particle_weight, N_hi = 2.0 * n * 0.015 * A / sf.particle_weight;
  CHECK(std::fabs(n_lo - N_lo) < 1.0);  // count = floor(expected + U)
  CHECK(std::fabs(n_hi - N_hi) < 1.0);
  CHECK(std::fabs(ux_lo / n_lo - 500.0) < 5.0 * std::sqrt(kB * 300.0 / m / n_lo));
  CHECK_THAT(e_hi2 / n_hi, WithinRel(3.0 * kB * 600.0 / m, 5.0 * std::sqrt(2.0 / (3.0 * n_hi))));
}

TEST_CASE("Rejected: inflow faces", "[dsmc][faces][config]") {
  CHECK_THROWS(argon_tube(1e20, 300, 100, 0.02, 0.005, 0.0025, 10, "    x_min: inflow\n"));
  CHECK_THROWS(argon_tube(1e20, 300, 100, 0.02, 0.005, 0.0025, 10,
                          "    x_min: {type: inflow, temperature_K: 300}\n"));  // no density
  CHECK_THROWS(argon_tube(1e20, 300, 100, 0.02, 0.005, 0.0025, 10,
                          "    x_min: {type: wall, temperature_K: 300, number_density_m3: 1e20}\n"));
  CHECK_THROWS(argon_tube(1e20, 300, 100, 0.02, 0.005, 0.0025, 10,
                          inflow_face("x_min", 1e20, 300, 100, 0.01) +
                              "    y_min: {type: inflow, number_density_m3: 1e20, temperature_K: 300, fill_depth_m: "
                              "0.001}\n"));  // fill on two axes
}

TEST_CASE("Normal shock in argon at Ma 1.4: Rankine-Hugoniot plateaus and Navier-Stokes thickness",
          "[dsmc][faces][physics][shock][.]") {
  const SpeciesInfo ar = *find_species("Ar");
  const double m = ar.mass_kg;
  const double M = 1.4, T1 = 273.0, n1 = 1e21;
  const double a1 = std::sqrt(kGamma * kB * T1 / m), u1 = M * a1;
  const auto rh = rankine_hugoniot(M);
  const double n2 = n1 * rh[0], u2 = u1 * rh[1], T2 = T1 * rh[2];
  // mean free path of the upstream gas (solver definition)
  Case probe = argon_tube(n1, T1, u1, 0.01, 0.01, 0.001, 10, "    x_min: specular\n");
  const double lambda1 = resolve_dsmc_settings(probe.cfg, probe.inflow.domain).lambda_inf_m;
  const double L = 60.0 * lambda1, dx = lambda1 / 4.0;
  Case c = argon_tube(n1, T1, u1, L, dx, dx, 100,
                      inflow_face("x_min", n1, T1, u1, 0.5 * L) + inflow_face("x_max", n2, T2, u2, 0.5 * L));
  DsmcSettings s = resolve_dsmc_settings(c.cfg, c.inflow.domain);
  s.timestep_s = 0.1 * s.tau_inf_s;
  DsmcSimulation sim(c.cfg, c.geometry, c.inflow, s);
  sim.fill_freestream();
  for (int k = 0; k < 3000; ++k) sim.step(false);

  // Sampling: every 5 steps the particles are binned along x (bins dx / 2);
  // windows of 400 steps are aligned on the shock centre (n = (n1 + n2) / 2)
  // before they are added, so the slow random walk of the shock does not smear
  // the profile. 10 blocks for the standard errors.
  const int bins = static_cast<int>(std::lround(L / (0.5 * dx)));
  const double bw = L / bins;
  const int window = 400, windows_per_block = 6 / VLEO_TEST_SCALE > 0 ? 6 / VLEO_TEST_SCALE : 1, blocks = 10;
  const int half = bins / 2;
  const double cell_volume = bw * dx * dx;
  std::vector<std::vector<double>> block_n(blocks, std::vector<double>(bins, 0.0)),
      block_u(blocks, std::vector<double>(bins, 0.0)), block_e(blocks, std::vector<double>(bins, 0.0)),
      block_c(blocks, std::vector<double>(bins, 0.0));
  std::vector<int> block_windows(blocks, 0);
  double max_shift = 0.0;
  for (int b = 0; b < blocks; ++b) {
    for (int wdw = 0; wdw < windows_per_block; ++wdw) {
      std::vector<double> cn(bins, 0.0), cu(bins, 0.0), ce(bins, 0.0);
      int samples = 0;
      for (int k = 0; k < window; ++k) {
        sim.step(true);
        if (k % 5 != 0) continue;
        const auto pos = sim.positions();
        const auto vel = sim.velocities();
        for (std::size_t i = 0; i < pos.size(); ++i) {
          const int j = std::clamp(static_cast<int>((pos[i].x + 0.5 * L) / bw), 0, bins - 1);
          cn[static_cast<std::size_t>(j)] += 1.0;
          cu[static_cast<std::size_t>(j)] += vel[i].x;
          ce[static_cast<std::size_t>(j)] += norm2(vel[i]);
        }
        ++samples;
      }
      // shock centre of this window
      const double nm = 0.5 * (n1 + n2) * cell_volume / s.particle_weight * samples;
      int jc = half;
      for (int j = 2; j < bins - 2; ++j) {
        const double a = (cn[j - 2] + cn[j - 1] + cn[j]) / 3.0, bb = (cn[j] + cn[j + 1] + cn[j + 2]) / 3.0;
        if (a < nm && bb >= nm) {
          jc = j;
          break;
        }
      }
      max_shift = std::max(max_shift, std::fabs((jc - half) * bw));
      const int shift = jc - half;
      for (int j = 0; j < bins; ++j) {
        const int src = j + shift;
        if (src < 0 || src >= bins) continue;
        block_n[b][j] += cn[src];
        block_u[b][j] += cu[src];
        block_e[b][j] += ce[src];
        block_c[b][j] += samples;
      }
      ++block_windows[b];
    }
  }
  INFO("largest shock displacement " << max_shift / lambda1 << " lambda1");
  REQUIRE(max_shift < 0.2 * L);  // the shock stayed well inside the box

  std::vector<double> xs(bins);
  for (int j = 0; j < bins; ++j) xs[j] = (j + 0.5) * bw - 0.5 * L;
  const std::vector<double> n_ns = navier_stokes_density(ar, n1, T1, u1, M, xs);
  const double d_ns = thickness(xs, n_ns, n1, n2, 20.0 * lambda1);
  std::vector<double> d_b, n1_b, n2_b, u1_b, u2_b, T2_b, flux_b;
  for (int b = 0; b < blocks; ++b) {
    std::vector<double> nb(bins);
    for (int j = 0; j < bins; ++j)  // bins not covered after the alignment: plateau value
      nb[j] = block_c[b][j] > 0 ? block_n[b][j] * s.particle_weight / (cell_volume * block_c[b][j])
                                : (xs[j] < 0 ? n1 : n2);
    // plateaus 15-25 lambda1 from the centre
    double su = 0, sd = 0, uu = 0, ud = 0, Td = 0, cu = 0, cd = 0;
    for (int j = 0; j < bins; ++j) {
      const double x = xs[j] / lambda1;
      const double nj = block_n[b][j];
      if (nj <= 0) continue;
      const double ux = block_u[b][j] / nj;
      const double T = m * (block_e[b][j] / nj - ux * ux) / (3.0 * kB);
      if (x < -15 && x > -25) {
        su += nb[j];
        uu += ux;
        cu += 1;
      } else if (x > 15 && x < 25) {
        sd += nb[j];
        ud += ux;
        Td += T;
        cd += 1;
      }
    }
    n1_b.push_back(su / cu);
    n2_b.push_back(sd / cd);
    u1_b.push_back(uu / cu);
    u2_b.push_back(ud / cd);
    T2_b.push_back(Td / cd);
    flux_b.push_back((sd / cd) * (ud / cd) / ((su / cu) * (uu / cu)));
    d_b.push_back(thickness(xs, nb, n1, n2, 20.0 * lambda1));
  }
  const double d = mean(d_b), d_se = std_error(d_b);
  INFO("Ma " << M << ": n2/n1 " << mean(n2_b) / mean(n1_b) << " (RH " << rh[0] << "), T2/T1 " << mean(T2_b) / T1
             << " (RH " << rh[2] << "), u1 " << mean(u1_b) << " (" << u1 << "), u2 " << mean(u2_b) << " (" << u2
             << ")");
  INFO("thickness " << d / lambda1 << " +- " << d_se / lambda1 << " lambda1, Navier-Stokes " << d_ns / lambda1
                    << " lambda1, lambda1 / delta " << lambda1 / d << " (NS " << lambda1 / d_ns << ")");
  CHECK_THAT(mean(n1_b), WithinRel(n1, 0.01));
  CHECK_THAT(mean(n2_b), WithinRel(n2, 0.01));
  CHECK_THAT(mean(u1_b), WithinRel(u1, 0.01));
  CHECK_THAT(mean(u2_b), WithinRel(u2, 0.01));
  CHECK_THAT(mean(T2_b), WithinRel(T2, 0.015));
  CHECK_THAT(mean(flux_b), WithinRel(1.0, 0.01));
  CHECK(d > d_ns - 5.0 * d_se);       // not thinner than the continuum solution
  CHECK(d < 1.2 * d_ns + 5.0 * d_se);  // and close to it at this weak shock
}
