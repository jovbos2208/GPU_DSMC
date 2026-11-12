// M2.1-M2.4, M2.13-M2.15 (kernel level): Maxwell and CLL scattering kernels.
//
// Tolerances (fixed before the tests were run, docs/tpmc.md section 7):
//   deterministic identities: 1e-14 relative
//   statistical tests: 5 standard errors; Kolmogorov-Smirnov sqrt(n) D < 1.95 (p ~ 0.1 %)

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "DeviceTest.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/gsi/Scatter.hpp"

using namespace vleo;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const double kMass = 15.999 * constants::amu;  // atomic oxygen
constexpr double kTw = 300.0;
constexpr int kN = 300000;

struct ScatterBatch {
  SurfaceProperties sp;
  Vec3 v_in;
  Vec3 n;
  double mass;
  KOKKOS_INLINE_FUNCTION Vec3 operator()(int i) const {
    RandomStream rng(31, static_cast<std::uint64_t>(i), 1, RngStream::SurfaceInteraction);
    return scatter_particle(v_in, sp.wall_velocity, n, mass, 0.0, sp, rng).velocity_out;
  }
};

SurfaceProperties maxwell(double fd) {
  SurfaceProperties s;
  s.model = GSIModel::Maxwell;
  s.temperature_K = kTw;
  s.diffuse_fraction = fd;
  return s;
}

SurfaceProperties cll(double an, double st) {
  SurfaceProperties s;
  s.model = GSIModel::CLL;
  s.temperature_K = kTw;
  s.alpha_n = an;
  s.sigma_t = st;
  return s;
}

// Local components (t1, t2, n) of all samples, relative to the wall.
std::vector<Vec3> sample_local(const SurfaceProperties& sp, const Vec3& v_in, const Vec3& n) {
  const auto out = test::eval_on_device<Vec3>("scatter", kN, ScatterBatch{sp, v_in, n, kMass});
  const OrthonormalBasis b = make_basis_from_normal(n);
  std::vector<Vec3> l(kN);
  for (int i = 0; i < kN; ++i) l[static_cast<std::size_t>(i)] = b.to_local(out(i) - sp.wall_velocity);
  return l;
}

double ks(std::vector<double> x, double (*cdf)(double, double), double p) {
  std::sort(x.begin(), x.end());
  const double n = static_cast<double>(x.size());
  double d = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double f = cdf(x[i], p);
    d = std::max({d, (static_cast<double>(i) + 1.0) / n - f, f - static_cast<double>(i) / n});
  }
  return std::sqrt(n) * d;
}

double normal_cdf(double x, double sigma) { return 0.5 * std::erfc(-x / (sigma * std::sqrt(2.0))); }
double rayleigh_cdf(double x, double sigma) { return x <= 0 ? 0.0 : 1.0 - std::exp(-x * x / (2 * sigma * sigma)); }

std::vector<double> component(const std::vector<Vec3>& v, int k) {
  std::vector<double> c(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) c[i] = v[i][k];
  return c;
}

double mean_of(const std::vector<double>& x) {
  double s = 0;
  for (double v : x) s += v;
  return s / static_cast<double>(x.size());
}

double var_of(const std::vector<double>& x) {
  const double m = mean_of(x);
  double s = 0;
  for (double v : x) s += (v - m) * (v - m);
  return s / static_cast<double>(x.size() - 1);
}

// Checks that the samples follow the fully diffuse (wall-Maxwellian, flux-weighted) emission.
void check_diffuse(const std::vector<Vec3>& l) {
  const double s = std::sqrt(constants::k_B * kTw / kMass);
  for (const Vec3& v : l) REQUIRE(v.z > 0.0);
  CHECK(ks(component(l, 0), normal_cdf, s) < 1.95);
  CHECK(ks(component(l, 1), normal_cdf, s) < 1.95);
  CHECK(ks(component(l, 2), rayleigh_cdf, s) < 1.95);
  // flux-weighted mean energy 2 k T_w  <=>  <|c|^2> = 4 s^2
  double e = 0, e2 = 0;
  for (const Vec3& v : l) {
    e += norm2(v);
    e2 += norm2(v) * norm2(v);
  }
  e /= kN;
  const double se = std::sqrt((e2 / kN - e * e) / kN);
  CHECK(std::fabs(e - 4 * s * s) < 5 * se);
}

}  // namespace

TEST_CASE("Maxwell f_d = 0 and CLL(0, 0) are specular", "[gsi]") {
  const Vec3 n = normalized(Vec3(0.3, -0.2, 1.0));
  for (const Vec3& v : {Vec3(-7000, 300, -2000), Vec3(1, 2, -3), Vec3(-1e4, 0, -1e-3)}) {
    const Vec3 expect = reflect(v, n);
    for (const SurfaceProperties& sp : {maxwell(0.0), cll(0.0, 0.0)}) {
      const auto out = test::eval_on_device<Vec3>("specular", 1000, ScatterBatch{sp, v, n, kMass});
      for (int i = 0; i < 1000; ++i) {
        INFO("model " << static_cast<int>(sp.model) << " sample " << i);
        REQUIRE(approx_equal(out(i), expect, 0.0, 1e-14));
        REQUIRE_THAT(norm2(out(i)), WithinRel(norm2(v), 1e-14));  // no energy exchange
      }
    }
  }
}

TEST_CASE("Diffuse Maxwell emission follows the wall Maxwellian, independent of the incident velocity", "[gsi][statistics]") {
  const Vec3 n = normalized(Vec3(-1.0, 2.0, 0.5));
  const Vec3 v_in = GENERATE(Vec3(-7800, 0, 0), Vec3(100, -50, -10));
  const Vec3 incident = dot(v_in, n) < 0 ? v_in : -v_in;  // must point into the wall
  check_diffuse(sample_local(maxwell(1.0), incident, n));
}

TEST_CASE("CLL(1, 1) equals fully diffuse emission", "[gsi][statistics]") {
  const Vec3 n(0, 0, 1);
  check_diffuse(sample_local(cll(1.0, 1.0), Vec3(-6000, 2500, -3000), n));
}

TEST_CASE("Maxwell mixture: diffuse fraction and specular part", "[gsi][statistics]") {
  const Vec3 n(0, 0, 1);
  const Vec3 v_in(-7000, 1000, -3000);
  const double fd = 0.3;
  const auto l = sample_local(maxwell(fd), v_in, n);
  const Vec3 spec = reflect(v_in, n);
  double specular = 0;
  for (const Vec3& v : l) specular += approx_equal(v, spec, 0.0, 1e-12) ? 1.0 : 0.0;
  const double p = 1.0 - fd;
  CHECK(std::fabs(specular / kN - p) < 5 * std::sqrt(p * (1 - p) / kN));
}

TEST_CASE("CLL moments: tangential momentum and normal/tangential energy accommodation", "[gsi][statistics]") {
  const auto [an, st] = GENERATE(table<double, double>({{0.6, 0.3}, {0.85, 0.65}, {0.2, 1.5}, {1.0, 2.0}}));
  INFO("alpha_n = " << an << ", sigma_t = " << st);
  const Vec3 n(0, 0, 1);
  const Vec3 c_in(-5000, 1200, -2500);
  const auto l = sample_local(cll(an, st), c_in, n);
  const double s2 = constants::k_B * kTw / kMass;
  for (const Vec3& v : l) REQUIRE(v.z > 0.0);
  for (int k = 0; k < 2; ++k) {
    const auto c = component(l, k);
    const double expected_mean = (1 - st) * c_in[k];
    const double expected_var = s2 * st * (2 - st);
    CHECK(std::fabs(mean_of(c) - expected_mean) < 5 * std::sqrt(expected_var / kN) + 1e-12 * std::fabs(c_in[k]));
    if (expected_var > 0) CHECK(std::fabs(var_of(c) / expected_var - 1) < 5 * std::sqrt(2.0 / kN));
  }
  // <c_n^2> = (1 - alpha_n) c_in,n^2 + 2 alpha_n s^2
  double m = 0, m2 = 0;
  for (const Vec3& v : l) {
    m += v.z * v.z;
    m2 += v.z * v.z * v.z * v.z;
  }
  m /= kN;
  const double se = std::sqrt((m2 / kN - m * m) / kN);
  CHECK(std::fabs(m - ((1 - an) * c_in.z * c_in.z + 2 * an * s2)) < 5 * se);
}

TEST_CASE("Wall velocity: scattering happens in the wall frame", "[gsi][statistics]") {
  const Vec3 n(0, 0, 1);
  SurfaceProperties sp = maxwell(1.0);
  sp.wall_velocity = Vec3(250.0, -100.0, 0.0);  // tangential wall motion
  const auto out = test::eval_on_device<Vec3>("moving_wall", kN, ScatterBatch{sp, Vec3(-3000, 0, -1000), n, kMass});
  double mx = 0, my = 0;
  for (int i = 0; i < kN; ++i) {
    mx += out(i).x;
    my += out(i).y;
    REQUIRE(out(i).z > 0.0);
  }
  const double s = std::sqrt(constants::k_B * kTw / kMass);
  CHECK(std::fabs(mx / kN - 250.0) < 5 * s / std::sqrt(kN));
  CHECK(std::fabs(my / kN + 100.0) < 5 * s / std::sqrt(kN));

  // Specular in the wall frame: v_r = v_w + reflect(v_i - v_w).
  SurfaceProperties sp0 = cll(0.0, 0.0);
  sp0.wall_velocity = sp.wall_velocity;
  const Vec3 v_in(-3000, 40, -1000);
  RandomStream rng(1, 1, 1, RngStream::Test);
  const Vec3 r = scatter_particle(v_in, sp0.wall_velocity, n, kMass, 0.0, sp0, rng).velocity_out;
  CHECK(approx_equal(r, sp0.wall_velocity + reflect(v_in - sp0.wall_velocity, n), 0.0, 1e-14));
}

TEST_CASE("Re-emission always points into the gas, also for extreme incidence", "[gsi]") {
  const Vec3 n = normalized(Vec3(1, 1, 1));
  const OrthonormalBasis b = make_basis_from_normal(n);
  for (const Vec3& local : {Vec3(1e4, 0, -1e-9), Vec3(0, 0, -1e-6), Vec3(0, 0, -3e4), Vec3(-2e4, 7e3, -1.0)}) {
    for (const SurfaceProperties& sp :
         {maxwell(1.0), maxwell(0.5), cll(1.0, 1.0), cll(0.0, 2.0), cll(1e-12, 1e-12), cll(0.5, 0.5)}) {
      const auto out = test::eval_on_device<Vec3>("extreme", 20000, ScatterBatch{sp, b.to_global(local), n, kMass});
      for (int i = 0; i < 20000; ++i) {
        REQUIRE(dot(out(i), n) > 0.0);
        REQUIRE(is_finite(out(i)));
      }
    }
  }
}

TEST_CASE("Host and device scattering agree", "[gsi][device]") {
  const Vec3 n = normalized(Vec3(0.1, 0.2, -1));
  const Vec3 v_in(300, -7000, 4000);
  for (const SurfaceProperties& sp : {maxwell(0.7), cll(0.85, 0.65)}) {
    const ScatterBatch f{sp, v_in, n, kMass};
    const auto dev = test::eval_on_device<Vec3>("host_device", 2000, f);
    for (int i = 0; i < 2000; ++i) REQUIRE(approx_equal(dev(i), f(i), 1e-9, 1e-12));
  }
}
