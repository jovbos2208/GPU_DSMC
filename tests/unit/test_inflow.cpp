// M1.10-M1.12: particle storage, domain, flux-weighted inflow distribution.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "DeviceTest.hpp"
#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/particles/Inflow.hpp"
#include "vleo/particles/ParticleStorage.hpp"
#include "vleo/particles/Transport.hpp"

using namespace vleo;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kSqrtPi = 1.77245385090551602730;

struct NormalSpeedSampler {
  double s, p_a, exp_s2;
  KOKKOS_INLINE_FUNCTION double operator()(int i) const {
    RandomStream rng(99, static_cast<std::uint64_t>(i), 0, RngStream::Injection);
    return sample_flux_normal(s, p_a, exp_s2, rng);
  }
};

struct InflowSampler {
  DeviceInflow inflow;
  KOKKOS_INLINE_FUNCTION InflowSample operator()(int i) const {
    RandomStream rng(5, static_cast<std::uint64_t>(i), 0, RngStream::Injection);
    return inflow.sample(rng);
  }
};

double ks_statistic(std::vector<double> x, double s) {
  std::sort(x.begin(), x.end());
  const double n = static_cast<double>(x.size());
  double d = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double f = flux_normal_cdf(x[i], s);
    d = std::max({d, (static_cast<double>(i) + 1.0) / n - f, f - static_cast<double>(i) / n});
  }
  return std::sqrt(n) * d;
}

GasSection make_gas(const Vec3& u, std::vector<std::pair<std::string, double>> species) {
  GasSection g;
  g.number_density_m3 = 1e14;
  g.translational_temperature_K = 900.0;
  g.bulk_velocity_body_mps = u;
  for (const auto& [name, x] : species) g.species.push_back({name, x, find_species(name)->mass_kg, find_species(name)->vhs, find_species(name)->modes});
  return g;
}

}  // namespace

TEST_CASE("Flux factor: limits, symmetry and tail accuracy", "[inflow]") {
  CHECK_THAT(flux_factor(0.0), WithinAbs(1.0, 1e-15));
  // F(s) - F(-s) = 2 sqrt(pi) s (net flux of a drifting gas)
  for (double s : {0.1, 0.7, 2.0, 5.0})
    CHECK_THAT(flux_factor(s) - flux_factor(-s), WithinRel(2.0 * kSqrtPi * s, 1e-13));
  // s >> 1: F ~ 2 sqrt(pi) s
  CHECK_THAT(flux_factor(8.0), WithinRel(2.0 * kSqrtPi * 8.0, 1e-15));
  // s << 0: F ~ exp(-s^2) / (2 s^2) (1 - 3/(2 s^2) + 15/(4 s^4) - 105/(8 s^6))
  for (double s : {-6.0, -8.0, -12.0}) {
    const double s2 = s * s;
    const double asym = std::exp(-s2) / (2 * s2) * (1 - 3 / (2 * s2) + 15 / (4 * s2 * s2) - 105 / (8 * s2 * s2 * s2));
    INFO("s = " << s);
    // truncation error of the series: next term 945 / (16 s^8) relative (3.5e-5 at s = -6)
    CHECK_THAT(flux_factor(s), WithinRel(asym, 1e-4));
    CHECK(flux_factor(s) > 0.0);
  }
  // number flux of a resting gas: n c / (2 sqrt pi) = n sqrt(kT / (2 pi m))
  const double m = find_species("O")->mass_kg, T = 900.0, c = std::sqrt(2 * constants::k_B * T / m);
  CHECK_THAT(number_flux(1e14, c, 0.0), WithinRel(1e14 * std::sqrt(constants::k_B * T / (2 * constants::pi * m)), 1e-14));
}

TEST_CASE("Flux-weighted normal speed matches the analytic distribution", "[inflow][statistics][device]") {
  const double s = GENERATE(-1.5, -0.3, 0.0, 0.4, 1.17, 3.0, 8.0);
  INFO("s = " << s);
  const double e = std::exp(-s * s);
  double p_a = 0.0;
  if (s >= 0.0) {
    const double ma = 1.0 - 0.5 * e, mb = 0.5 * kSqrtPi * s * std::erfc(-s);
    p_a = ma / (ma + mb);
  }
  constexpr int n = 400000;
  const auto z = test::eval_on_device<double>("normal_speed", n, NormalSpeedSampler{s, p_a, e});
  std::vector<double> v(z.data(), z.data() + n);
  for (double x : v) REQUIRE((x > 0.0 && std::isfinite(x)));
  // KS: P(sqrt(n) D > 1.95) ~ 0.1 %
  CHECK(ks_statistic(v, s) < 1.95);
  // Mean: <z> = int z^2 e^{-(z-s)^2} / int z e^{-(z-s)^2}; numerator closed form:
  // int_0^inf z^2 e^{-(z-s)^2} dz = (s e^{-s^2} + sqrt(pi)/2 (1 + 2 s^2)(1 + erf s)) / 2
  const double num = 0.5 * (s * e + 0.5 * kSqrtPi * (1 + 2 * s * s) * std::erfc(-s));
  const double den = 0.5 * flux_factor(s);
  double mean = 0.0, m2 = 0.0;
  for (double x : v) {
    mean += x;
    m2 += x * x;
  }
  mean /= n;
  const double var = m2 / n - mean * mean;
  CHECK(std::fabs(mean - num / den) < 5.0 * std::sqrt(var / n));
}

TEST_CASE("Domain construction", "[inflow]") {
  const Aabb mesh{{-1, -2, -3}, {1, 2, 3}};
  DomainSection cfg;
  Domain d = make_domain(cfg, mesh, 1e-6);  // default margin: 5 % of the largest extent (6 m)
  CHECK(approx_equal(d.box.lo, Vec3(-1.3, -2.3, -3.3), 1e-15));
  CHECK(approx_equal(d.box.hi, Vec3(1.3, 2.3, 3.3), 1e-15));
  cfg.margin_m = 0.5;
  d = make_domain(cfg, mesh, 1e-6);
  CHECK(d.box.hi == Vec3(1.5, 2.5, 3.5));
  CHECK_THAT(d.area_of_face(0), WithinRel(5.0 * 7.0, 1e-15));
  CHECK_THAT(d.area_of_face(5), WithinRel(3.0 * 5.0, 1e-15));
  CHECK_THAT(d.exit_distance({0, 0, 0}, {1, 0, 0}), WithinRel(1.5, 1e-15));
  CHECK_THAT(d.exit_distance({0, 0, 0}, {-1, -1, -1}), WithinRel(1.5, 1e-15));
  CHECK_THAT(d.exit_distance({0, 0, 0}, {0, 0, -7}), WithinRel(0.5, 1e-15));
  CHECK(d.exit_distance({0, 0, 0}, {0, 0, 0}) == Kokkos::infinity_v<double>);

  DomainSection box;
  box.box_min_m = Vec3(-5, -5, -5);
  box.box_max_m = Vec3(5, 5, 3.0);  // touches the body in z
  try {
    (void)make_domain(box, mesh, 1e-6);
    FAIL("accepted");
  } catch (const GeometryError& e) {
    CHECK_THAT(std::string(e.what()), ContainsSubstring("along axis z"));
  }
  box.box_max_m = Vec3(5, 5, 3.1);
  CHECK(make_domain(box, mesh, 1e-6).box.lo == Vec3(-5, -5, -5));
  cfg.margin_m = 1e-9;
  CHECK_THROWS_AS(make_domain(cfg, mesh, 1e-6), GeometryError);
}

TEST_CASE("Inflow: channel rates, face/species frequencies, velocities and positions", "[inflow][statistics][device]") {
  DomainSection cfg;
  cfg.margin_m = 0.25;
  const Domain dom = make_domain(cfg, Aabb{{-1, -0.5, -0.25}, {1, 0.5, 0.25}}, 1e-6);
  const Vec3 u(-1200.0, 400.0, -150.0);  // generic direction, moderate speed ratios
  const GasSection gas = make_gas(u, {{"O", 0.7}, {"He", 0.3}});
  const InflowModel model = InflowModel::create(dom, gas);
  REQUIRE(model.channels.size() == 12);

  // Net flux: (in through all faces) - (out through all faces) = 0 for a
  // uniform gas, i.e. sum over faces of the signed flux n u . e A vanishes;
  // equivalently in - out per axis equals n u_axis A_axis.
  for (const auto& ch : model.channels) {
    const auto& sp = gas.species[ch.species];
    const double c = std::sqrt(2 * constants::k_B * gas.translational_temperature_K / sp.mass_kg);
    CHECK_THAT(ch.thermal_speed, WithinRel(c, 1e-15));
    const double n_s = gas.number_density_m3 * sp.mole_fraction;
    CHECK_THAT(ch.rate_per_s, WithinRel(number_flux(n_s, c, ch.s) * dom.area_of_face(ch.face), 1e-15));
  }
  for (int axis = 0; axis < 3; ++axis) {
    for (std::uint32_t sp = 0; sp < 2; ++sp) {
      double in_lo = 0, in_hi = 0;
      for (const auto& ch : model.channels)
        if (ch.face / 2 == axis && ch.species == sp) (ch.face % 2 == 0 ? in_lo : in_hi) = ch.rate_per_s;
      const double n_s = gas.number_density_m3 * gas.species[sp].mole_fraction;
      CHECK_THAT(in_lo - in_hi, WithinRel(n_s * u[axis] * dom.area_of_face(2 * axis), 1e-12));
    }
  }

  const DeviceInflow dev = upload(model);
  constexpr int n = 600000;
  const auto smp = test::eval_on_device<InflowSample>("inflow_samples", n, InflowSampler{dev});

  // Which face did each sample enter through? (position exactly on that face, velocity inward)
  std::vector<double> count(12, 0.0);
  std::vector<double> sum_t(12 * 3, 0.0), sum_t2(12 * 3, 0.0);
  std::vector<double> pos_sum(12 * 3, 0.0);
  for (int i = 0; i < n; ++i) {
    const InflowSample& s = smp(i);
    int face = -1;
    for (int f = 0; f < 6 && face < 0; ++f) {
      const int a = f / 2;
      const double plane = f % 2 == 0 ? dom.box.lo[a] : dom.box.hi[a];
      const double inward = f % 2 == 0 ? s.velocity[a] : -s.velocity[a];
      if (s.position[a] == plane && inward > 0.0) face = f;
    }
    REQUIRE(face >= 0);
    REQUIRE(s.species < 2);
    REQUIRE(dom.box.contains(s.position));
    const int ch = face * 2 + static_cast<int>(s.species);
    count[static_cast<std::size_t>(ch)] += 1.0;
    for (int k = 0; k < 3; ++k) {
      sum_t[static_cast<std::size_t>(ch * 3 + k)] += s.velocity[k];
      sum_t2[static_cast<std::size_t>(ch * 3 + k)] += s.velocity[k] * s.velocity[k];
      pos_sum[static_cast<std::size_t>(ch * 3 + k)] += s.position[k];
    }
  }
  // Channel frequencies vs. rates (chi-square, 11 dof; 5 sigma of the normal approx.)
  double chi2 = 0.0;
  for (std::size_t ch = 0; ch < 12; ++ch) {
    const double expected = n * model.channels[ch].rate_per_s / model.total_rate_per_s;
    chi2 += (count[ch] - expected) * (count[ch] - expected) / expected;
  }
  CHECK((chi2 - 11.0) / std::sqrt(22.0) < 5.0);

  // Tangential components: mean u_t, variance kT/m; positions uniform on the face.
  for (std::size_t ch = 0; ch < 12; ++ch) {
    const int face = static_cast<int>(ch / 2), axis = face / 2;
    const double cnt = count[ch];
    if (cnt < 1000) continue;
    const double sigma = model.channels[ch].thermal_speed / std::sqrt(2.0);
    for (int k = 0; k < 3; ++k) {
      if (k == axis) continue;
      INFO("channel " << ch << " component " << k);
      const double mean = sum_t[ch * 3 + static_cast<std::size_t>(k)] / cnt;
      const double var = sum_t2[ch * 3 + static_cast<std::size_t>(k)] / cnt - mean * mean;
      CHECK(std::fabs(mean - u[k]) < 5.0 * sigma / std::sqrt(cnt));
      CHECK(std::fabs(var / (sigma * sigma) - 1.0) < 5.0 * std::sqrt(2.0 / cnt));
      const double pos_mean = pos_sum[ch * 3 + static_cast<std::size_t>(k)] / cnt;
      const double half = 0.5 * (dom.box.hi[k] - dom.box.lo[k]);
      CHECK(std::fabs(pos_mean - dom.box.center()[k]) < 5.0 * half / std::sqrt(3.0 * cnt));
    }
  }
}

TEST_CASE("Particle storage", "[particles][device]") {
  ParticleStorage p(1000);
  CHECK(p.capacity() == 1000);
  p.clear();
  const TransportCounters c = count_particles(p, 1000);
  CHECK(c.active == 1000);  // free slots count as not finished
  CHECK(c.surface_interactions == 0);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.status);
  CHECK(h(999) == static_cast<std::uint8_t>(ParticleStatus::Free));
  CHECK(std::string(to_string(ParticleStatus::Truncated)) == "truncated");
}
