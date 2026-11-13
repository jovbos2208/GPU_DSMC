// M3.4, M3.6, M3.14: VHS cross-section, equilibrium averages, binary collisions.
//
// Tolerances (docs/dsmc.md section 7): conservation 1e-13 relative,
// statistics 5 standard errors / KS sqrt(n) D < 1.95.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "DeviceTest.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/dsmc/Vhs.hpp"

using namespace vleo;
using Catch::Matchers::WithinRel;

namespace {

struct Pair {
  Vec3 a, b;
};

struct CollideBatch {
  double m_p, m_q;
  KOKKOS_INLINE_FUNCTION Pair operator()(int i) const {
    RandomStream pre(5, static_cast<std::uint64_t>(i), 0, RngStream::Test);
    Vec3 a(3000 * pre.normal(), 500 * pre.normal(), -800 * pre.normal());
    Vec3 b(-200 * pre.normal(), 1200 * pre.normal(), 90 * pre.normal());
    RandomStream rng(5, static_cast<std::uint64_t>(i), 1, RngStream::GasCollision);
    vhs_collide(a, b, m_p, m_q, rng);
    return {a, b};
  }
};

Pair pre_collision(int i) {
  RandomStream pre(5, static_cast<std::uint64_t>(i), 0, RngStream::Test);
  Vec3 a(3000 * pre.normal(), 500 * pre.normal(), -800 * pre.normal());
  Vec3 b(-200 * pre.normal(), 1200 * pre.normal(), 90 * pre.normal());
  return {a, b};
}

}  // namespace

TEST_CASE("VHS cross-section: formula, hard-sphere limit, temperature dependence", "[vhs]") {
  const SpeciesInfo ar = *find_species("Ar");
  const VhsPair p = make_vhs_pair(ar.mass_kg, ar.vhs, ar.mass_kg, ar.vhs);
  const double mr = 0.5 * ar.mass_kg, d = ar.vhs.d_ref_m, w = ar.vhs.omega;
  for (double g : {100.0, 600.0, 3000.0, 15000.0}) {
    const double ref = constants::pi * d * d * std::pow(2 * constants::k_B * 273.0 / (mr * g * g), w - 0.5) /
                       std::tgamma(2.5 - w);
    CHECK_THAT(p.sigma(g), WithinRel(ref, 1e-13));
    CHECK_THAT(p.sigma_g(g), WithinRel(ref * g, 1e-13));
  }
  CHECK(p.sigma_g(0.0) == 0.0);
  // omega = 1/2 is the hard sphere: sigma = pi d^2 for all g
  VhsParameters hs{true, 3e-10, 0.5, 273.0, 3e-10, 1.0, "test"};
  const VhsPair h = make_vhs_pair(1e-26, hs, 1e-26, hs);
  CHECK_THAT(h.sigma(10.0), WithinRel(constants::pi * 9e-20, 1e-14));
  CHECK_THAT(h.sigma(1e4), WithinRel(constants::pi * 9e-20, 1e-14));
  CHECK_THAT(vhs_mean_free_path(1e-26, hs, 1e20, 300.0), WithinRel(1.0 / (std::sqrt(2.0) * constants::pi * 9e-20 * 1e20), 1e-13));
}

TEST_CASE("Equilibrium <sigma g> agrees with numerical integration over the Maxwellian", "[vhs]") {
  for (const char* name : {"He", "Ar"}) {
    const SpeciesInfo s = *find_species(name);
    const VhsPair p = make_vhs_pair(s.mass_kg, s.vhs, s.mass_kg, s.vhs);
    for (double T : {100.0, 300.0, 2000.0}) {
      // relative speed distribution: f(g) = 4 pi (m_r / 2 pi k T)^1.5 g^2 exp(-m_r g^2 / 2 k T)
      const double mr = 0.5 * s.mass_kg, a = mr / (2 * constants::k_B * T);
      const double gmax = 12.0 / std::sqrt(a);
      const int n = 200000;
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        const double g = (i + 0.5) * gmax / n;
        sum += p.sigma_g(g) * 4 * constants::pi * std::pow(a / constants::pi, 1.5) * g * g * std::exp(-a * g * g);
      }
      sum *= gmax / n;
      INFO(name << " T = " << T);
      CHECK_THAT(vhs_mean_sigma_g(s.mass_kg, s.vhs, s.mass_kg, s.vhs, T), WithinRel(sum, 1e-8));
    }
  }
}

TEST_CASE("Binary collisions conserve momentum and energy exactly (equal and unequal masses)", "[vhs][device]") {
  const double mHe = find_species("He")->mass_kg, mAr = find_species("Ar")->mass_kg;
  for (const auto& [mp, mq] : {std::pair{mAr, mAr}, std::pair{mHe, mAr}, std::pair{mAr, mHe}}) {
    constexpr int n = 100000;
    const auto post = test::eval_on_device<Pair>("collide", n, CollideBatch{mp, mq});
    for (int i = 0; i < n; ++i) {
      const Pair a = pre_collision(i);
      const Pair b = post(i);
      const Vec3 P0 = mp * a.a + mq * a.b, P1 = mp * b.a + mq * b.b;
      const double E0 = 0.5 * (mp * norm2(a.a) + mq * norm2(a.b)), E1 = 0.5 * (mp * norm2(b.a) + mq * norm2(b.b));
      const double pscale = mp * norm(a.a) + mq * norm(a.b);
      REQUIRE(max_abs_component(P1 - P0) <= 1e-13 * pscale);
      REQUIRE(std::fabs(E1 - E0) <= 1e-13 * E0);
      REQUIRE_THAT(norm(b.a - b.b), WithinRel(norm(a.a - a.b), 1e-13));  // |g| preserved
    }
  }
}

TEST_CASE("Post-collision relative velocities are isotropic", "[vhs][statistics][device]") {
  const double m = find_species("Ar")->mass_kg;
  constexpr int n = 200000;
  const auto post = test::eval_on_device<Pair>("isotropy", n, CollideBatch{m, m});
  std::vector<double> cz(n);
  double mean_dot = 0.0;
  for (int i = 0; i < n; ++i) {
    const Vec3 g_new = normalized(post(i).a - post(i).b);
    const Pair pre = pre_collision(i);
    const Vec3 g_old = normalized(pre.a - pre.b);
    cz[static_cast<std::size_t>(i)] = g_new.z;
    mean_dot += dot(g_new, g_old);
  }
  // cos(theta) of an isotropic direction is uniform on [-1, 1]
  std::sort(cz.begin(), cz.end());
  double d = 0.0;
  for (int i = 0; i < n; ++i) {
    const double f = 0.5 * (cz[static_cast<std::size_t>(i)] + 1.0);
    d = std::max({d, (i + 1.0) / n - f, f - static_cast<double>(i) / n});
  }
  CHECK(std::sqrt(static_cast<double>(n)) * d < 1.95);
  // no memory of the incident direction: <g'.g> = 0 (variance 1/3)
  CHECK(std::fabs(mean_dot / n) < 5.0 * std::sqrt(1.0 / 3.0 / n));
}
