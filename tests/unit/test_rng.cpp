// M0.11 / M0.12: counter-based RNG - known answers, determinism, stream
// independence, distribution quality and numerical edge cases.
//
// All statistical tests use fixed seeds and are therefore deterministic; the
// acceptance thresholds are ~5 standard deviations, so a correct generator
// passes with overwhelming probability for any seed.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "DeviceTest.hpp"
#include "vleo/core/Random.hpp"

using namespace vleo;
using Catch::Matchers::WithinAbs;

namespace {

constexpr std::uint64_t kSeed = 0x1234'5678'9ABC'DEF0ull;

// Per-particle first draws (one stream per particle) for host/device comparison.
struct FirstDraws {
  std::uint32_t w0, w1, w2, w3, w4;  // spans two Philox blocks
  double u;
  double up;
  double z0, z1;
};

struct DrawFunctor {
  std::uint64_t seed;
  std::uint32_t event;
  KOKKOS_INLINE_FUNCTION FirstDraws operator()(int i) const {
    RandomStream r(seed, static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ull, event, RngStream::SurfaceInteraction);
    FirstDraws d;
    d.w0 = r.next_u32();
    d.w1 = r.next_u32();
    d.w2 = r.next_u32();
    d.w3 = r.next_u32();
    d.w4 = r.next_u32();
    d.u = r.uniform();
    d.up = r.uniform_pos();
    d.z0 = r.normal();
    d.z1 = r.normal();
    return d;
  }
};

// Many uniforms / normals from many short streams (as in a particle code).
constexpr int kStreams = 1 << 14;
constexpr int kPerStream = 64;

struct UniformBatch {
  std::uint64_t seed;
  KOKKOS_INLINE_FUNCTION double operator()(int idx) const {
    const int p = idx / kPerStream;
    const int k = idx % kPerStream;
    RandomStream r(seed, static_cast<std::uint64_t>(p), 7u, RngStream::Injection);
    double u = 0.0;
    for (int j = 0; j <= k; ++j) u = r.uniform();  // k-th draw of stream p
    return u;
  }
};

struct NormalBatch {
  std::uint64_t seed;
  KOKKOS_INLINE_FUNCTION double operator()(int p) const {
    RandomStream r(seed, static_cast<std::uint64_t>(p), 3u, RngStream::GasCollision);
    return r.normal();  // first normal of each stream (cos branch)
  }
};

struct NormalPairBatch {
  std::uint64_t seed;
  KOKKOS_INLINE_FUNCTION double operator()(int idx) const {
    RandomStream r(seed, static_cast<std::uint64_t>(idx / 2), 3u, RngStream::GasCollision);
    const double a = r.normal();
    const double b = r.normal();
    return (idx % 2 == 0) ? a : b;
  }
};

// Draws of DrawFunctor evaluated in reversed index order.
struct Reversed {
  DrawFunctor f;
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(int i) const { return f(n - 1 - i).u; }
};

double mean(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) s += v;
  return s / static_cast<double>(x.size());
}

double central_moment(const std::vector<double>& x, double m, int k) {
  double s = 0.0;
  for (double v : x) s += std::pow(v - m, k);
  return s / static_cast<double>(x.size());
}

// Regularised upper incomplete gamma is overkill here: for large dof the
// chi-square statistic is approximately normal, (X - dof) / sqrt(2 dof).
double chi_square_z(const std::vector<double>& u, int bins) {
  std::vector<double> count(static_cast<std::size_t>(bins), 0.0);
  for (double v : u) count[static_cast<std::size_t>(std::min(bins - 1, static_cast<int>(v * bins)))] += 1.0;
  const double expected = static_cast<double>(u.size()) / bins;
  double chi2 = 0.0;
  for (double c : count) chi2 += (c - expected) * (c - expected) / expected;
  const double dof = bins - 1;
  return (chi2 - dof) / std::sqrt(2.0 * dof);
}

// Kolmogorov-Smirnov statistic sqrt(n) D_n against U(0,1).
double ks_uniform(std::vector<double> u) {
  std::sort(u.begin(), u.end());
  const double n = static_cast<double>(u.size());
  double d = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    const double di = static_cast<double>(i);
    d = std::max(d, std::max((di + 1.0) / n - u[i], u[i] - di / n));
  }
  return std::sqrt(n) * d;
}

std::vector<double> to_vector(const Kokkos::View<double*, Kokkos::HostSpace>& v) {
  return std::vector<double>(v.data(), v.data() + v.extent(0));
}

}  // namespace

TEST_CASE("Philox4x32-10 matches the Random123 known-answer vectors", "[rng]") {
  using P = Philox4x32x10;
  {
    const auto r = P::generate({{0u, 0u, 0u, 0u}}, {{0u, 0u}});
    CHECK(r.v[0] == 0x6627e8d5u);
    CHECK(r.v[1] == 0xe169c58du);
    CHECK(r.v[2] == 0xbc57ac4cu);
    CHECK(r.v[3] == 0x9b00dbd8u);
  }
  {
    const auto r = P::generate({{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}}, {{0xffffffffu, 0xffffffffu}});
    CHECK(r.v[0] == 0x408f276du);
    CHECK(r.v[1] == 0x41c83b0eu);
    CHECK(r.v[2] == 0xa20bc7c6u);
    CHECK(r.v[3] == 0x6d5451fdu);
  }
  {
    const auto r = P::generate({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}}, {{0xa4093822u, 0x299f31d0u}});
    CHECK(r.v[0] == 0xd16cfe09u);
    CHECK(r.v[1] == 0x94fdccebu);
    CHECK(r.v[2] == 0x5001e420u);
    CHECK(r.v[3] == 0x24126ea1u);
  }
}

TEST_CASE("RandomStream maps (seed, particle, event, stream) onto the Philox input", "[rng]") {
  const std::uint64_t seed = 0xa4093822'299f31d0ull;      // key = {0x299f31d0, 0xa4093822}
  const std::uint64_t particle = 0x85a308d3'243f6a88ull;  // ctr0 = 0x243f6a88, ctr1 = 0x85a308d3
  RandomStream r(seed, particle, 0x13198a2eu, static_cast<std::uint16_t>(0x0370));
  const auto ref = Philox4x32x10::generate({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03700000u}},
                                           {{0x299f31d0u, 0xa4093822u}});
  for (int i = 0; i < 4; ++i) CHECK(r.next_u32() == ref.v[i]);
  // Second block: block index 1 in the low 16 bits of ctr[3].
  const auto ref2 = Philox4x32x10::generate({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03700001u}},
                                            {{0x299f31d0u, 0xa4093822u}});
  for (int i = 0; i < 4; ++i) CHECK(r.next_u32() == ref2.v[i]);
  CHECK(r.blocks_used() == 2);
}

TEST_CASE("RandomStream is deterministic and replayable", "[rng]") {
  RandomStream a(kSeed, 42, 3, RngStream::SurfaceInteraction);
  RandomStream b(kSeed, 42, 3, RngStream::SurfaceInteraction);
  for (int i = 0; i < 1000; ++i) {
    REQUIRE(a.next_u32() == b.next_u32());
    REQUIRE(a.uniform() == b.uniform());
    REQUIRE(a.normal() == b.normal());
  }
  // next_u64 = (first word << 32) | second word
  RandomStream c(kSeed, 1, 0, RngStream::Test), d(kSeed, 1, 0, RngStream::Test);
  const std::uint64_t hi = d.next_u32(), lo = d.next_u32();
  CHECK(c.next_u64() == ((hi << 32) | lo));
}

TEST_CASE("Different seeds, particles, events and streams give different sequences", "[rng]") {
  std::set<std::vector<std::uint32_t>> seen;
  auto first_words = [](std::uint64_t seed, std::uint64_t p, std::uint32_t e, std::uint16_t s) {
    RandomStream r(seed, p, e, s);
    return std::vector<std::uint32_t>{r.next_u32(), r.next_u32(), r.next_u32(), r.next_u32()};
  };
  const std::uint64_t seeds[] = {0ull, 1ull, 1ull << 32, ~0ull};
  const std::uint64_t particles[] = {0ull, 1ull, 1ull << 32, (1ull << 32) + 1, ~0ull};
  const std::uint32_t events[] = {0u, 1u, 0xFFFFFFFFu};
  const std::uint16_t streams[] = {0, 1, 2, 3, 0xFFFF};
  std::size_t combos = 0;
  for (auto s : seeds)
    for (auto p : particles)
      for (auto e : events)
        for (auto st : streams) {
          seen.insert(first_words(s, p, e, st));
          ++combos;
        }
  CHECK(seen.size() == combos);
}

TEST_CASE("Host and device produce bitwise identical integer and uniform draws", "[rng][device]") {
  constexpr int n = 4096;
  const DrawFunctor f{kSeed, 11u};
  const auto dev = test::eval_on_device<FirstDraws>("rng_first_draws", n, f);
  for (int i = 0; i < n; ++i) {
    const FirstDraws h = f(i);
    INFO("particle " << i);
    REQUIRE(dev(i).w0 == h.w0);
    REQUIRE(dev(i).w1 == h.w1);
    REQUIRE(dev(i).w2 == h.w2);
    REQUIRE(dev(i).w3 == h.w3);
    REQUIRE(dev(i).w4 == h.w4);
    REQUIRE(dev(i).u == h.u);
    REQUIRE(dev(i).up == h.up);
    // log/sqrt/sin/cos may differ in the last ulps between math libraries.
    REQUIRE_THAT(dev(i).z0, WithinAbs(h.z0, 1e-13));
    REQUIRE_THAT(dev(i).z1, WithinAbs(h.z1, 1e-13));
  }
}

TEST_CASE("Results do not depend on the parallel execution order", "[rng][device]") {
  // Evaluate the same draws with a reversed index mapping; every particle's
  // values must be identical irrespective of which thread computed them.
  constexpr int n = 2048;
  const DrawFunctor f{kSeed, 5u};
  const auto fwd = test::eval_on_device<FirstDraws>("rng_fwd", n, f);
  const auto rev = test::eval_on_device<double>("rng_rev", n, Reversed{f, n});
  for (int i = 0; i < n; ++i) REQUIRE(fwd(i).u == rev(n - 1 - i));
}

TEST_CASE("Uniform variates follow U[0,1)", "[rng][statistics]") {
  const auto u = to_vector(test::eval_on_device<double>("uniform_batch", kStreams * kPerStream, UniformBatch{kSeed}));
  const double n = static_cast<double>(u.size());
  for (double v : u) REQUIRE((v >= 0.0 && v < 1.0));

  const double m = mean(u);
  const double var = central_moment(u, m, 2);
  // sigma(mean) = sqrt(1/12 / n); sigma(var) = sqrt((1/80 - 1/144) / n)
  CHECK(std::fabs(m - 0.5) < 5.0 * std::sqrt(1.0 / 12.0 / n));
  CHECK(std::fabs(var - 1.0 / 12.0) < 5.0 * std::sqrt((1.0 / 80.0 - 1.0 / 144.0) / n));
  CHECK(std::fabs(chi_square_z(u, 1000)) < 5.0);
  // KS: P(sqrt(n) D > 1.95) ~ 0.001
  CHECK(ks_uniform(u) < 1.95);

  // Serial correlation between consecutive draws of the same stream and
  // between the same draw index of neighbouring streams.
  double c_seq = 0.0, c_par = 0.0;
  for (int p = 0; p < kStreams; ++p)
    for (int k = 0; k + 1 < kPerStream; ++k)
      c_seq += (u[static_cast<std::size_t>(p * kPerStream + k)] - 0.5) *
               (u[static_cast<std::size_t>(p * kPerStream + k + 1)] - 0.5);
  for (int p = 0; p + 1 < kStreams; ++p)
    for (int k = 0; k < kPerStream; ++k)
      c_par += (u[static_cast<std::size_t>(p * kPerStream + k)] - 0.5) *
               (u[static_cast<std::size_t>((p + 1) * kPerStream + k)] - 0.5);
  const double n_seq = kStreams * (kPerStream - 1.0), n_par = (kStreams - 1.0) * kPerStream;
  // correlation coefficient = c / (n var); sigma ~ 1/sqrt(n)
  CHECK(std::fabs(c_seq / n_seq * 12.0) < 5.0 / std::sqrt(n_seq));
  CHECK(std::fabs(c_par / n_par * 12.0) < 5.0 / std::sqrt(n_par));
}

TEST_CASE("Normal variates follow N(0,1)", "[rng][statistics]") {
  constexpr int n = 1 << 20;
  for (int variant = 0; variant < 2; ++variant) {
    // variant 0: first normal of each stream; variant 1: both values of each Box-Muller pair.
    const auto z = to_vector(variant == 0 ? test::eval_on_device<double>("normal_batch", n, NormalBatch{kSeed})
                                          : test::eval_on_device<double>("normal_pairs", n, NormalPairBatch{kSeed}));
    INFO("variant " << variant);
    const double m = mean(z);
    const double var = central_moment(z, m, 2);
    const double skew = central_moment(z, m, 3) / std::pow(var, 1.5);
    const double kurt = central_moment(z, m, 4) / (var * var);
    const double dn = static_cast<double>(n);
    CHECK(std::fabs(m) < 5.0 / std::sqrt(dn));
    CHECK(std::fabs(var - 1.0) < 5.0 * std::sqrt(2.0 / dn));
    CHECK(std::fabs(skew) < 5.0 * std::sqrt(6.0 / dn));
    CHECK(std::fabs(kurt - 3.0) < 5.0 * std::sqrt(24.0 / dn));
    // P(|Z| < 1) = erf(1/sqrt 2)
    double inside = 0.0;
    for (double v : z) inside += std::fabs(v) < 1.0 ? 1.0 : 0.0;
    const double p = std::erf(1.0 / std::sqrt(2.0));
    CHECK(std::fabs(inside / dn - p) < 5.0 * std::sqrt(p * (1.0 - p) / dn));
    for (double v : z) REQUIRE((std::isfinite(v) && std::fabs(v) < 8.6));
  }
  // Pair members are uncorrelated.
  const auto z = to_vector(test::eval_on_device<double>("normal_pairs_corr", n, NormalPairBatch{kSeed}));
  double c = 0.0;
  for (std::size_t i = 0; i + 1 < z.size(); i += 2) c += z[i] * z[i + 1];
  CHECK(std::fabs(c / (n / 2)) < 5.0 / std::sqrt(n / 2.0));
}

TEST_CASE("Uniform conversion edge cases", "[rng][edge]") {
  constexpr double eps53 = 0x1.0p-53;
  static_assert(RandomStream::uniform_from_words(0u, 0u) == 0.0);
  static_assert(RandomStream::uniform_from_words(0xFFFFFFFFu, 0xFFFFFFFFu) == 1.0 - eps53);
  static_assert(RandomStream::uniform_from_words(0u, 0x40u) == eps53);  // smallest non-zero
  static_assert(RandomStream::uniform_from_words(0u, 0x3Fu) == 0.0);    // discarded low bits
  static_assert(RandomStream::uniform_from_words(0x80000000u, 0u) == 0.5);
  static_assert(RandomStream::uniform_from_words(0xFFFFFFFFu, 0xFFFFFFFFu) < 1.0);

  // uniform_pos = 1 - uniform lies in (0, 1]; its extreme values keep log finite.
  CHECK(1.0 - RandomStream::uniform_from_words(0xFFFFFFFFu, 0xFFFFFFFFu) == eps53);
  double z0 = 0.0, z1 = 0.0;
  RandomStream::box_muller(eps53, 0.0, z0, z1);  // largest possible radius
  CHECK(std::isfinite(z0));
  CHECK(std::isfinite(z1));
  CHECK_THAT(z0, WithinAbs(std::sqrt(-2.0 * std::log(eps53)), 1e-12));
  CHECK(z0 < 8.6);
  RandomStream::box_muller(1.0, 0.25, z0, z1);  // radius exactly 0
  CHECK(z0 == 0.0);
  CHECK(z1 == 0.0);
}

TEST_CASE("Extreme identifiers are valid inputs", "[rng][edge]") {
  RandomStream a(~0ull, ~0ull, 0xFFFFFFFFu, static_cast<std::uint16_t>(0xFFFF));
  RandomStream b(~0ull, ~0ull, 0xFFFFFFFFu, static_cast<std::uint16_t>(0xFFFE));
  bool differ = false;
  for (int i = 0; i < 8; ++i) {
    const double ua = a.uniform(), ub = b.uniform();
    REQUIRE((ua >= 0.0 && ua < 1.0));
    differ = differ || (ua != ub);
  }
  CHECK(differ);
  CHECK(!a.exhausted());
}

TEST_CASE("Stream exhaustion is detected", "[rng][edge]") {
  RandomStream r(kSeed, 9, 9, RngStream::Test);
  const std::uint32_t first = r.next_u32();
  const std::uint64_t words = 4ull * RandomStream::kBlocksPerStream;
  for (std::uint64_t i = 1; i < words; ++i) r.next_u32();
  CHECK(r.blocks_used() == RandomStream::kBlocksPerStream);
  CHECK(!r.exhausted());  // exactly 2^16 blocks used: still the unique sequence
  const std::uint32_t wrapped = r.next_u32();
  CHECK(r.exhausted());
  CHECK(wrapped == first);  // documented wrap-around
}
