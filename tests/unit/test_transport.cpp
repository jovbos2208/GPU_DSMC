// M1.13-M1.15 and Gate M1: transport to surface hit or domain exit, multiple
// reflections with truncation, complete particle balance.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <set>

#include "DeviceTest.hpp"
#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/particles/Transport.hpp"

using namespace vleo;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Re-emits in a random cosine-distributed direction with the incident speed.
// Stresses self-hit avoidance: re-emission close to edges and corners in all
// directions, including grazing ones.
struct RandomCosineInteraction {
  KOKKOS_INLINE_FUNCTION Vec3 operator()(const SurfaceEvent& e, RandomStream& rng) const {
    const OrthonormalBasis b = make_basis_from_normal(e.normal_gas);
    const double r2 = rng.uniform();
    const double phi = constants::two_pi * rng.uniform();
    const double sr = Kokkos::sqrt(r2);
    const Vec3 local(sr * Kokkos::cos(phi), sr * Kokkos::sin(phi), Kokkos::sqrt(1.0 - r2));
    if (!(local.z > 0.0)) return e.normal_gas * norm(e.velocity_in);  // r2 == 1 has probability 2^-53
    return norm(e.velocity_in) * b.to_global(local);
  }
};

// Deliberately broken model: re-emits into the surface.
struct IntoTheWallInteraction {
  KOKKOS_INLINE_FUNCTION Vec3 operator()(const SurfaceEvent& e, RandomStream&) const { return -e.normal_gas; }
};

GasSection make_gas(const Vec3& u) {
  GasSection g;
  g.number_density_m3 = 1e14;
  g.translational_temperature_K = 900.0;
  g.bulk_velocity_body_mps = u;
  g.species.push_back({"O", 0.8, find_species("O")->mass_kg});
  g.species.push_back({"He", 0.2, find_species("He")->mass_kg});
  return g;
}

struct Case {
  Geometry geometry;
  InflowModel inflow;
  TransportSetup setup;
};

Case make_case(const TriangleSoup& soup, const Vec3& u, double margin, bool require_closed = true) {
  Case c;
  MeshOptions o;
  o.require_closed = require_closed;
  c.geometry = make_geometry(soup, o);
  DomainSection d;
  d.margin_m = margin;
  const Domain dom = make_domain(d, c.geometry.host.bbox, 1e3 * c.geometry.device.surface_offset_m);
  c.inflow = InflowModel::create(dom, make_gas(u));
  c.setup.geometry = c.geometry.device;
  c.setup.domain = dom;
  c.setup.inflow = upload(c.inflow);
  c.setup.seed = 2024;
  c.setup.max_interactions = 100;
  c.setup.weight = 1.0;
  return c;
}

// Expected rate of free-stream molecules hitting a convex body: sum over its
// faces of the one-sided flux towards the face.
double convex_hit_rate(const HostMesh& mesh, const GasSection& gas) {
  double rate = 0.0;
  for (std::size_t i = 0; i < mesh.size(); ++i) {
    for (const auto& sp : gas.species) {
      const double c = std::sqrt(2 * constants::k_B * gas.translational_temperature_K / sp.mass_kg);
      const double s = dot(gas.bulk_velocity_body_mps, -mesh.normals[i]) / c;
      rate += number_flux(gas.number_density_m3 * sp.mole_fraction, c, s) * mesh.areas[i];
    }
  }
  return rate;
}

void set_particle(ParticleStorage& p, std::size_t i, const Vec3& pos, const Vec3& vel) {
  // Helper for analytic tests: writes one particle through host mirrors.
  auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
  auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
  auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
  auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
  auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
  auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.status);
  auto hid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
  hx(i) = pos.x;
  hy(i) = pos.y;
  hz(i) = pos.z;
  hvx(i) = vel.x;
  hvy(i) = vel.y;
  hvz(i) = vel.z;
  hs(i) = static_cast<std::uint8_t>(ParticleStatus::Active);
  hid(i) = i;
  Kokkos::deep_copy(p.x, hx);
  Kokkos::deep_copy(p.y, hy);
  Kokkos::deep_copy(p.z, hz);
  Kokkos::deep_copy(p.vx, hvx);
  Kokkos::deep_copy(p.vy, hvy);
  Kokkos::deep_copy(p.vz, hvz);
  Kokkos::deep_copy(p.status, hs);
  Kokkos::deep_copy(p.particle_id, hid);
}

struct FinalState {
  std::uint8_t status;
  std::uint32_t interactions;
  Vec3 pos, vel;
};

FinalState read_particle(const ParticleStorage& p, std::size_t i) {
  auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.status);
  auto hk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.interactions);
  auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
  auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
  auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
  auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
  auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
  return {hs(i), hk(i), {hx(i), hy(i), hz(i)}, {hvx(i), hvy(i), hvz(i)}};
}

}  // namespace

TEST_CASE("Gate M1: cube - complete particle balance and analytic hit rate", "[transport][gate][device]") {
  const Vec3 u(-7000.0, 1500.0, -900.0);  // generic direction: every cube face sees flux
  Case c = make_case(primitives::box({-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}), u, 0.3);
  constexpr std::size_t n = 400000;
  constexpr std::size_t batch = 100000;
  ParticleStorage p(batch);
  TransportCounters total;
  for (std::size_t first = 0; first < n; first += batch) {
    const TransportCounters b = run_histories(c.setup, SpecularInteraction{}, p, first, batch);
    REQUIRE(b.balanced());
    total += b;
  }
  CHECK(total.injected == n);
  CHECK(total.active == 0);
  CHECK(total.exited == n);  // convex body, specular: every history leaves
  CHECK(total.truncated == 0);
  CHECK(total.lost_inside_body == 0);
  CHECK(total.invalid == 0);
  CHECK(total.max_interactions == 1);  // a convex body is hit at most once
  CHECK(total.surface_interactions == total.histories_with_hits);

  // Fraction of injected histories that hit the cube = hit rate / inflow rate.
  const double p_hit = convex_hit_rate(c.geometry.host, make_gas(u)) / c.inflow.total_rate_per_s;
  const double observed = static_cast<double>(total.histories_with_hits) / n;
  INFO("expected " << p_hit << ", observed " << observed);
  CHECK(std::fabs(observed - p_hit) < 5.0 * std::sqrt(p_hit * (1 - p_hit) / n));
  CHECK(p_hit > 0.05);
}

TEST_CASE("Histories are reproducible and independent of batching", "[transport][device]") {
  Case c = make_case(read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl"),
                     {-7800, 0, 0}, 0.2);
  ParticleStorage a(20000), b(5000);
  (void)run_histories(c.setup, RandomCosineInteraction{}, a, 0, 20000);
  const FinalState ref = read_particle(a, 12345);
  // The same history in a different batch / slot gives bitwise the same result.
  (void)run_histories(c.setup, RandomCosineInteraction{}, b, 10000, 5000);
  const FinalState again = read_particle(b, 2345);
  CHECK(ref.status == again.status);
  CHECK(ref.interactions == again.interactions);
  CHECK(ref.pos == again.pos);
  CHECK(ref.vel == again.vel);
  auto ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b.particle_id);
  std::set<std::uint64_t> unique(ids.data(), ids.data() + 5000);
  CHECK(unique.size() == 5000);
  CHECK(*unique.begin() == 10000);
}

TEST_CASE("Concave gaps and diffuse-like re-emission: no particle is lost", "[transport][device]") {
  // Two boxes separated by a thin gap plus the satellite: many re-emissions
  // near edges and corners, multiple reflections in all directions.
  TriangleSoup soup = primitives::box({-1, -1, -1}, {0, 1, 1});
  primitives::add_box(soup, {0.01, -1, -1}, {1, 1, 1});
  primitives::add_box(soup, {-0.3, -0.3, 1.0005}, {0.3, 0.3, 1.5});
  Case c = make_case(soup, {-3000, 200, 100}, 0.5);
  c.setup.max_interactions = 100000;
  ParticleStorage p(200000);
  const TransportCounters t = run_histories(c.setup, RandomCosineInteraction{}, p, 0, 200000);
  INFO("interactions " << t.surface_interactions << ", max per history " << t.max_interactions);
  CHECK(t.balanced());
  CHECK(t.lost_inside_body == 0);
  CHECK(t.invalid == 0);
  CHECK(t.truncated == 0);
  CHECK(t.exited == 200000);
  CHECK(t.max_interactions > 10);  // particles entering the gap bounce many times
}

TEST_CASE("Analytic multiple reflections between parallel plates", "[transport][device]") {
  // Plates z = 0 and z = 1 (open, two-sided), x in [0, 10], y in [-1, 1].
  // A particle starting at (0, 0, 0.5) with v = (1, 0, 1) hits the upper
  // plate at x = 0.5, the lower one at x = 1.5, ..., the 10th time at x = 9.5;
  // it then passes the plate ends and leaves the domain at x = 10.5.
  TriangleSoup soup;
  primitives::add_rectangle(soup, {0, -1, 0}, {10, 0, 0}, {0, 2, 0});
  primitives::add_rectangle(soup, {0, -1, 1}, {0, 2, 0}, {10, 0, 0});
  Case c = make_case(soup, {-1000, 0, 0}, 0.5, /*require_closed=*/false);
  ParticleStorage p(4);
  p.clear();
  set_particle(p, 0, {0.0, 0.0, 0.5}, {1.0, 0.0, 1.0});
  set_particle(p, 1, {0.0, 0.0, 0.5}, {1.0, 0.0, 1.0});
  set_particle(p, 2, {0.0, 0.0, 0.5}, {1.0, 0.0, 1.0});
  set_particle(p, 3, {5.0, 0.0, 0.5}, {0.0, 0.0, 1.0});  // pure z: bounces forever

  c.setup.max_interactions = 100;
  TransportCounters t = transport_particles(c.setup, SpecularInteraction{}, p, 1);
  FinalState s = read_particle(p, 0);
  CHECK(s.status == static_cast<std::uint8_t>(ParticleStatus::Exited));
  CHECK(s.interactions == 10);  // hits at x = 0.5, 1.5, ..., 9.5
  CHECK_THAT(s.pos.x, WithinRel(10.5, 1e-12));  // domain boundary x = 10 + margin
  CHECK(s.vel == Vec3(1.0, 0.0, 1.0));          // even number of reflections
  CHECK(t.balanced());

  // Truncation: the 4th hit is not processed.
  c.setup.max_interactions = 3;
  (void)transport_particles(c.setup, SpecularInteraction{}, p, 2);
  s = read_particle(p, 1);
  CHECK(s.status == static_cast<std::uint8_t>(ParticleStatus::Truncated));
  CHECK(s.interactions == 3);
  // each re-emission starts surface_offset (1e-8 m here) off the plate: <= 3 offsets of drift
  CHECK_THAT(s.pos.x, WithinAbs(3.5, 3.0 * c.geometry.device.surface_offset_m * (1 + 1e-6)));
  CHECK(s.pos.x < 3.5);
  CHECK_THAT(s.pos.z, WithinAbs(0.0, 1e-12));

  // Endless bouncing is truncated at the limit and counted, not lost.
  c.setup.max_interactions = 1000;
  t = transport_particles(c.setup, SpecularInteraction{}, p, 4);
  s = read_particle(p, 3);
  CHECK(s.status == static_cast<std::uint8_t>(ParticleStatus::Truncated));
  CHECK(s.interactions == 1000);
  CHECK(read_particle(p, 2).interactions == 10);
  CHECK(t.exited == 2);     // slots 0 and 2
  CHECK(t.truncated == 2);  // slots 1 (earlier launch) and 3
  CHECK(t.balanced());
}

TEST_CASE("Failure states: inside a closed body and invalid re-emission", "[transport][device]") {
  Case c = make_case(primitives::box({0, 0, 0}, {1, 1, 1}), {-1000, 0, 0}, 0.5);
  ParticleStorage p(3);
  p.clear();
  set_particle(p, 0, {0.5, 0.5, 0.5}, {1, 0.1, 0.2});   // starts inside the cube
  set_particle(p, 1, {-0.25, 0.5, 0.5}, {1, 0, 0});     // hits the cube; model re-emits into the wall
  set_particle(p, 2, {-0.25, 0.5, 0.5}, {0, 0, 0});     // zero velocity
  TransportCounters t = transport_particles(c.setup, IntoTheWallInteraction{}, p, 3);
  CHECK(read_particle(p, 0).status == static_cast<std::uint8_t>(ParticleStatus::LostInsideBody));
  CHECK(read_particle(p, 1).status == static_cast<std::uint8_t>(ParticleStatus::Invalid));
  CHECK(read_particle(p, 2).status == static_cast<std::uint8_t>(ParticleStatus::Invalid));
  CHECK(t.lost_inside_body == 1);
  CHECK(t.invalid == 2);
  CHECK(t.balanced());
}
