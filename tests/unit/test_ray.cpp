// M1.5-M1.9: ray-triangle intersection, edge/vertex robustness, BVH vs brute force.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "DeviceTest.hpp"
#include "vleo/core/Random.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/geometry/Primitives.hpp"

using namespace vleo;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kInf = Kokkos::infinity_v<double>;

struct SingleRay {
  Vec3 o, d;
  Triangle tri;
  double t_max;
};

struct HitResult {
  int hit;
  double t, b0, b1, b2;
};

struct EvalRays {
  Kokkos::View<const SingleRay*> rays;
  KOKKOS_INLINE_FUNCTION HitResult operator()(int i) const {
    const SingleRay& r = rays(i);
    TriangleHit h{0, 0, 0, 0};
    const bool ok = intersect(WatertightRay(r.o, r.d), r.tri, r.t_max, h);
    return {ok ? 1 : 0, h.t, h.b0, h.b1, h.b2};
  }
};

std::vector<HitResult> on_device(const std::vector<SingleRay>& rays) {
  Kokkos::View<SingleRay*> v("rays", rays.size());
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < rays.size(); ++i) h(i) = rays[i];
  Kokkos::deep_copy(v, h);
  const auto r = test::eval_on_device<HitResult>("eval_rays", static_cast<int>(rays.size()), EvalRays{v});
  return std::vector<HitResult>(r.data(), r.data() + r.extent(0));
}

// Rays from a geometry query, evaluated with BVH and brute force.
struct QueryRay {
  Vec3 o, d;
  double t_max;
  int ignore;
};

struct BvhVsBrute {
  DeviceGeometry g;
  Kokkos::View<const QueryRay*> rays;
  KOKKOS_INLINE_FUNCTION Kokkos::Array<RayHit, 2> operator()(int i) const {
    const QueryRay& q = rays(i);
    return {g.closest_hit(q.o, q.d, q.t_max, q.ignore), closest_hit_brute_force(g.mesh, q.o, q.d, q.t_max, q.ignore)};
  }
};

Kokkos::View<RayHit*[2], Kokkos::HostSpace> compare_bvh(const Geometry& g, const std::vector<QueryRay>& q) {
  Kokkos::View<QueryRay*> v("queries", q.size());
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < q.size(); ++i) h(i) = q[i];
  Kokkos::deep_copy(v, h);
  const auto r = test::eval_on_device<Kokkos::Array<RayHit, 2>>("bvh_vs_brute", static_cast<int>(q.size()),
                                                                 BvhVsBrute{g.device, v});
  Kokkos::View<RayHit*[2], Kokkos::HostSpace> out("out", q.size());
  for (std::size_t i = 0; i < q.size(); ++i) {
    out(i, 0) = r(i)[0];
    out(i, 1) = r(i)[1];
  }
  return out;
}

Vec3 random_unit(RandomStream& r) {
  for (;;) {
    const Vec3 v(2 * r.uniform() - 1, 2 * r.uniform() - 1, 2 * r.uniform() - 1);
    const double n = norm(v);
    if (n > 0.1 && n <= 1.0) return v / n;
  }
}

const Triangle kTri{{0, 0, 0}, {2, 0, 0}, {0, 2, 0}};

}  // namespace

TEST_CASE("Ray-triangle: analytic hits and misses", "[ray]") {
  const std::vector<SingleRay> rays = {
      {{0.5, 0.25, 3.0}, {0, 0, -1}, kTri, kInf},            // 0 interior hit, t = 3
      {{0.5, 0.25, -2.0}, {0, 0, 2}, kTri, kInf},            // 1 from below (back face), t = 1
      {{0.5, 0.25, 3.0}, {0, 0, 1}, kTri, kInf},             // 2 pointing away
      {{1.5, 1.5, 1.0}, {0, 0, -1}, kTri, kInf},             // 3 outside (x + y > 2)
      {{0.5, 0.5, 1.0}, {1, 0, 0}, kTri, kInf},              // 4 parallel to the plane
      {{0.5, 0.25, 3.0}, {0, 0, -1}, kTri, 2.999},           // 5 beyond t_max
      {{0.5, 0.25, 3.0}, {0, 0, -1}, kTri, 3.0},             // 6 exactly at t_max
      {{1.0, 0.0, 1.0}, {0, 0, -1}, kTri, kInf},             // 7 on edge a-b
      {{1.0, 1.0, 1.0}, {0, 0, -1}, kTri, kInf},             // 8 on edge b-c (hypotenuse)
      {{0.0, 0.0, 1.0}, {0, 0, -1}, kTri, kInf},             // 9 on vertex a
      {{2.0, 0.0, 1.0}, {0, 0, -1}, kTri, kInf},             // 10 on vertex b
      {{0.0, 0.0, 0.0}, {0, 0, -1}, kTri, kInf},             // 11 origin on the triangle: t = 0 rejected
      {{-1.0, 0.5, 1.0}, {1, 0, -1}, kTri, kInf},            // 12 oblique, hits (0, 0.5, 0) at t = 1
      {{0.25, 0.25, 1.0}, {1e-3, -2e-3, -1}, kTri, kInf},    // 13 nearly normal oblique
  };
  for (int pass = 0; pass < 2; ++pass) {
    INFO((pass == 0 ? "host" : "device"));
    std::vector<HitResult> r;
    if (pass == 0) {
      for (const auto& ray : rays) {
        TriangleHit h{0, 0, 0, 0};
        const bool ok = intersect(WatertightRay(ray.o, ray.d), ray.tri, ray.t_max, h);
        r.push_back({ok ? 1 : 0, h.t, h.b0, h.b1, h.b2});
      }
    } else {
      r = on_device(rays);
    }
    CHECK(r[0].hit == 1);
    CHECK_THAT(r[0].t, WithinRel(3.0, 1e-15));
    CHECK_THAT(r[0].b1, WithinAbs(0.25, 1e-15));  // weight of vertex b = x / 2
    CHECK_THAT(r[0].b2, WithinAbs(0.125, 1e-15));
    CHECK_THAT(r[0].b0 + r[0].b1 + r[0].b2, WithinAbs(1.0, 1e-15));
    CHECK(r[1].hit == 1);
    CHECK_THAT(r[1].t, WithinRel(1.0, 1e-15));
    CHECK(r[2].hit == 0);
    CHECK(r[3].hit == 0);
    CHECK(r[4].hit == 0);
    CHECK(r[5].hit == 0);
    CHECK(r[6].hit == 1);
    CHECK(r[7].hit == 1);
    CHECK(r[8].hit == 1);
    CHECK(r[9].hit == 1);
    CHECK(r[10].hit == 1);
    CHECK_THAT(r[10].b1, WithinAbs(1.0, 1e-15));
    CHECK(r[11].hit == 0);
    CHECK(r[12].hit == 1);
    CHECK_THAT(r[12].t, WithinRel(1.0, 1e-15));
    CHECK(r[13].hit == 1);
  }
}

TEST_CASE("Closed meshes are watertight at edges and vertices", "[ray][device]") {
  // Rays aimed exactly at mesh vertices and at points on edges, coming from
  // outside and *entering* the solid, must always hit the closed mesh: a miss
  // would be a leak between adjacent triangles. (Rays that only graze a
  // silhouette edge may legitimately miss by an ulp, because the direction
  // target - origin is rounded; they are excluded by construction: the
  // origin lies in the outward radial direction of the convex component,
  // perturbed by at most 0.2, so the ray is transversal to every adjacent face.)
  const Vec3 centers[2] = {{0.1, -0.2, 0.3}, {3.5, 0.0, 0.0}};
  TriangleSoup soup = primitives::uv_sphere(centers[0], 1.0, 24, 48);
  primitives::add_box(soup, {3, -1, -1}, {4, 1, 1}, 1);
  const Geometry g = make_geometry(soup);
  RandomStream rng(42, 0, 0, RngStream::Test);
  std::vector<QueryRay> q;
  for (std::size_t i = 0; i < g.host.size(); ++i) {
    const Triangle& t = g.host.triangles[i];
    const Vec3 center = centers[g.host.component[i]];
    for (const Vec3& target : {t.a, t.b, t.c, 0.5 * (t.a + t.b), t.b + 0.25 * (t.c - t.b)}) {
      for (int k = 0; k < 2; ++k) {
        const Vec3 m = normalized(normalized(target - center) + 0.2 * random_unit(rng));
        const Vec3 origin = target + 10.0 * m;
        q.push_back({origin, target - origin, kInf, -1});
      }
    }
  }
  const auto r = compare_bvh(g, q);
  std::size_t misses = 0;
  for (std::size_t i = 0; i < q.size(); ++i) {
    // Rays may first hit the other component; any hit counts, a miss is a leak.
    misses += r(i, 0).valid() ? 0 : 1;
    REQUIRE(r(i, 0).triangle == r(i, 1).triangle);
  }
  CHECK(misses == 0);
  CHECK(q.size() > 10000);
}

TEST_CASE("Rays through a cube edge or corner hit exactly one closest triangle", "[ray]") {
  const Geometry g = make_geometry(primitives::box({0, 0, 0}, {1, 1, 1}));
  std::vector<QueryRay> q = {
      {{0.5, -1.0, -1.0}, {0, 1, 1}, kInf, -1},     // diagonal onto the edge y = z = 0
      {{-1.0, -1.0, -1.0}, {1, 1, 1}, kInf, -1},    // onto the corner (0,0,0)
      {{0.5, 0.5, 5.0}, {0, 0, -1}, kInf, -1},      // onto the face diagonal of the top face
      {{-1.0, 0.0, 0.5}, {1, 0, 0}, kInf, -1},      // grazing along the face y = 0, hits edge x = 0
  };
  const auto r = compare_bvh(g, q);
  CHECK(r(0, 0).valid());
  CHECK_THAT(r(0, 0).t, WithinRel(1.0, 1e-15));
  CHECK(r(1, 0).valid());
  CHECK_THAT(r(1, 0).t, WithinRel(1.0, 1e-15));
  CHECK(r(2, 0).valid());
  CHECK_THAT(r(2, 0).t, WithinRel(4.0, 1e-15));
  CHECK(r(3, 0).valid());
  CHECK_THAT(r(3, 0).t, WithinRel(1.0, 1e-15));
  for (std::size_t i = 0; i < q.size(); ++i) CHECK(r(i, 0).triangle == r(i, 1).triangle);
}

TEST_CASE("BVH and brute force return identical closest hits on random meshes", "[ray][bvh][device]") {
  const int variant = GENERATE(0, 1, 2, 3);
  RandomStream rng(7, static_cast<std::uint64_t>(variant), 0, RngStream::Test);
  TriangleSoup soup;
  std::string name;
  if (variant == 0) {
    name = "random triangle soup";
    for (int i = 0; i < 2000; ++i) {
      const Vec3 c(4 * rng.uniform() - 2, 4 * rng.uniform() - 2, 4 * rng.uniform() - 2);
      const double s = 0.02 + 0.3 * rng.uniform();
      soup.triangles.push_back({c + s * random_unit(rng), c + s * random_unit(rng), c + s * random_unit(rng)});
    }
  } else if (variant == 1) {
    name = "fine sphere";
    soup = primitives::uv_sphere({0, 0, 0}, 1.5, 32, 64);
  } else if (variant == 2) {
    name = "many small boxes (identical centroids along lines)";
    for (int i = 0; i < 400; ++i) {
      const Vec3 lo(0.1 * (i % 20), 0.1 * (i / 20), 0.0);
      primitives::add_box(soup, lo, lo + Vec3(0.05, 0.05, 1.0 + 0.001 * i));
    }
  } else {
    name = "satellite";
    soup = read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl");
  }
  INFO(name);
  MeshOptions mopt;
  mopt.require_closed = variant != 0;
  const Geometry g = make_geometry(soup, mopt);
  CHECK(g.bvh.stats.max_depth < kMaxBvhDepth);

  const Aabb& b = g.host.bbox;
  const Vec3 c = b.center();
  const double R = norm(b.extent());
  std::vector<QueryRay> q;
  for (int i = 0; i < 8000; ++i) {
    const Vec3 origin = (i % 3 == 0) ? c + (rng.uniform() * 0.5 * R) * random_unit(rng)  // inside the box
                                     : c + 2.0 * R * random_unit(rng);                  // outside
    const Vec3 target = c + (rng.uniform() * 0.6 * R) * random_unit(rng);
    const double t_max = (i % 5 == 0) ? 0.5 + rng.uniform() : kInf;  // finite segments too
    const int ignore = (i % 7 == 0) ? static_cast<int>(rng.next_u32() % g.host.size()) : -1;
    q.push_back({origin, (target - origin) * (0.5 + rng.uniform()), t_max, ignore});
  }
  // Axis-parallel rays exercise the slab test with infinite inverse directions.
  for (int i = 0; i < 1500; ++i) {
    Vec3 d;
    d[i % 3] = (i % 2 == 0) ? 1.0 : -1.0;
    Vec3 o = c + (0.7 * R) * random_unit(rng);
    o[i % 3] = c[i % 3] - d[i % 3] * 2.0 * R;
    q.push_back({o, d, kInf, -1});
  }
  const auto r = compare_bvh(g, q);
  std::size_t hits = 0;
  for (std::size_t i = 0; i < q.size(); ++i) {
    INFO("ray " << i);
    REQUIRE(r(i, 0).triangle == r(i, 1).triangle);
    if (r(i, 0).valid()) {
      ++hits;
      REQUIRE(r(i, 0).t == r(i, 1).t);  // same primitive test: bitwise identical
      REQUIRE(r(i, 0).triangle != q[i].ignore);
      REQUIRE(r(i, 0).t <= q[i].t_max);
    }
  }
  CHECK(hits > q.size() / 10);  // the test is not vacuous
}
