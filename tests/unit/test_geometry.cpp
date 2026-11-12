// M0: basic geometric operations on triangles and bounding boxes.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "DeviceTest.hpp"
#include "vleo/core/Mat3.hpp"
#include "vleo/geometry/Triangle.hpp"

using namespace vleo;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct TriangleProps {
  double area;
  Vec3 normal;
  Vec3 centroid;
};

struct RotatedTriangle {
  KOKKOS_INLINE_FUNCTION TriangleProps operator()(int i) const {
    const Mat3 r = rotation_about_axis(normalized(Vec3(1.0, 2.0, 3.0)), 0.1 * i);
    const Triangle t{r * Vec3(0, 0, 0), r * Vec3(2, 0, 0), r * Vec3(0, 3, 0)};
    return {area(t), unit_normal(t), centroid(t)};
  }
};

}  // namespace

TEST_CASE("Triangle area, normal and centroid", "[geometry]") {
  const Triangle t{{0, 0, 0}, {2, 0, 0}, {0, 3, 0}};
  CHECK_THAT(area(t), WithinAbs(3.0, 1e-15));
  CHECK(unit_normal(t) == Vec3(0, 0, 1));  // counter-clockwise seen from +z
  CHECK(approx_equal(centroid(t), Vec3(2.0 / 3.0, 1.0, 0.0), 1e-15));
  CHECK(!is_degenerate(t));

  // Reversing the vertex order flips the normal.
  const Triangle flipped{t.a, t.c, t.b};
  CHECK(unit_normal(flipped) == Vec3(0, 0, -1));
  CHECK(point_from_barycentric(t, 0.5, 0.5) == Vec3(1.0, 1.5, 0.0));
}

TEST_CASE("Degenerate triangles are detected", "[geometry]") {
  CHECK(is_degenerate(Triangle{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}));  // collinear
  CHECK(is_degenerate(Triangle{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));  // point
  CHECK(!is_degenerate(Triangle{{0, 0, 0}, {1e-6, 0, 0}, {0, 1e-6, 0}}));  // tiny but valid
}

TEST_CASE("Axis-aligned bounding boxes", "[geometry]") {
  Aabb box;
  CHECK(box.empty());
  box.expand(Vec3(1, -2, 3));
  CHECK(!box.empty());
  CHECK(box.extent() == Vec3(0, 0, 0));
  box.expand(Vec3(-1, 4, 3.5));
  CHECK(box.lo == Vec3(-1, -2, 3));
  CHECK(box.hi == Vec3(1, 4, 3.5));
  CHECK(box.center() == Vec3(0, 1, 3.25));
  CHECK(box.contains(Vec3(0, 0, 3.2)));
  CHECK(!box.contains(Vec3(0, 0, 4)));

  Aabb other;
  box.expand(other);  // expanding by an empty box is a no-op
  CHECK(box.lo == Vec3(-1, -2, 3));

  const Aabb tb = bounding_box(Triangle{{0, 0, 0}, {2, 0, -1}, {0, 3, 0}});
  CHECK(tb.lo == Vec3(0, 0, -1));
  CHECK(tb.hi == Vec3(2, 3, 0));
}

TEST_CASE("Triangle properties are rotation invariant on host and device", "[geometry][device]") {
  constexpr int n = 64;
  const auto dev = test::eval_on_device<TriangleProps>("rotated_triangle", n, RotatedTriangle{});
  const RotatedTriangle f;
  for (int i = 0; i < n; ++i) {
    const TriangleProps h = f(i);
    INFO("i = " << i);
    CHECK_THAT(h.area, WithinRel(3.0, 1e-14));
    CHECK_THAT(norm(h.normal), WithinRel(1.0, 1e-15));
    CHECK_THAT(dev(i).area, WithinRel(h.area, 1e-14));
    CHECK(approx_equal(dev(i).normal, h.normal, 1e-14));
    CHECK(approx_equal(dev(i).centroid, h.centroid, 1e-14));
  }
}
