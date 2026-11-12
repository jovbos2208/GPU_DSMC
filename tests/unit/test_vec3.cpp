// M0.6: GPU-compatible Vec3 / Mat3 types.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iomanip>
#include <limits>

#include "DeviceTest.hpp"
#include "vleo/core/Mat3.hpp"
#include "vleo/core/Vec3.hpp"

using namespace vleo;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Deterministic pseudo-random test vector for index i (host and device).
// Components are small integers / 8, i.e. exactly representable, so host and
// device start from bitwise identical inputs (no transcendental functions or
// FMA-contractible expressions in the input generation).
KOKKOS_INLINE_FUNCTION Vec3 test_vector(int i) {
  const int a = (i * 37 + 11) % 97 - 48;
  const int b = (i * 53 + 29) % 89 - 44;
  const int c = (i * 71 + 5) % 83 - 41;
  return {a / 8.0, b / 8.0 + 0.125, c / 8.0 - 0.0625};
}

// Evaluates a fixed sequence of Vec3 operations; identical code runs on host and device.
struct Vec3Ops {
  KOKKOS_INLINE_FUNCTION Vec3 operator()(int i) const {
    const Vec3 a = test_vector(i);
    const Vec3 b = test_vector(i + 1000);
    Vec3 r = cross(a, b) + 0.5 * a - b / 4.0;
    r += Vec3(dot(a, b), norm(a), norm2(b));
    r -= reflect(a, normalized(b));
    r *= 1.5;
    return r;
  }
};

}  // namespace

TEST_CASE("Vec3 basic arithmetic", "[vec3]") {
  constexpr Vec3 a{1.0, 2.0, 3.0};
  constexpr Vec3 b{-4.0, 5.0, 0.5};
  static_assert((a + b) == Vec3(-3.0, 7.0, 3.5));
  static_assert((a - b) == Vec3(5.0, -3.0, 2.5));
  static_assert((2.0 * a) == Vec3(2.0, 4.0, 6.0));
  static_assert((a * 2.0) == Vec3(2.0, 4.0, 6.0));
  static_assert((a / 2.0) == Vec3(0.5, 1.0, 1.5));
  static_assert(-a == Vec3(-1.0, -2.0, -3.0));
  static_assert(dot(a, b) == -4.0 + 10.0 + 1.5);
  static_assert(hadamard(a, b) == Vec3(-4.0, 10.0, 1.5));
  static_assert(a[0] == 1.0 && a[1] == 2.0 && a[2] == 3.0);

  Vec3 c = a;
  c[1] = 7.0;
  CHECK(c == Vec3(1.0, 7.0, 3.0));
  CHECK(c != a);
  CHECK_THAT(norm(Vec3(3.0, 4.0, 12.0)), WithinAbs(13.0, 0.0));
  CHECK_THAT(distance(Vec3(1, 1, 1), Vec3(4, 5, 1)), WithinAbs(5.0, 0.0));
  CHECK(Vec3() == Vec3::zero());
}

TEST_CASE("Vec3 cross product is right-handed", "[vec3]") {
  static_assert(cross(Vec3::unit_x(), Vec3::unit_y()) == Vec3::unit_z());
  static_assert(cross(Vec3::unit_y(), Vec3::unit_z()) == Vec3::unit_x());
  static_assert(cross(Vec3::unit_z(), Vec3::unit_x()) == Vec3::unit_y());
  for (int i = 0; i < 50; ++i) {
    const Vec3 a = test_vector(i), b = test_vector(i + 7);
    const Vec3 c = cross(a, b);
    CHECK_THAT(dot(c, a), WithinAbs(0.0, 1e-12));
    CHECK_THAT(dot(c, b), WithinAbs(0.0, 1e-12));
    CHECK(approx_equal(cross(b, a), -c, 0.0));
    // Lagrange identity |a x b|^2 = |a|^2 |b|^2 - (a.b)^2
    CHECK_THAT(norm2(c), WithinRel(norm2(a) * norm2(b) - dot(a, b) * dot(a, b), 1e-12));
  }
}

TEST_CASE("Vec3 normalisation, projection and reflection", "[vec3]") {
  const Vec3 v{0.0, 3.0, -4.0};
  CHECK(approx_equal(normalized(v), Vec3(0.0, 0.6, -0.8), 1e-15));
  CHECK(normalized_or(Vec3::zero(), Vec3::unit_z()) == Vec3::unit_z());
  CHECK(!is_finite(normalized(Vec3::zero())));
  CHECK(is_finite(v));
  CHECK(!is_finite(Vec3(1.0, std::numeric_limits<double>::quiet_NaN(), 0.0)));
  CHECK(!is_finite(Vec3(std::numeric_limits<double>::infinity(), 0.0, 0.0)));

  const Vec3 n = Vec3::unit_z();
  const Vec3 a{1.0, -2.0, -3.0};
  CHECK(project_onto_unit(a, n) == Vec3(0.0, 0.0, -3.0));
  CHECK(reject_from_unit(a, n) == Vec3(1.0, -2.0, 0.0));
  // Specular reflection flips the normal component and keeps the tangential one.
  CHECK(reflect(a, n) == Vec3(1.0, -2.0, 3.0));
  for (int i = 0; i < 50; ++i) {
    const Vec3 u = test_vector(i);
    const Vec3 nn = normalized(test_vector(i + 3));
    const Vec3 r = reflect(u, nn);
    CHECK_THAT(norm(r), WithinRel(norm(u), 1e-14));
    CHECK_THAT(dot(r, nn), WithinAbs(-dot(u, nn), 1e-12));
    CHECK(approx_equal(reflect(r, nn), u, 1e-12));
  }
  CHECK_THAT(max_abs_component(Vec3(-7.0, 2.0, 5.0)), WithinAbs(7.0, 0.0));
}

TEST_CASE("Mat3 operations", "[vec3][mat3]") {
  constexpr Mat3 I = Mat3::identity();
  constexpr Vec3 v{1.0, -2.0, 0.5};
  static_assert(I * v == v);
  static_assert(I.determinant() == 1.0);

  const Mat3 a = Mat3::from_rows({1, 2, 3}, {0, 1, 4}, {5, 6, 0});
  CHECK(a.row(1) == Vec3(0, 1, 4));
  CHECK(a.col(2) == Vec3(3, 4, 0));
  CHECK(a.transposed().row(2) == Vec3(3, 4, 0));
  CHECK(Mat3::from_columns({1, 0, 5}, {2, 1, 6}, {3, 4, 0}).row(0) == Vec3(1, 2, 3));
  CHECK_THAT(a.determinant(), WithinAbs(1.0, 1e-14));
  CHECK(max_abs_difference(a * I, a) == 0.0);
  CHECK(a * v == Vec3(1 - 4 + 1.5, -2 + 2, 5 - 12));
}

TEST_CASE("Rotation matrices are proper orthogonal and right-handed", "[vec3][mat3]") {
  const double half_pi = 0.5 * 3.14159265358979323846;
  // +90 deg about z maps x -> y (right-hand rule).
  CHECK(approx_equal(rotation_about_axis(Vec3::unit_z(), half_pi) * Vec3::unit_x(), Vec3::unit_y(), 1e-15));
  CHECK(approx_equal(rotation_about_axis(Vec3::unit_x(), half_pi) * Vec3::unit_y(), Vec3::unit_z(), 1e-15));
  CHECK(approx_equal(rotation_about_axis(Vec3::unit_y(), half_pi) * Vec3::unit_z(), Vec3::unit_x(), 1e-15));

  for (int i = 0; i < 50; ++i) {
    const Vec3 axis = normalized(test_vector(i));
    const double angle = 0.3 * i - 4.0;
    const Mat3 r = rotation_about_axis(axis, angle);
    CHECK(max_abs_difference(r * r.transposed(), Mat3::identity()) < 1e-14);
    CHECK_THAT(r.determinant(), WithinAbs(1.0, 1e-14));
    CHECK(approx_equal(r * axis, axis, 1e-14));  // axis is invariant
    const Vec3 u = test_vector(i + 11);
    CHECK(approx_equal(r.transposed() * (r * u), u, 1e-13));
    CHECK_THAT(norm(r * u), WithinRel(norm(u), 1e-14));
  }
}

TEST_CASE("Vec3 operations give the same results on the device", "[vec3][device]") {
  constexpr int n = 256;
  const auto dev = test::eval_on_device<Vec3>("vec3_ops", n, Vec3Ops{});
  const Vec3Ops ops;
  for (int i = 0; i < n; ++i) {
    // Device math libraries may differ from the host in the last ulps
    // (sqrt/sin/cos, FMA contraction): compare with a tight relative tolerance.
    const Vec3 h = ops(i);
    INFO("i = " << i << std::setprecision(17) << " device (" << dev(i).x << ", " << dev(i).y << ", " << dev(i).z
                << ") host (" << h.x << ", " << h.y << ", " << h.z << ")");
    CHECK(approx_equal(dev(i), h, 1e-12, 1e-12));
  }
}
