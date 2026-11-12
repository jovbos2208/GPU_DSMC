// M0.6: coordinate transformations (local surface frame, aerodynamic frame).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "DeviceTest.hpp"
#include "vleo/core/Frames.hpp"

using namespace vleo;
using Catch::Matchers::WithinAbs;

namespace {

// Unit normals covering the sphere, including the poles and the equator
// (n.z = 0 is where the Duff et al. construction switches branch).
KOKKOS_INLINE_FUNCTION Vec3 sphere_normal(int i, int n) {
  if (i == 0) return {0.0, 0.0, 1.0};
  if (i == 1) return {0.0, 0.0, -1.0};
  if (i == 2) return {1.0, 0.0, 0.0};
  if (i == 3) return {0.0, -1.0, 0.0};
  if (i == 4) return {0.0, 0.0, -1.0 + 1e-17};  // rounds to -1
  if (i == 5) return normalized(Vec3(1e-9, 1e-9, -1.0));
  // Fibonacci sphere for the rest
  const double golden = 2.399963229728653;  // pi (3 - sqrt 5)
  const double z = 1.0 - 2.0 * (i + 0.5) / n;
  const double r = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - z * z));
  return {r * Kokkos::cos(golden * i), r * Kokkos::sin(golden * i), z};
}

struct BasisError {
  double orthonormality;  // max |B B^T - I|
  double handedness;      // |t1 x t2 - n|_inf
  double roundtrip;       // |to_global(to_local(v)) - v|_inf
};

constexpr int kNormals = 2000;

struct BasisCheck {
  KOKKOS_INLINE_FUNCTION BasisError operator()(int i) const {
    const Vec3 n = sphere_normal(i, kNormals);
    const OrthonormalBasis b = make_basis_from_normal(n);
    const Mat3 m = b.global_to_local();
    BasisError e;
    e.orthonormality = max_abs_difference(m * m.transposed(), Mat3::identity());
    e.handedness = max_abs_component(cross(b.t1, b.t2) - b.n);
    const Vec3 v(0.3 - i * 1e-3, -2.0, 0.7);
    e.roundtrip = max_abs_component(b.to_global(b.to_local(v)) - v);
    return e;
  }
};

}  // namespace

TEST_CASE("Local surface basis is right-handed and orthonormal", "[frames]") {
  const BasisCheck check;
  for (int i = 0; i < kNormals; ++i) {
    const BasisError e = check(i);
    INFO("normal index " << i);
    REQUIRE(e.orthonormality < 1e-14);
    REQUIRE(e.handedness < 1e-14);
    REQUIRE(e.roundtrip < 1e-14);
  }
}

TEST_CASE("Local surface basis: normal component convention", "[frames]") {
  // An incoming particle moving against the outward normal has c.n < 0.
  const Vec3 n = normalized(Vec3(1.0, 1.0, 0.0));
  const OrthonormalBasis b = make_basis_from_normal(n);
  const Vec3 c_in = -3.0 * n + 2.0 * b.t1;
  const Vec3 local = b.to_local(c_in);
  CHECK_THAT(local.z, WithinAbs(-3.0, 1e-14));
  CHECK_THAT(local.x, WithinAbs(2.0, 1e-14));
  CHECK_THAT(local.y, WithinAbs(0.0, 1e-14));
  CHECK(b.global_to_local() * c_in == local);
}

TEST_CASE("Local surface basis on the device", "[frames][device]") {
  const auto dev = test::eval_on_device<BasisError>("basis_check", kNormals, BasisCheck{});
  for (int i = 0; i < kNormals; ++i) {
    INFO("normal index " << i);
    REQUIRE(dev(i).orthonormality < 1e-14);
    REQUIRE(dev(i).handedness < 1e-14);
    REQUIRE(dev(i).roundtrip < 1e-14);
  }
}

TEST_CASE("Aerodynamic axes: drag along the flow, right-handed triad", "[frames]") {
  AeroAxes ax;
  // Typical config: flow towards -x in the body frame, lift hint +z.
  REQUIRE(make_aero_axes({-7800.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, ax) == AeroAxesStatus::Ok);
  CHECK(ax.drag == Vec3(-1.0, 0.0, 0.0));
  CHECK(ax.lift == Vec3(0.0, 0.0, 1.0));
  CHECK(ax.side == cross(ax.lift, ax.drag));
  CHECK(ax.side == Vec3(0.0, -1.0, 0.0));
  CHECK(approx_equal(cross(ax.drag, ax.side), ax.lift, 1e-15));

  // Flow +x, lift +z  =>  side +y.
  REQUIRE(make_aero_axes({1.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, ax) == AeroAxesStatus::Ok);
  CHECK(ax.side == Vec3(0.0, 1.0, 0.0));

  // Non-orthogonal lift hint is projected onto the plane normal to the flow.
  REQUIRE(make_aero_axes({1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, ax) == AeroAxesStatus::Ok);
  CHECK(approx_equal(ax.lift, Vec3(0.0, 0.0, 1.0), 1e-15));

  // Angle of attack: flow rotated in the x-z plane, lift stays orthogonal.
  const Vec3 v = rotation_about_axis(Vec3::unit_y(), -0.2) * Vec3(1.0, 0.0, 0.0);
  REQUIRE(make_aero_axes(v, {0.0, 0.0, 1.0}, ax) == AeroAxesStatus::Ok);
  CHECK_THAT(dot(ax.drag, ax.lift), WithinAbs(0.0, 1e-15));
  CHECK_THAT(dot(ax.drag, ax.side), WithinAbs(0.0, 1e-15));
  CHECK_THAT(dot(ax.side, ax.lift), WithinAbs(0.0, 1e-15));
  CHECK(ax.lift.z > 0.0);
  const Mat3 m = ax.body_to_aero();
  CHECK_THAT(m.determinant(), WithinAbs(1.0, 1e-14));
  CHECK(approx_equal(m * v, Vec3(1.0, 0.0, 0.0), 1e-15));  // flow is pure drag direction
}

TEST_CASE("Aerodynamic axes: invalid inputs are rejected", "[frames]") {
  AeroAxes ax;
  CHECK(make_aero_axes({0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, ax) == AeroAxesStatus::ZeroVelocity);
  CHECK(make_aero_axes({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, ax) == AeroAxesStatus::ZeroLiftAxis);
  CHECK(make_aero_axes({1.0, 0.0, 0.0}, {-3.0, 0.0, 0.0}, ax) == AeroAxesStatus::LiftParallelToFlow);
  CHECK(make_aero_axes({1.0, 0.0, 0.0}, {1.0, 1e-9, 0.0}, ax) == AeroAxesStatus::LiftParallelToFlow);
}
