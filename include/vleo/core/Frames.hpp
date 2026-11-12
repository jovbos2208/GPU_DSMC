#pragma once

// Coordinate frames used throughout the solver. Conventions are documented in
// docs/conventions.md; the essentials are:
//
//  * Body frame: the frame of the input geometry (STL), right-handed, SI metres.
//  * Local surface frame (t1, t2, n): right-handed orthonormal basis with n the
//    unit surface normal pointing from the solid INTO the gas, t1 x t2 = n.
//  * Aerodynamic frame (e_D, e_Y, e_L): e_D points along the free-stream
//    velocity (= drag direction), e_L is the user lift axis made orthogonal to
//    e_D, e_Y = e_L x e_D completes a right-handed triad (e_D x e_Y = e_L).

#include "vleo/core/Mat3.hpp"
#include "vleo/core/Vec3.hpp"

namespace vleo {

// Right-handed orthonormal basis (t1, t2, n).
struct OrthonormalBasis {
  Vec3 t1;
  Vec3 t2;
  Vec3 n;

  // Components of a global (body-frame) vector in the local frame:
  // result = (v.t1, v.t2, v.n).
  KOKKOS_INLINE_FUNCTION constexpr Vec3 to_local(const Vec3& v) const { return {dot(v, t1), dot(v, t2), dot(v, n)}; }

  // Global vector from local components (inverse of to_local).
  KOKKOS_INLINE_FUNCTION constexpr Vec3 to_global(const Vec3& l) const { return l.x * t1 + l.y * t2 + l.z * n; }

  // Rotation matrix global -> local (rows t1, t2, n).
  KOKKOS_INLINE_FUNCTION constexpr Mat3 global_to_local() const { return Mat3::from_rows(t1, t2, n); }
};

// Builds a right-handed orthonormal basis (t1, t2, n) from a unit normal n.
// Branch-free construction of Duff et al., "Building an Orthonormal Basis,
// Revisited", JCGT 6(1), 2017. Continuous except across n.z = 0 sign change,
// accurate to machine precision for every unit n (including n = -e_z).
KOKKOS_INLINE_FUNCTION OrthonormalBasis make_basis_from_normal(const Vec3& n) {
  const double sign = Kokkos::copysign(1.0, n.z);
  const double a = -1.0 / (sign + n.z);
  const double b = n.x * n.y * a;
  OrthonormalBasis basis;
  basis.t1 = Vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
  basis.t2 = Vec3(b, sign + n.y * n.y * a, -n.y);
  basis.n = n;
  return basis;
}

// Aerodynamic axes in body-frame components.
struct AeroAxes {
  Vec3 drag;  // e_D: along the free-stream velocity
  Vec3 side;  // e_Y = e_L x e_D
  Vec3 lift;  // e_L: lift axis, orthogonal to e_D

  // Rotation matrix body -> aerodynamic components (rows e_D, e_Y, e_L).
  KOKKOS_INLINE_FUNCTION constexpr Mat3 body_to_aero() const { return Mat3::from_rows(drag, side, lift); }
};

enum class AeroAxesStatus : int {
  Ok = 0,
  ZeroVelocity = 1,       // |V_inf| == 0 or non-finite
  ZeroLiftAxis = 2,       // |lift hint| == 0 or non-finite
  LiftParallelToFlow = 3  // lift hint has no component orthogonal to the flow
};

// Constructs the aerodynamic axes from the free-stream velocity and a lift
// hint. The hint is projected onto the plane orthogonal to the flow; it is
// rejected if the orthogonal part is smaller than `parallel_tol * |hint|`.
KOKKOS_INLINE_FUNCTION AeroAxesStatus make_aero_axes(const Vec3& freestream_velocity, const Vec3& lift_hint,
                                                     AeroAxes& out, double parallel_tol = 1.0e-6) {
  const double v = norm(freestream_velocity);
  if (!(v > 0.0) || !Kokkos::isfinite(v)) return AeroAxesStatus::ZeroVelocity;
  const double h = norm(lift_hint);
  if (!(h > 0.0) || !Kokkos::isfinite(h)) return AeroAxesStatus::ZeroLiftAxis;

  const Vec3 e_d = freestream_velocity / v;
  const Vec3 l_perp = reject_from_unit(lift_hint, e_d);
  const double lp = norm(l_perp);
  if (!(lp > parallel_tol * h)) return AeroAxesStatus::LiftParallelToFlow;

  out.drag = e_d;
  out.lift = l_perp / lp;
  out.side = cross(out.lift, out.drag);
  return AeroAxesStatus::Ok;
}

}  // namespace vleo
