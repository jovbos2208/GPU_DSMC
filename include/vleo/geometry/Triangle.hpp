#pragma once

// Basic triangle and bounding-box operations (host and device).
//
// Orientation convention (docs/conventions.md): vertices (a, b, c) are ordered
// counter-clockwise when viewed from the gas side, so the right-hand-rule
// normal (b - a) x (c - a) points from the solid into the gas.

#include "vleo/core/Vec3.hpp"

namespace vleo {

struct Triangle {
  Vec3 a;
  Vec3 b;
  Vec3 c;
};

// (b - a) x (c - a); its length is twice the triangle area.
KOKKOS_INLINE_FUNCTION constexpr Vec3 area_vector_x2(const Triangle& t) { return cross(t.b - t.a, t.c - t.a); }

KOKKOS_INLINE_FUNCTION double area(const Triangle& t) { return 0.5 * norm(area_vector_x2(t)); }

// Unit outward (gas-side) normal. Undefined for degenerate triangles; check
// is_degenerate() first.
KOKKOS_INLINE_FUNCTION Vec3 unit_normal(const Triangle& t) { return normalized(area_vector_x2(t)); }

KOKKOS_INLINE_FUNCTION constexpr Vec3 centroid(const Triangle& t) { return (t.a + t.b + t.c) / 3.0; }

// True if the area is below `rel_tol` times the squared longest edge, i.e.
// the triangle is (numerically) a line or a point.
KOKKOS_INLINE_FUNCTION bool is_degenerate(const Triangle& t, double rel_tol = 1.0e-12) {
  const double e = Kokkos::fmax(norm2(t.b - t.a), Kokkos::fmax(norm2(t.c - t.b), norm2(t.a - t.c)));
  return !(norm(area_vector_x2(t)) > rel_tol * e);
}

// Point a + u (b - a) + v (c - a) from barycentric coordinates (u, v).
KOKKOS_INLINE_FUNCTION constexpr Vec3 point_from_barycentric(const Triangle& t, double u, double v) {
  return t.a + u * (t.b - t.a) + v * (t.c - t.a);
}

struct Aabb {
  Vec3 lo{Kokkos::infinity_v<double>, Kokkos::infinity_v<double>,
          Kokkos::infinity_v<double>};
  Vec3 hi{-Kokkos::infinity_v<double>, -Kokkos::infinity_v<double>,
          -Kokkos::infinity_v<double>};

  KOKKOS_INLINE_FUNCTION bool empty() const { return lo.x > hi.x || lo.y > hi.y || lo.z > hi.z; }

  KOKKOS_INLINE_FUNCTION void expand(const Vec3& p) {
    lo = Vec3(Kokkos::fmin(lo.x, p.x), Kokkos::fmin(lo.y, p.y), Kokkos::fmin(lo.z, p.z));
    hi = Vec3(Kokkos::fmax(hi.x, p.x), Kokkos::fmax(hi.y, p.y), Kokkos::fmax(hi.z, p.z));
  }

  KOKKOS_INLINE_FUNCTION void expand(const Aabb& o) {
    if (o.empty()) return;
    expand(o.lo);
    expand(o.hi);
  }

  KOKKOS_INLINE_FUNCTION constexpr Vec3 extent() const { return hi - lo; }
  KOKKOS_INLINE_FUNCTION constexpr Vec3 center() const { return 0.5 * (lo + hi); }

  KOKKOS_INLINE_FUNCTION bool contains(const Vec3& p) const {
    return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z;
  }
};

KOKKOS_INLINE_FUNCTION Aabb bounding_box(const Triangle& t) {
  Aabb box;
  box.expand(t.a);
  box.expand(t.b);
  box.expand(t.c);
  return box;
}

}  // namespace vleo
