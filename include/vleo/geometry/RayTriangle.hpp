#pragma once

// Watertight ray-triangle intersection (Woop, Benthin, Wald: "Watertight
// Ray/Triangle Intersection", JCGT 2(1), 2013) in double precision.
//
// Robustness / epsilon strategy (docs/geometry.md):
//  * No epsilon in the inside test. The ray is transformed so that it points
//    along +z; the 2D edge functions U, V, W of a shared edge are evaluated
//    from the same two transformed vertices, so the neighbouring triangle sees
//    exactly the negated value. A ray hitting an edge or vertex therefore hits
//    at least one of the adjacent triangles: no particle can slip through a
//    closed mesh. Points exactly on an edge are accepted (U, V or W == 0), so
//    both neighbours may report the hit; the BVH keeps the closest hit and
//    breaks exact ties by the smaller triangle index (deterministic).
//  * The products in the edge functions are evaluated without fused
//    multiply-add (FMA contraction would break the exact negation on GPUs).
//  * Hits are accepted for 0 < t <= t_max. Self-hits after a surface
//    interaction are avoided by the caller: the new origin is offset along
//    the gas-side normal and the triangle just hit is excluded
//    (see Transport.hpp, surface_offset).
//  * Rays parallel to the triangle plane (det == 0) never hit.

#include "vleo/core/Vec3.hpp"
#include "vleo/geometry/Triangle.hpp"

namespace vleo {

// Product that is never contracted into an FMA.
KOKKOS_INLINE_FUNCTION double mul_rn(double a, double b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __dmul_rn(a, b);
#else
  // Host: GCC/Clang do not contract across separate statements in ISO mode
  // without -ffp-contract=fast; volatile guarantees it for every compiler.
  volatile double p = a * b;
  return p;
#endif
}

// Per-ray precomputation (shear transform onto +z).
struct WatertightRay {
  Vec3 origin;
  Vec3 dir;
  int kx = 0, ky = 1, kz = 2;
  double sx = 0.0, sy = 0.0, sz = 0.0;

  KOKKOS_INLINE_FUNCTION WatertightRay() = default;

  KOKKOS_INLINE_FUNCTION WatertightRay(const Vec3& o, const Vec3& d) : origin(o), dir(d) {
    const double ax = Kokkos::fabs(d.x), ay = Kokkos::fabs(d.y), az = Kokkos::fabs(d.z);
    kz = (ax > ay) ? (ax > az ? 0 : 2) : (ay > az ? 1 : 2);
    kx = (kz + 1) % 3;
    ky = (kx + 1) % 3;
    if (d[kz] < 0.0) {  // preserve winding
      const int tmp = kx;
      kx = ky;
      ky = tmp;
    }
    sx = d[kx] / d[kz];
    sy = d[ky] / d[kz];
    sz = 1.0 / d[kz];
  }
};

struct TriangleHit {
  double t;     // ray parameter of the hit point (origin + t * dir)
  double b0;    // barycentric weight of vertex a
  double b1;    // ... of vertex b
  double b2;    // ... of vertex c
};

// Returns true and fills `hit` if the ray hits the triangle with
// 0 < t <= t_max. Back faces are hit as well; the caller decides.
KOKKOS_INLINE_FUNCTION bool intersect(const WatertightRay& r, const Vec3& v0, const Vec3& v1, const Vec3& v2,
                                      double t_max, TriangleHit& hit) {
  const Vec3 a = v0 - r.origin;
  const Vec3 b = v1 - r.origin;
  const Vec3 c = v2 - r.origin;

  const double ax = a[r.kx] - r.sx * a[r.kz];
  const double ay = a[r.ky] - r.sy * a[r.kz];
  const double bx = b[r.kx] - r.sx * b[r.kz];
  const double by = b[r.ky] - r.sy * b[r.kz];
  const double cx = c[r.kx] - r.sx * c[r.kz];
  const double cy = c[r.ky] - r.sy * c[r.kz];

  const double u = mul_rn(cx, by) - mul_rn(cy, bx);
  const double v = mul_rn(ax, cy) - mul_rn(ay, cx);
  const double w = mul_rn(bx, ay) - mul_rn(by, ax);

  if ((u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0)) return false;
  const double det = u + v + w;
  if (det == 0.0) return false;

  const double az = r.sz * a[r.kz];
  const double bz = r.sz * b[r.kz];
  const double cz = r.sz * c[r.kz];
  const double t_scaled = u * az + v * bz + w * cz;
  const double t = t_scaled / det;
  if (!(t > 0.0) || !(t <= t_max)) return false;

  const double inv = 1.0 / det;
  hit.t = t;
  hit.b0 = u * inv;
  hit.b1 = v * inv;
  hit.b2 = w * inv;
  return true;
}

KOKKOS_INLINE_FUNCTION bool intersect(const WatertightRay& r, const Triangle& tri, double t_max, TriangleHit& hit) {
  return intersect(r, tri.a, tri.b, tri.c, t_max, hit);
}

}  // namespace vleo
