#pragma once

// Simulation domain: an axis-aligned box that strictly contains the body.
// In free-molecular (TPMC) flow a particle that leaves the convex box can
// never return, so the box size does not influence the result - only the
// efficiency (fraction of injected particles that reach the body).

#include "vleo/geometry/Triangle.hpp"

namespace vleo {

struct DomainSection;

struct Domain {
  Aabb box;

  // Distance parameter t >= 0 at which origin + t * dir leaves the box
  // (origin inside or on the boundary). +inf for a zero direction.
  KOKKOS_INLINE_FUNCTION double exit_distance(const Vec3& origin, const Vec3& dir) const {
    double t = Kokkos::infinity_v<double>;
    for (int k = 0; k < 3; ++k) {
      if (dir[k] > 0.0)
        t = Kokkos::fmin(t, (box.hi[k] - origin[k]) / dir[k]);
      else if (dir[k] < 0.0)
        t = Kokkos::fmin(t, (box.lo[k] - origin[k]) / dir[k]);
    }
    return Kokkos::fmax(t, 0.0);
  }

  KOKKOS_INLINE_FUNCTION double area_of_face(int face) const {
    const Vec3 e = box.extent();
    const int axis = face / 2;
    return axis == 0 ? e.y * e.z : (axis == 1 ? e.x * e.z : e.x * e.y);
  }
};

// Domain from the configuration and the mesh bounding box. Throws
// GeometryError if an explicit box does not contain the body with a margin
// of at least `min_margin` on every side.
Domain make_domain(const DomainSection& cfg, const Aabb& mesh_box, double min_margin);

}  // namespace vleo
