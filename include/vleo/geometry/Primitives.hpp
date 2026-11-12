#pragma once

// Simple triangulated shapes (host side) for tests, examples and analytic
// verification cases. All closed shapes have outward normals (vertices
// counter-clockwise seen from outside).

#include <cstdint>

#include "vleo/geometry/Stl.hpp"

namespace vleo::primitives {

// Axis-aligned box [lo, hi], 12 triangles.
void add_box(TriangleSoup& soup, const Vec3& lo, const Vec3& hi, std::uint32_t material = 0);

// Rectangle (open surface, 2 triangles) spanned by corner + u, corner + v;
// normal along u x v.
void add_rectangle(TriangleSoup& soup, const Vec3& corner, const Vec3& u, const Vec3& v, std::uint32_t material = 0);

// UV sphere with n_theta >= 2 latitude bands and n_phi >= 3 longitudes
// (2 n_phi (n_theta - 1) triangles).
void add_uv_sphere(TriangleSoup& soup, const Vec3& center, double radius, int n_theta, int n_phi,
                   std::uint32_t material = 0);

TriangleSoup box(const Vec3& lo, const Vec3& hi);
TriangleSoup uv_sphere(const Vec3& center, double radius, int n_theta, int n_phi);

}  // namespace vleo::primitives
