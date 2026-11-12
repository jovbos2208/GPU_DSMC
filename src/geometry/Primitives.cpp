#include "vleo/geometry/Primitives.hpp"

#include <cmath>

#include "vleo/core/PhysicalConstants.hpp"

namespace vleo::primitives {

namespace {
void add_tri(TriangleSoup& s, const Vec3& a, const Vec3& b, const Vec3& c, std::uint32_t m) {
  s.triangles.push_back({a, b, c});
  s.material.push_back(m);
  s.stored_normals.push_back(normalized_or(cross(b - a, c - a), Vec3()));
  while (s.solid_names.size() <= m) s.solid_names.push_back("solid" + std::to_string(s.solid_names.size()));
}
}  // namespace

void add_rectangle(TriangleSoup& soup, const Vec3& corner, const Vec3& u, const Vec3& v, std::uint32_t material) {
  add_tri(soup, corner, corner + u, corner + u + v, material);
  add_tri(soup, corner, corner + u + v, corner + v, material);
}

void add_box(TriangleSoup& soup, const Vec3& lo, const Vec3& hi, std::uint32_t material) {
  const Vec3 e = hi - lo;
  const Vec3 ex(e.x, 0, 0), ey(0, e.y, 0), ez(0, 0, e.z);
  // Each face: corner + u x v points outward.
  add_rectangle(soup, lo, ey, ex, material);       // z = lo, normal -z
  add_rectangle(soup, lo + ez, ex, ey, material);  // z = hi, normal +z
  add_rectangle(soup, lo, ex, ez, material);       // y = lo, normal -y
  add_rectangle(soup, lo + ey, ez, ex, material);  // y = hi, normal +y
  add_rectangle(soup, lo, ez, ey, material);       // x = lo, normal -x
  add_rectangle(soup, lo + ex, ey, ez, material);  // x = hi, normal +x
}

void add_uv_sphere(TriangleSoup& soup, const Vec3& center, double radius, int n_theta, int n_phi,
                   std::uint32_t material) {
  auto point = [&](int it, int ip) {
    if (it == 0) return center + Vec3(0, 0, radius);
    if (it == n_theta) return center - Vec3(0, 0, radius);
    const double th = constants::pi * it / n_theta;
    const double ph = constants::two_pi * (ip % n_phi) / n_phi;
    return center + radius * Vec3(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th));
  };
  for (int it = 0; it < n_theta; ++it) {
    for (int ip = 0; ip < n_phi; ++ip) {
      const Vec3 a = point(it, ip), b = point(it + 1, ip), c = point(it + 1, ip + 1), d = point(it, ip + 1);
      if (it > 0) add_tri(soup, a, b, d, material);             // upper triangle (absent at the north pole)
      if (it < n_theta - 1) add_tri(soup, d, b, c, material);   // lower triangle (absent at the south pole)
    }
  }
}

TriangleSoup box(const Vec3& lo, const Vec3& hi) {
  TriangleSoup s;
  add_box(s, lo, hi);
  return s;
}

TriangleSoup uv_sphere(const Vec3& center, double radius, int n_theta, int n_phi) {
  TriangleSoup s;
  add_uv_sphere(s, center, radius, n_theta, n_phi);
  return s;
}

}  // namespace vleo::primitives
