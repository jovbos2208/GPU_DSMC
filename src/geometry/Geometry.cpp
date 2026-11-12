#include "vleo/geometry/Geometry.hpp"

#include <algorithm>
#include <cmath>

namespace vleo {

Geometry make_host_geometry(const TriangleSoup& soup, const MeshOptions& options) {
  Geometry g;
  g.host = build_host_mesh(soup, options, g.report);
  g.bvh = build_bvh(g.host.triangles);
  g.host.permute(g.bvh.order);

  const Aabb& b = g.host.bbox;
  g.length_scale_m = std::max({max_abs_component(b.lo), max_abs_component(b.hi), max_abs_component(b.extent())});
  g.device.surface_offset_m = kSurfaceOffsetRelative * g.length_scale_m;
  return g;
}

void upload_geometry(Geometry& g) {
  g.device.mesh = upload(g.host);
  g.device.bvh = upload(g.bvh);
}

Geometry make_geometry(const TriangleSoup& soup, const MeshOptions& options) {
  Geometry g = make_host_geometry(soup, options);
  upload_geometry(g);
  return g;
}

Geometry load_geometry(const std::filesystem::path& stl, double scale_to_m, const MeshOptions& options) {
  return make_geometry(read_stl(stl, scale_to_m), options);
}

}  // namespace vleo
