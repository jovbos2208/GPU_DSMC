#pragma once

// Geometry pipeline: STL -> validated, oriented mesh -> BVH -> device arrays.

#include <filesystem>

#include "vleo/geometry/Bvh.hpp"
#include "vleo/geometry/Mesh.hpp"
#include "vleo/geometry/Stl.hpp"

namespace vleo {

// Relative origin offset applied after a surface interaction, in units of the
// geometry length scale (largest absolute coordinate or extent of the mesh
// bounding box). See docs/geometry.md.
inline constexpr double kSurfaceOffsetRelative = 1.0e-9;

// Device-side view of the geometry (cheap to copy into kernels).
struct DeviceGeometry {
  DeviceMesh mesh;
  DeviceBvh bvh;
  double surface_offset_m = 0.0;

  KOKKOS_INLINE_FUNCTION RayHit closest_hit(const Vec3& origin, const Vec3& dir, double t_max,
                                            int ignore = -1) const {
    return bvh.closest_hit(mesh, origin, dir, t_max, ignore);
  }
};

struct Geometry {
  HostMesh host;     // triangles in BVH leaf order
  MeshReport report;
  HostBvh bvh;
  DeviceGeometry device;
  double length_scale_m = 0.0;
};

// Host part only (validation, orientation, BVH); no Kokkos runtime needed.
// `device` stays empty until upload_geometry() is called.
Geometry make_host_geometry(const TriangleSoup& soup, const MeshOptions& options = {});

// Copies mesh and BVH to the default memory space (Kokkos must be initialised).
void upload_geometry(Geometry& g);

// make_host_geometry() followed by upload_geometry().
Geometry make_geometry(const TriangleSoup& soup, const MeshOptions& options = {});

// Reads an STL file (coordinates multiplied by scale_to_m) and builds the geometry.
Geometry load_geometry(const std::filesystem::path& stl, double scale_to_m, const MeshOptions& options = {});

}  // namespace vleo
