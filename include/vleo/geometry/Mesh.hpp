#pragma once

// Validated triangle mesh: host representation, topology checks and the
// device (GPU) representation as structure of arrays.
//
// Topology (vertices are welded by exact coordinate equality, -0 == +0):
//  * degenerate triangles (zero area within a relative tolerance) are removed
//    and reported;
//  * every undirected edge is classified as interior (2 triangles traversing
//    it in opposite directions), inconsistent (2 triangles, same direction),
//    boundary (1 triangle) or non-manifold (> 2 triangles);
//  * connected components (via shared edges) are closed if none of their
//    edges is boundary, non-manifold or inconsistent.
// Orientation: a closed component with negative signed volume is flipped as a
// whole, so every closed component has outward (gas-side) normals afterwards.
// Open components are only accepted with geometry.require_closed = false and
// are then treated as two-sided surfaces.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "vleo/geometry/Stl.hpp"
#include "vleo/geometry/Triangle.hpp"

namespace vleo {

class GeometryError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct MeshOptions {
  bool require_closed = true;
  double degenerate_rel_tol = 1.0e-12;  // see is_degenerate()
};

struct ComponentInfo {
  std::uint32_t triangles = 0;
  bool closed = false;
  bool flipped = false;        // orientation was reversed to make normals point outward
  double signed_volume_m3 = 0.0;  // after orientation fix (> 0 for closed components)
  double area_m2 = 0.0;
};

struct MeshReport {
  std::size_t input_triangles = 0;
  std::size_t degenerate_removed = 0;
  std::size_t triangles = 0;
  std::size_t vertices = 0;  // after welding
  std::size_t edges = 0;
  std::size_t boundary_edges = 0;
  std::size_t nonmanifold_edges = 0;
  std::size_t inconsistent_edges = 0;
  std::size_t stored_normal_disagreements = 0;  // file normals opposite to the vertex-order normal
  std::vector<ComponentInfo> components;
  bool closed = false;  // all components closed
  double total_area_m2 = 0.0;
  Aabb bbox;
  std::vector<std::string> warnings;
};

// Host mesh: one entry per triangle in every array.
struct HostMesh {
  std::vector<Triangle> triangles;
  std::vector<Vec3> normals;              // unit outward normal (right-hand rule)
  std::vector<double> areas;
  std::vector<std::uint32_t> material;
  std::vector<std::uint32_t> component;
  std::vector<std::uint8_t> two_sided;    // 1 for triangles of open components
  std::vector<std::uint32_t> source_index;  // triangle index in the input file
  std::vector<std::string> material_names;
  Aabb bbox;

  std::size_t size() const { return triangles.size(); }
  // Reorders all per-triangle arrays: new[i] = old[order[i]].
  void permute(const std::vector<std::uint32_t>& order);
};

// Validates and orients the soup. Throws GeometryError if the mesh is empty or
// violates the options; the message lists the problems.
HostMesh build_host_mesh(const TriangleSoup& soup, const MeshOptions& options, MeshReport& report);

// Device mesh: read-only views in the default memory space.
struct DeviceMesh {
  Kokkos::View<const Vec3*> v0, v1, v2;
  Kokkos::View<const Vec3*> normal;
  Kokkos::View<const double*> area;
  Kokkos::View<const std::uint32_t*> material;
  Kokkos::View<const std::uint8_t*> two_sided;
  int num_triangles = 0;

  KOKKOS_INLINE_FUNCTION Triangle triangle(int i) const { return {v0(i), v1(i), v2(i)}; }
};

DeviceMesh upload(const HostMesh& mesh);

}  // namespace vleo
