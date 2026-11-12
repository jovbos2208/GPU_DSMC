#pragma once

// Bounding volume hierarchy over the mesh triangles.
//
// Built on the host with a binned surface-area heuristic (SAH) and stored as a
// flat node array in depth-first order: the left child of an inner node is
// the next node, the right child index is stored explicitly. Leaves reference
// a contiguous triangle range; the mesh is reordered into leaf order after
// the build (HostMesh::permute), so leaf ranges index the mesh directly.
//
// Traversal is iterative with a fixed-size stack (max depth kMaxBvhDepth,
// enforced by the builder) and runs unchanged on CPU, CUDA and HIP.
//
// The ray/box test is conservative in t: the slab interval [t_near, t_far]
// is widened by the relative slack kBoxSlack before it is compared with the
// current search limit. The limit itself is the *computed* t of the best
// triangle so far, which carries rounding errors of the triangle test; the
// slack (1e-9, far above these errors) guarantees that no box containing a
// triangle with computed t <= limit is culled - in particular for rays aimed
// exactly at vertices on box corners. Hence the BVH returns the same closest
// hit as a brute-force loop over all triangles with the same tie-breaking
// rule; extra cost: boxes within 1e-9 relative distance are visited.

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "vleo/geometry/Mesh.hpp"
#include "vleo/geometry/RayTriangle.hpp"

namespace vleo {

inline constexpr int kMaxBvhDepth = 64;

struct BvhNode {
  Vec3 lo;
  Vec3 hi;
  std::int32_t right_or_first = 0;  // inner: index of right child; leaf: first triangle
  std::int32_t count = 0;           // leaf: number of triangles (> 0); inner: 0
};

struct BvhBuildOptions {
  int max_leaf_size = 4;    // leaves are split above this size if SAH allows
  int bins = 16;            // SAH bins per axis
  double traversal_cost = 1.0;      // relative to one triangle test
};

struct BvhStats {
  std::size_t nodes = 0;
  std::size_t leaves = 0;
  int max_depth = 0;
  int max_leaf_triangles = 0;
  double sah_cost = 0.0;
};

struct HostBvh {
  std::vector<BvhNode> nodes;
  std::vector<std::uint32_t> order;  // order[i] = original triangle index of leaf slot i
  BvhStats stats;
};

// Builds the BVH for the given triangles. The returned `order` must be applied
// to the mesh (HostMesh::permute) before the node ranges are valid.
HostBvh build_bvh(const std::vector<Triangle>& triangles, const BvhBuildOptions& options = {});

struct RayHit {
  double t = 0.0;
  int triangle = -1;  // -1: no hit
  double b0 = 0.0, b1 = 0.0, b2 = 0.0;

  KOKKOS_INLINE_FUNCTION bool valid() const { return triangle >= 0; }
};

// Precomputed reciprocal direction for slab tests.
struct BoxRay {
  Vec3 origin;
  Vec3 inv_dir;
  KOKKOS_INLINE_FUNCTION BoxRay(const Vec3& o, const Vec3& d) : origin(o), inv_dir(1.0 / d.x, 1.0 / d.y, 1.0 / d.z) {}
};

inline constexpr double kBoxSlack = 1.0e-9;

// Conservative slab test: true if the ray segment [0, t_max] may touch the box.
KOKKOS_INLINE_FUNCTION bool hit_box(const BoxRay& r, const Vec3& lo, const Vec3& hi, double t_max, double& t_entry) {
  double t0 = 0.0, t1 = t_max * (1.0 + kBoxSlack);
  for (int k = 0; k < 3; ++k) {
    double tn = (lo[k] - r.origin[k]) * r.inv_dir[k];
    double tf = (hi[k] - r.origin[k]) * r.inv_dir[k];
    if (tn > tf) {
      const double tmp = tn;
      tn = tf;
      tf = tmp;
    }
    tn *= tn > 0.0 ? 1.0 - kBoxSlack : 1.0 + kBoxSlack;
    tf *= tf > 0.0 ? 1.0 + kBoxSlack : 1.0 - kBoxSlack;
    // fmax/fmin ignore NaN (0 * inf for a ray in the slab plane): conservative.
    t0 = Kokkos::fmax(t0, tn);
    t1 = Kokkos::fmin(t1, tf);
    if (t0 > t1) return false;
  }
  t_entry = t0;
  return true;
}

// Keeps the closest hit; exact ties go to the smaller triangle index.
KOKKOS_INLINE_FUNCTION void consider(RayHit& best, int tri, const TriangleHit& h) {
  if (!best.valid() || h.t < best.t || (h.t == best.t && tri < best.triangle)) {
    best.t = h.t;
    best.triangle = tri;
    best.b0 = h.b0;
    best.b1 = h.b1;
    best.b2 = h.b2;
  }
}

struct DeviceBvh {
  Kokkos::View<const BvhNode*> nodes;

  // Closest hit with 0 < t <= t_max, skipping triangle `ignore` (-1: none).
  KOKKOS_INLINE_FUNCTION RayHit closest_hit(const DeviceMesh& mesh, const Vec3& origin, const Vec3& dir,
                                            double t_max, int ignore = -1) const {
    RayHit best;
    const WatertightRay wr(origin, dir);
    const BoxRay br(origin, dir);
    double limit = t_max;
    double t_entry = 0.0;
    if (!hit_box(br, nodes(0).lo, nodes(0).hi, limit, t_entry)) return best;

    int stack[kMaxBvhDepth];
    int sp = 0;
    int node = 0;
    for (;;) {
      const BvhNode& n = nodes(node);
      if (n.count > 0) {
        for (int i = n.right_or_first; i < n.right_or_first + n.count; ++i) {
          if (i == ignore) continue;
          TriangleHit h;
          if (intersect(wr, mesh.v0(i), mesh.v1(i), mesh.v2(i), limit, h)) {
            consider(best, i, h);
            limit = best.t;
          }
        }
        if (sp == 0) break;
        node = stack[--sp];
        continue;
      }
      const int left = node + 1;
      const int right = n.right_or_first;
      double tl = 0.0, tr = 0.0;
      const bool hl = hit_box(br, nodes(left).lo, nodes(left).hi, limit, tl);
      const bool hr = hit_box(br, nodes(right).lo, nodes(right).hi, limit, tr);
      if (hl && hr) {
        // Visit the nearer child first; the farther one is culled later if a
        // closer hit was found meanwhile (the limit check is repeated).
        const int near_child = tl <= tr ? left : right;
        const int far_child = tl <= tr ? right : left;
        stack[sp++] = far_child;
        node = near_child;
      } else if (hl) {
        node = left;
      } else if (hr) {
        node = right;
      } else {
        if (sp == 0) break;
        node = stack[--sp];
      }
    }
    return best;
  }
};

DeviceBvh upload(const HostBvh& bvh);

// Reference: closest hit by testing every triangle (same tie-breaking).
KOKKOS_INLINE_FUNCTION RayHit closest_hit_brute_force(const DeviceMesh& mesh, const Vec3& origin, const Vec3& dir,
                                                      double t_max, int ignore = -1) {
  RayHit best;
  const WatertightRay wr(origin, dir);
  double limit = t_max;
  for (int i = 0; i < mesh.num_triangles; ++i) {
    if (i == ignore) continue;
    TriangleHit h;
    if (intersect(wr, mesh.v0(i), mesh.v1(i), mesh.v2(i), limit, h)) {
      consider(best, i, h);
      limit = best.t;
    }
  }
  return best;
}

}  // namespace vleo
