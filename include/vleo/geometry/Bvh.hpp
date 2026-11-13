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

// Device node with box coordinates of type BvhReal (M5.5, mixed precision).
// GPU builds use float boxes (32-byte nodes): RDNA3/consumer GPUs execute
// FP64 at 1/32 of the FP32 rate and the box tests dominate BVH traversal.
// CPU builds keep double boxes (float conversions cost more than they save
// there; measured in benchmarks/README.md). In both cases the boxes are
// rounded outwards and padded (float: 1e-5 times the geometry scale) and the
// slab interval is widened (float: 1e-5, double: 1e-9 relative); the margins
// exceed the rounding errors of the box test, so no box containing a
// candidate hit is culled and the closest hit is bitwise identical to brute
// force (test_ray). Triangle intersection is always double precision.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
using BvhReal = float;
inline constexpr float kBvhBoxSlack = 1.0e-5f;
inline constexpr double kBvhBoxPad = 1.0e-5;
#else
using BvhReal = double;
inline constexpr double kBvhBoxSlack = kBoxSlack;
inline constexpr double kBvhBoxPad = 0.0;
#endif

struct BvhNodeDevice {
  BvhReal lo[3];
  BvhReal hi[3];
  std::int32_t right_or_first;
  std::int32_t count;
};

struct BoxRayDevice {
  BvhReal origin[3];
  BvhReal inv_dir[3];
  KOKKOS_INLINE_FUNCTION BoxRayDevice(const Vec3& o, const Vec3& d)
      : origin{static_cast<BvhReal>(o.x), static_cast<BvhReal>(o.y), static_cast<BvhReal>(o.z)},
        inv_dir{static_cast<BvhReal>(1.0 / d.x), static_cast<BvhReal>(1.0 / d.y), static_cast<BvhReal>(1.0 / d.z)} {}
};

// Conservative slab test in BvhReal (see above).
KOKKOS_INLINE_FUNCTION bool hit_box_device(const BoxRayDevice& r, const BvhNodeDevice& n, double t_max, double& t_entry) {
  const BvhReal one = 1;
  BvhReal t0 = 0;
  BvhReal t1 = t_max < 1.0e38 ? static_cast<BvhReal>(t_max) * (one + kBvhBoxSlack) : Kokkos::infinity_v<BvhReal>;
  for (int k = 0; k < 3; ++k) {
    BvhReal tn = (n.lo[k] - r.origin[k]) * r.inv_dir[k];
    BvhReal tf = (n.hi[k] - r.origin[k]) * r.inv_dir[k];
    if (tn > tf) {
      const BvhReal tmp = tn;
      tn = tf;
      tf = tmp;
    }
    tn *= tn > 0 ? one - kBvhBoxSlack : one + kBvhBoxSlack;
    tf *= tf > 0 ? one + kBvhBoxSlack : one - kBvhBoxSlack;
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
  Kokkos::View<const BvhNodeDevice*> nodes;

  // Number of closed-surface (one-sided) triangles crossed by the ray
  // origin + t dir, t > 0 (open, two-sided surfaces are ignored). Odd parity
  // means the origin lies inside a closed component. A ray passing exactly
  // through an edge may count both neighbours; callers use generic directions
  // and a majority vote (CollisionGrid.cpp).
  KOKKOS_INLINE_FUNCTION int count_crossings(const DeviceMesh& mesh, const Vec3& origin, const Vec3& dir) const {
    const WatertightRay wr(origin, dir);
    const BoxRayDevice br(origin, dir);
    constexpr double inf = Kokkos::infinity_v<double>;
    int count = 0;
    double t_entry = 0.0;
    if (!hit_box_device(br, nodes(0), inf, t_entry)) return 0;
    int stack[kMaxBvhDepth];
    int sp = 0;
    int node = 0;
    for (;;) {
      const BvhNodeDevice& n = nodes(node);
      if (n.count > 0) {
        for (int i = n.right_or_first; i < n.right_or_first + n.count; ++i) {
          if (mesh.two_sided(i) != 0) continue;
          TriangleHit h;
          if (intersect(wr, mesh.v0(i), mesh.v1(i), mesh.v2(i), inf, h)) ++count;
        }
        if (sp == 0) break;
        node = stack[--sp];
        continue;
      }
      const int left = node + 1, right = n.right_or_first;
      double tl = 0.0, tr = 0.0;
      const bool hl = hit_box_device(br, nodes(left), inf, tl);
      const bool hr = hit_box_device(br, nodes(right), inf, tr);
      if (hl && hr) {
        stack[sp++] = right;
        node = left;
      } else if (hl) {
        node = left;
      } else if (hr) {
        node = right;
      } else {
        if (sp == 0) break;
        node = stack[--sp];
      }
    }
    return count;
  }

  // Closest hit with 0 < t <= t_max, skipping triangle `ignore` (-1: none).
  KOKKOS_INLINE_FUNCTION RayHit closest_hit(const DeviceMesh& mesh, const Vec3& origin, const Vec3& dir,
                                            double t_max, int ignore = -1) const {
    RayHit best;
    const WatertightRay wr(origin, dir);
    const BoxRayDevice br(origin, dir);
    double limit = t_max;
    double t_entry = 0.0;
    if (!hit_box_device(br, nodes(0), limit, t_entry)) return best;

    int stack[kMaxBvhDepth];
    int sp = 0;
    int node = 0;
    for (;;) {
      const BvhNodeDevice& n = nodes(node);
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
      const bool hl = hit_box_device(br, nodes(left), limit, tl);
      const bool hr = hit_box_device(br, nodes(right), limit, tr);
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
