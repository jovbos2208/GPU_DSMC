#include "vleo/geometry/Bvh.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace vleo {

namespace {

double surface_area(const Aabb& b) {
  if (b.empty()) return 0.0;
  const Vec3 e = b.extent();
  return 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x);
}

struct Prim {
  Aabb box;
  Vec3 centroid;
  std::uint32_t index;
};

class Builder {
 public:
  Builder(std::vector<Prim>& prims, const BvhBuildOptions& opt, HostBvh& out) : prims_(prims), opt_(opt), out_(out) {}

  int build(std::size_t begin, std::size_t end, int depth) {
    const int node_index = static_cast<int>(out_.nodes.size());
    out_.nodes.emplace_back();
    out_.stats.max_depth = std::max(out_.stats.max_depth, depth);

    Aabb box, cbox;
    for (std::size_t i = begin; i < end; ++i) {
      box.expand(prims_[i].box);
      cbox.expand(prims_[i].centroid);
    }
    out_.nodes[static_cast<std::size_t>(node_index)].lo = box.lo;
    out_.nodes[static_cast<std::size_t>(node_index)].hi = box.hi;

    const std::size_t n = end - begin;
    std::size_t mid = begin;
    const bool split = n > static_cast<std::size_t>(opt_.max_leaf_size) && depth < kMaxBvhDepth - 1 &&
                       choose_split(begin, end, box, cbox, mid);
    if (!split) {
      make_leaf(node_index, begin, end);
      return node_index;
    }
    build(begin, mid, depth + 1);  // left child = node_index + 1
    const int right = build(mid, end, depth + 1);
    out_.nodes[static_cast<std::size_t>(node_index)].right_or_first = right;
    out_.nodes[static_cast<std::size_t>(node_index)].count = 0;
    return node_index;
  }

 private:
  void make_leaf(int node_index, std::size_t begin, std::size_t end) {
    BvhNode& node = out_.nodes[static_cast<std::size_t>(node_index)];
    node.right_or_first = static_cast<std::int32_t>(out_.order.size());
    node.count = static_cast<std::int32_t>(end - begin);
    for (std::size_t i = begin; i < end; ++i) out_.order.push_back(prims_[i].index);
    ++out_.stats.leaves;
    out_.stats.max_leaf_triangles = std::max(out_.stats.max_leaf_triangles, node.count);
  }

  // Binned SAH over all three axes. Returns false if a leaf is preferable.
  // Falls back to a median split if the SAH cannot separate the primitives
  // but the leaf would be too large (e.g. identical centroids).
  bool choose_split(std::size_t begin, std::size_t end, const Aabb& box, const Aabb& cbox, std::size_t& mid) {
    const std::size_t n = end - begin;
    const int nb = opt_.bins;
    const double parent_area = surface_area(box);

    double best_cost = std::numeric_limits<double>::infinity();
    int best_axis = -1, best_bin = -1;
    std::vector<Aabb> bin_box(static_cast<std::size_t>(nb));
    std::vector<std::size_t> bin_count(static_cast<std::size_t>(nb));
    std::vector<double> right_area(static_cast<std::size_t>(nb));
    std::vector<std::size_t> right_count(static_cast<std::size_t>(nb));

    for (int axis = 0; axis < 3; ++axis) {
      const double lo = cbox.lo[axis], ext = cbox.hi[axis] - cbox.lo[axis];
      if (!(ext > 0.0)) continue;
      std::fill(bin_box.begin(), bin_box.end(), Aabb{});
      std::fill(bin_count.begin(), bin_count.end(), 0);
      for (std::size_t i = begin; i < end; ++i) {
        const int b = bin_of(prims_[i].centroid[axis], lo, ext, nb);
        bin_box[static_cast<std::size_t>(b)].expand(prims_[i].box);
        ++bin_count[static_cast<std::size_t>(b)];
      }
      Aabb acc;
      std::size_t cnt = 0;
      for (int b = nb - 1; b > 0; --b) {
        acc.expand(bin_box[static_cast<std::size_t>(b)]);
        cnt += bin_count[static_cast<std::size_t>(b)];
        right_area[static_cast<std::size_t>(b)] = surface_area(acc);
        right_count[static_cast<std::size_t>(b)] = cnt;
      }
      acc = Aabb{};
      cnt = 0;
      for (int b = 0; b < nb - 1; ++b) {  // split between bin b and b+1
        acc.expand(bin_box[static_cast<std::size_t>(b)]);
        cnt += bin_count[static_cast<std::size_t>(b)];
        const std::size_t rc = right_count[static_cast<std::size_t>(b + 1)];
        if (cnt == 0 || rc == 0) continue;
        const double cost =
            opt_.traversal_cost + (surface_area(acc) * static_cast<double>(cnt) +
                                   right_area[static_cast<std::size_t>(b + 1)] * static_cast<double>(rc)) /
                                      parent_area;
        if (cost < best_cost) {
          best_cost = cost;
          best_axis = axis;
          best_bin = b;
        }
      }
    }

    const double leaf_cost = static_cast<double>(n);
    if (best_axis >= 0 && (best_cost < leaf_cost || n > 4 * static_cast<std::size_t>(opt_.max_leaf_size))) {
      const double lo = cbox.lo[best_axis], ext = cbox.hi[best_axis] - cbox.lo[best_axis];
      const auto it = std::partition(prims_.begin() + static_cast<std::ptrdiff_t>(begin),
                                     prims_.begin() + static_cast<std::ptrdiff_t>(end), [&](const Prim& p) {
                                       return bin_of(p.centroid[best_axis], lo, ext, nb) <= best_bin;
                                     });
      mid = static_cast<std::size_t>(it - prims_.begin());
      if (mid > begin && mid < end) return true;
    }
    if (n > 4 * static_cast<std::size_t>(opt_.max_leaf_size)) {
      // Median split along the largest box extent (deterministic by index).
      const Vec3 e = box.extent();
      const int axis = (e.x >= e.y && e.x >= e.z) ? 0 : (e.y >= e.z ? 1 : 2);
      mid = begin + n / 2;
      std::nth_element(prims_.begin() + static_cast<std::ptrdiff_t>(begin),
                       prims_.begin() + static_cast<std::ptrdiff_t>(mid),
                       prims_.begin() + static_cast<std::ptrdiff_t>(end), [axis](const Prim& a, const Prim& b) {
                         return a.centroid[axis] < b.centroid[axis] ||
                                (a.centroid[axis] == b.centroid[axis] && a.index < b.index);
                       });
      return true;
    }
    return false;
  }

  static int bin_of(double c, double lo, double ext, int nb) {
    const int b = static_cast<int>(static_cast<double>(nb) * (c - lo) / ext);
    return std::clamp(b, 0, nb - 1);
  }

  std::vector<Prim>& prims_;
  const BvhBuildOptions& opt_;
  HostBvh& out_;
};

}  // namespace

HostBvh build_bvh(const std::vector<Triangle>& triangles, const BvhBuildOptions& options) {
  if (triangles.empty()) throw GeometryError("cannot build a BVH for an empty mesh");
  if (triangles.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw GeometryError("too many triangles for 32-bit BVH indices");
  std::vector<Prim> prims(triangles.size());
  for (std::size_t i = 0; i < triangles.size(); ++i) {
    prims[i].box = bounding_box(triangles[i]);
    prims[i].centroid = prims[i].box.center();
    prims[i].index = static_cast<std::uint32_t>(i);
  }
  HostBvh bvh;
  bvh.nodes.reserve(2 * triangles.size());
  bvh.order.reserve(triangles.size());
  Builder(prims, options, bvh).build(0, prims.size(), 0);
  bvh.stats.nodes = bvh.nodes.size();

  // SAH cost of the final tree (diagnostic).
  const double root_area = surface_area(Aabb{bvh.nodes[0].lo, bvh.nodes[0].hi});
  double cost = 0.0;
  for (const auto& nd : bvh.nodes) {
    const double a = root_area > 0.0 ? surface_area(Aabb{nd.lo, nd.hi}) / root_area : 1.0;
    cost += nd.count > 0 ? a * nd.count : a * options.traversal_cost;
  }
  bvh.stats.sah_cost = cost;
  return bvh;
}

DeviceBvh upload(const HostBvh& bvh) {
  Kokkos::View<BvhNode*> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("bvh_nodes")), bvh.nodes.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < bvh.nodes.size(); ++i) h(i) = bvh.nodes[i];
  Kokkos::deep_copy(d, h);
  DeviceBvh out;
  out.nodes = d;
  return out;
}

}  // namespace vleo
