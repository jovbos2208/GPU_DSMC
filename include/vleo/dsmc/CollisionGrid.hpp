#pragma once

// Cartesian background grid for DSMC collisions (docs/dsmc.md section 2).
//
// The grid covers the domain box with nx * ny * nz equal cells (cell edge as
// close as possible to the requested size, exactly dividing the box).
// Cell index c = ix + nx (iy + ny iz).
//
// Gas volume of each cell (M3.9): cells not touched by any closed surface are
// classified by a point-in-body test of their centre (fraction 0 or 1). Cells
// whose box overlaps a triangle's bounding box are "cut candidates"; their
// gas fraction is the share of n^3 stratified sample points (cell-centred
// sub-cell midpoints) that lie outside every closed component. Point-in-body
// uses crossing parity along three generic directions with a majority vote.
// Open (two-sided) surfaces have no volume.
//
// Small cut cells (M3.10): the collision volume is V_eff = max(f, f_min) V_cell
// so that tiny gas fractions cannot blow up the NTC candidate number; cells
// with 0 < f < f_min are counted and reported.

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "vleo/geometry/Geometry.hpp"
#include "vleo/particles/Domain.hpp"

namespace vleo {

struct DeviceGrid {
  Vec3 origin;
  Vec3 cell_size;
  Vec3 inv_cell_size;
  int nx = 1, ny = 1, nz = 1;

  KOKKOS_INLINE_FUNCTION int num_cells() const { return nx * ny * nz; }

  // Cell containing p; points on or (by rounding) slightly outside the domain
  // boundary are clamped to the nearest boundary cell.
  KOKKOS_INLINE_FUNCTION int cell_of(const Vec3& p) const {
    auto idx = [](double x, double inv, int n) {
      int i = static_cast<int>(Kokkos::floor(x * inv));
      return i < 0 ? 0 : (i >= n ? n - 1 : i);
    };
    const Vec3 r = p - origin;
    return idx(r.x, inv_cell_size.x, nx) + nx * (idx(r.y, inv_cell_size.y, ny) + ny * idx(r.z, inv_cell_size.z, nz));
  }

  KOKKOS_INLINE_FUNCTION void indices(int c, int& ix, int& iy, int& iz) const {
    ix = c % nx;
    iy = (c / nx) % ny;
    iz = c / (nx * ny);
  }

  KOKKOS_INLINE_FUNCTION Aabb cell_box(int c) const {
    int ix, iy, iz;
    indices(c, ix, iy, iz);
    Aabb b;
    b.lo = origin + Vec3(ix * cell_size.x, iy * cell_size.y, iz * cell_size.z);
    b.hi = b.lo + cell_size;
    return b;
  }

  KOKKOS_INLINE_FUNCTION double cell_volume() const { return cell_size.x * cell_size.y * cell_size.z; }
};

struct GridOptions {
  int samples_per_axis = 16;       // stratified samples per axis in cut candidates (16^3 = 4096)
  double small_fraction = 0.05;    // f_min for the collision volume
  std::int64_t max_cells = 1 << 24;
};

struct GridReport {
  std::int64_t cells = 0;
  std::int64_t gas_cells = 0;          // f > 0
  std::int64_t solid_cells = 0;        // f == 0
  std::int64_t cut_candidates = 0;     // sampled cells
  std::int64_t cut_cells = 0;          // 0 < f < 1
  std::int64_t small_cells = 0;        // 0 < f < f_min
  double min_positive_fraction = 1.0;
  double gas_volume_m3 = 0.0;
  double domain_volume_m3 = 0.0;
};

struct CollisionGrid {
  DeviceGrid grid;
  Kokkos::View<double*> gas_fraction;      // f per cell
  Kokkos::View<double*> collision_volume;  // V_eff per cell
  GridReport report;
};

// Grid over `domain` with cells of (at most) `cell_size` edge length.
DeviceGrid make_device_grid(const Domain& domain, double cell_size, std::int64_t max_cells = 1 << 24);

CollisionGrid make_collision_grid(const Domain& domain, double cell_size, const Geometry& geometry,
                                  const GridOptions& options = {});

// Point-in-body test (majority of three parity rays) for use in kernels.
KOKKOS_INLINE_FUNCTION bool inside_body(const DeviceGeometry& g, const Vec3& p) {
  const Vec3 d1(0.8379611023, 0.4517368852, 0.3062154887);
  const Vec3 d2(-0.3874019561, 0.8436724091, 0.3716015832);
  const Vec3 d3(0.2143879951, -0.3349601827, 0.9175223469);
  const int a = g.bvh.count_crossings(g.mesh, p, d1) & 1;
  const int b = g.bvh.count_crossings(g.mesh, p, d2) & 1;
  const int c = g.bvh.count_crossings(g.mesh, p, d3) & 1;
  return a + b + c >= 2;
}

}  // namespace vleo
