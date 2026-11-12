#include "vleo/dsmc/CollisionGrid.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace vleo {

DeviceGrid make_device_grid(const Domain& domain, double cell_size, std::int64_t max_cells) {
  if (!(cell_size > 0.0)) throw GeometryError("DSMC cell size must be positive");
  const Vec3 e = domain.box.extent();
  DeviceGrid g;
  g.origin = domain.box.lo;
  int n[3];
  for (int k = 0; k < 3; ++k) {
    const double cells = std::ceil(e[k] / cell_size * (1.0 - 1e-12));
    if (cells > 1e7) throw GeometryError("DSMC grid too fine along one axis");
    n[k] = std::max(1, static_cast<int>(cells));
  }
  g.nx = n[0];
  g.ny = n[1];
  g.nz = n[2];
  const std::int64_t total = static_cast<std::int64_t>(n[0]) * n[1] * n[2];
  if (total > max_cells) {
    std::ostringstream os;
    os << "DSMC grid with " << n[0] << " x " << n[1] << " x " << n[2] << " = " << total
       << " cells exceeds the limit of " << max_cells << " cells; increase dsmc.cell_size_m";
    throw GeometryError(os.str());
  }
  g.cell_size = Vec3(e.x / n[0], e.y / n[1], e.z / n[2]);
  g.inv_cell_size = Vec3(1.0 / g.cell_size.x, 1.0 / g.cell_size.y, 1.0 / g.cell_size.z);
  return g;
}

namespace {

// Classifies every cell: cut candidates are sampled, the others use their centre.
struct FractionKernel {
  DeviceGrid grid;
  DeviceGeometry geo;
  Kokkos::View<const std::uint8_t*> candidate;
  Kokkos::View<double*> fraction;
  int ns;
  bool has_closed;

  KOKKOS_INLINE_FUNCTION void operator()(const int c) const {
    if (!has_closed) {
      fraction(c) = 1.0;
      return;
    }
    const Aabb b = grid.cell_box(c);
    if (!candidate(c)) {
      fraction(c) = inside_body(geo, b.center()) ? 0.0 : 1.0;
      return;
    }
    int outside = 0;
    for (int k = 0; k < ns; ++k)
      for (int j = 0; j < ns; ++j)
        for (int i = 0; i < ns; ++i) {
          const Vec3 p(b.lo.x + (i + 0.5) * grid.cell_size.x / ns, b.lo.y + (j + 0.5) * grid.cell_size.y / ns,
                       b.lo.z + (k + 0.5) * grid.cell_size.z / ns);
          outside += inside_body(geo, p) ? 0 : 1;
        }
    fraction(c) = static_cast<double>(outside) / static_cast<double>(ns * ns * ns);
  }
};

}  // namespace

CollisionGrid make_collision_grid(const Domain& domain, double cell_size, const Geometry& geometry,
                                  const GridOptions& options) {
  CollisionGrid cg;
  cg.grid = make_device_grid(domain, cell_size, options.max_cells);
  const DeviceGrid& g = cg.grid;
  const int nc = g.num_cells();

  // Cut candidates: cells overlapping the bounding box of a closed-surface triangle.
  std::vector<std::uint8_t> cand(static_cast<std::size_t>(nc), 0);
  bool has_closed = false;
  auto clamp_index = [](double x, double inv, int n) {
    const int i = static_cast<int>(std::floor(x * inv));
    return std::clamp(i, 0, n - 1);
  };
  for (std::size_t t = 0; t < geometry.host.size(); ++t) {
    if (geometry.host.two_sided[t]) continue;
    has_closed = true;
    const Aabb tb = bounding_box(geometry.host.triangles[t]);
    const Vec3 lo = tb.lo - g.origin, hi = tb.hi - g.origin;
    const int x0 = clamp_index(lo.x, g.inv_cell_size.x, g.nx), x1 = clamp_index(hi.x, g.inv_cell_size.x, g.nx);
    const int y0 = clamp_index(lo.y, g.inv_cell_size.y, g.ny), y1 = clamp_index(hi.y, g.inv_cell_size.y, g.ny);
    const int z0 = clamp_index(lo.z, g.inv_cell_size.z, g.nz), z1 = clamp_index(hi.z, g.inv_cell_size.z, g.nz);
    // A triangle exactly on a cell face touches both neighbours: include them.
    for (int z = std::max(0, z0 - 1); z <= std::min(g.nz - 1, z1 + 1); ++z)
      for (int y = std::max(0, y0 - 1); y <= std::min(g.ny - 1, y1 + 1); ++y)
        for (int x = std::max(0, x0 - 1); x <= std::min(g.nx - 1, x1 + 1); ++x)
          cand[static_cast<std::size_t>(x + g.nx * (y + g.ny * z))] = 1;
  }

  Kokkos::View<std::uint8_t*> d_cand("cut_candidates", static_cast<std::size_t>(nc));
  {
    auto h = Kokkos::create_mirror_view(d_cand);
    for (int c = 0; c < nc; ++c) h(c) = cand[static_cast<std::size_t>(c)];
    Kokkos::deep_copy(d_cand, h);
  }
  cg.gas_fraction = Kokkos::View<double*>("gas_fraction", static_cast<std::size_t>(nc));
  Kokkos::parallel_for("vleo::cell_fractions", Kokkos::RangePolicy<>(0, nc),
                       FractionKernel{g, geometry.device, d_cand, cg.gas_fraction, options.samples_per_axis,
                                      has_closed});
  Kokkos::fence();

  auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cg.gas_fraction);
  Kokkos::View<double*> veff("collision_volume", static_cast<std::size_t>(nc));
  auto hv = Kokkos::create_mirror_view(veff);
  GridReport& r = cg.report;
  r.cells = nc;
  r.domain_volume_m3 = g.cell_volume() * nc;
  for (int c = 0; c < nc; ++c) {
    const double fc = f(c);
    r.cut_candidates += cand[static_cast<std::size_t>(c)];
    if (fc > 0.0) {
      ++r.gas_cells;
      r.min_positive_fraction = std::min(r.min_positive_fraction, fc);
    } else {
      ++r.solid_cells;
    }
    if (fc > 0.0 && fc < 1.0) ++r.cut_cells;
    if (fc > 0.0 && fc < options.small_fraction) ++r.small_cells;
    r.gas_volume_m3 += fc * g.cell_volume();
    hv(c) = std::max(fc, options.small_fraction) * g.cell_volume();
  }
  Kokkos::deep_copy(veff, hv);
  cg.collision_volume = veff;
  return cg;
}

}  // namespace vleo
