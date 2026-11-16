#include "vleo/dsmc/CollisionGrid.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "vleo/core/Comm.hpp"

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

// Classifies the cells that are not cut candidates by their centre (cut
// candidates: CutCellKernel).
struct FractionKernel {
  DeviceGrid grid;
  DeviceGeometry geo;
  Kokkos::View<const std::uint8_t*> candidate;
  Kokkos::View<double*> fraction;
  int ns;
  bool has_closed;
  int first = 0, stride = 1;  // cells first, first + stride, ... (one share per MPI rank)

  KOKKOS_INLINE_FUNCTION void operator()(const int share_index) const {
    const int c = first + stride * share_index;
    if (!has_closed) {
      fraction(c) = 1.0;
      return;
    }
    if (candidate(c)) return;
    fraction(c) = inside_body(geo, grid.cell_box(c).center()) ? 0.0 : 1.0;
  }
};

// Cut candidates: one team per cell, the ns^3 stratified samples are shared
// among the team's threads (M6.7: one thread per cell left the GPU waiting
// for the few thousand cells near the body, independent of their number).
// The outside count is an integer, so the fraction does not depend on the
// summation order.
struct CutCellKernel {
  using Member = Kokkos::TeamPolicy<>::member_type;
  DeviceGrid grid;
  DeviceGeometry geo;
  Kokkos::View<const int*> cells;
  Kokkos::View<double*> fraction;
  int ns;

  KOKKOS_INLINE_FUNCTION void operator()(const Member& t) const {
    const int c = cells(t.league_rank());
    const Aabb b = grid.cell_box(c);
    const int total = ns * ns * ns;
    int outside = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(t, total),
        [&](const int q, int& o) {
          const int i = q % ns, j = (q / ns) % ns, k = q / (ns * ns);
          const Vec3 p(b.lo.x + (i + 0.5) * grid.cell_size.x / ns, b.lo.y + (j + 0.5) * grid.cell_size.y / ns,
                       b.lo.z + (k + 0.5) * grid.cell_size.z / ns);
          o += inside_body(geo, p) ? 0 : 1;
        },
        outside);
    Kokkos::single(Kokkos::PerTeam(t), [&]() { fraction(c) = static_cast<double>(outside) / static_cast<double>(total); });
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
  // Several MPI ranks (M6.3): every rank classifies every P-th cell
  // (interleaved, so the expensive cut cells near the body are shared evenly);
  // the sum over ranks is exact (one non-zero contribution per cell).
  const int nranks = comm::size(), me = comm::rank();
  cg.gas_fraction = Kokkos::View<double*>("gas_fraction", static_cast<std::size_t>(nc));
  const int mine = nc > me ? (nc - me + nranks - 1) / nranks : 0;
  if (mine > 0)
    Kokkos::parallel_for("vleo::cell_fractions", Kokkos::RangePolicy<>(0, mine),
                         FractionKernel{g, geometry.device, d_cand, cg.gas_fraction, options.samples_per_axis,
                                        has_closed, me, nranks});
  if (has_closed) {  // cut candidates of this rank: every P-th in index order
    std::vector<int> own;
    int k = 0;
    for (int c = 0; c < nc; ++c)
      if (cand[static_cast<std::size_t>(c)] && k++ % nranks == me) own.push_back(c);
    if (!own.empty()) {
      Kokkos::View<int*> d_own("cut_cells", own.size());
      auto h = Kokkos::create_mirror_view(d_own);
      for (std::size_t q = 0; q < own.size(); ++q) h(q) = own[q];
      Kokkos::deep_copy(d_own, h);
      Kokkos::parallel_for("vleo::cut_cell_fractions",
                           Kokkos::TeamPolicy<>(static_cast<int>(own.size()), Kokkos::AUTO),
                           CutCellKernel{g, geometry.device, d_own, cg.gas_fraction, options.samples_per_axis});
    }
  }
  Kokkos::fence();

  Kokkos::View<double*, Kokkos::HostSpace> f("gas_fraction_host", static_cast<std::size_t>(nc));
  Kokkos::deep_copy(f, cg.gas_fraction);
  if (nranks > 1) {
    comm::sum(f.data(), f.size());
    Kokkos::deep_copy(cg.gas_fraction, f);
  }
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
