// M3.2, M3.3, M3.9, M3.10: collision grid, binning, cut-cell gas volumes.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <sstream>

#include "vleo/core/Species.hpp"
#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/io/ConfigParser.hpp"

using namespace vleo;
namespace fs = std::filesystem;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Domain box_domain(const Vec3& lo, const Vec3& hi) {
  Domain d;
  d.box.lo = lo;
  d.box.hi = hi;
  return d;
}

}  // namespace

TEST_CASE("Grid construction and position-to-cell mapping", "[grid]") {
  const Domain d = box_domain({-1, -2, 0}, {1, 2, 0.5});
  const DeviceGrid g = make_device_grid(d, 0.3);
  CHECK(g.nx == 7);  // ceil(2 / 0.3)
  CHECK(g.ny == 14);
  CHECK(g.nz == 2);
  CHECK_THAT(g.cell_size.x * g.nx, WithinRel(2.0, 1e-15));
  CHECK(g.cell_size.x <= 0.3);
  CHECK(make_device_grid(d, 0.5).nx == 4);  // exact division does not add a cell
  CHECK(g.cell_of({-1, -2, 0}) == 0);
  CHECK(g.cell_of({0.999999, 1.999999, 0.4999}) == g.num_cells() - 1);
  CHECK(g.cell_of({1.0, 2.0, 0.5}) == g.num_cells() - 1);          // upper boundary clamps
  CHECK(g.cell_of({-1.0 - 1e-15, -2.0, 0.0}) == 0);                 // rounding below the box clamps
  for (int c = 0; c < g.num_cells(); c += 7) {
    const Aabb b = g.cell_box(c);
    REQUIRE(g.cell_of(b.center()) == c);
    int ix, iy, iz;
    g.indices(c, ix, iy, iz);
    REQUIRE(ix + g.nx * (iy + g.ny * iz) == c);
  }
  CHECK_THROWS_AS(make_device_grid(d, 1e-5), GeometryError);  // too many cells
}

TEST_CASE("Cut cells: box body with analytically known fractions", "[grid][cutcell]") {
  // Domain [0,4]^3 with unit cells; body [0.25, 2.5] x [1.0, 2.75] x [0.5, 3.125]:
  // every face lies at a multiple of 1/16 of a cell, so 16^3 stratified samples
  // give the exact gas fraction of every cell.
  const Vec3 lo(0.25, 1.0, 0.5), hi(2.5, 2.75, 3.125);
  const Geometry geo = make_geometry(primitives::box(lo, hi));
  const CollisionGrid cg = make_collision_grid(box_domain({0, 0, 0}, {4, 4, 4}), 1.0, geo);
  auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cg.gas_fraction);
  auto overlap = [](double a0, double a1, double b0, double b1) { return std::max(0.0, std::min(a1, b1) - std::max(a0, b0)); };
  for (int c = 0; c < cg.grid.num_cells(); ++c) {
    const Aabb b = cg.grid.cell_box(c);
    const double solid = overlap(b.lo.x, b.hi.x, lo.x, hi.x) * overlap(b.lo.y, b.hi.y, lo.y, hi.y) *
                         overlap(b.lo.z, b.hi.z, lo.z, hi.z);
    INFO("cell " << c);
    REQUIRE(f(c) == 1.0 - solid);  // exact
  }
  const double body = (hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z);
  CHECK_THAT(cg.report.gas_volume_m3, WithinRel(64.0 - body, 1e-14));
  CHECK(cg.report.solid_cells == 2);  // cells fully inside: (1,1,1) and (1,1,2)
  CHECK(cg.report.cut_cells > 0);
}

TEST_CASE("Cut cells: sphere gas volume converges to the analytic value", "[grid][cutcell]") {
  const TriangleSoup sphere = primitives::uv_sphere({0.03, -0.02, 0.01}, 0.7, 48, 96);
  const Geometry geo = make_geometry(sphere);
  const double v_body = geo.report.components[0].signed_volume_m3;  // exact polyhedral volume
  const double area = geo.report.total_area_m2;
  for (int ns : {8, 16, 32}) {
    GridOptions o;
    o.samples_per_axis = ns;
    const CollisionGrid cg = make_collision_grid(box_domain({-1, -1, -1}, {1, 1, 1}), 0.25, geo, o);
    // misclassification is confined to a shell of half a sample spacing around the surface
    const double bound = area * 0.5 * 0.25 / ns;
    INFO("samples per axis " << ns << ": gas volume " << cg.report.gas_volume_m3 << ", exact " << 8.0 - v_body);
    CHECK(std::fabs(cg.report.gas_volume_m3 - (8.0 - v_body)) < bound);
  }
}

TEST_CASE("Small cut cells are limited and reported; open surfaces have no volume", "[grid][cutcell]") {
  // Body face at x = 31/32 of the first cell column: gas fraction 1/32 < f_min = 0.05.
  const Geometry geo = make_geometry(primitives::box({1.0 / 32.0, 0.0, 0.0}, {2.0, 1.0, 1.0}));
  GridOptions o;
  o.samples_per_axis = 32;
  const CollisionGrid cg = make_collision_grid(box_domain({0, 0, 0}, {3, 1, 1}), 1.0, geo, o);
  auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cg.gas_fraction);
  auto v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cg.collision_volume);
  CHECK(f(0) == 1.0 / 32.0);
  CHECK(f(1) == 0.0);
  CHECK(f(2) == 1.0);
  CHECK(v(0) == 0.05);  // V_eff = max(f, f_min) V_cell
  CHECK(v(2) == 1.0);
  CHECK(cg.report.small_cells == 1);
  CHECK(cg.report.min_positive_fraction == 1.0 / 32.0);

  TriangleSoup plate;
  primitives::add_rectangle(plate, {0.2, 0.2, 0.5}, {0.6, 0, 0}, {0, 0.6, 0});
  MeshOptions mo;
  mo.require_closed = false;
  const Geometry pg = make_geometry(plate, mo);
  const CollisionGrid pc = make_collision_grid(box_domain({0, 0, 0}, {1, 1, 1}), 0.25, pg);
  CHECK(pc.report.gas_volume_m3 == 1.0);
  CHECK(pc.report.cut_cells == 0);
}

TEST_CASE("Binning: every particle exactly once, in its cell, sorted by id", "[grid][binning][device]") {
  const fs::path dir = fs::temp_directory_path() / "vleo_test_grid";
  fs::create_directories(dir);
  write_stl_ascii(dir / "cube.stl", primitives::box({-0.2, -0.2, -0.2}, {0.2, 0.2, 0.2}));
  const std::string yaml = std::string("simulation: {name: bin, mode: dsmc, random_seed: 3}\n") +
                           "geometry: {file: cube.stl}\ndomain: {margin_m: 0.8}\n"
                           "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
                           "gas: {number_density_m3: 1.0e20, translational_temperature_K: 300, "
                           "bulk_velocity_body_mps: [-1000, 0, 0], species: {Ar: 1.0}}\n"
                           "surface: {model: maxwell, temperature_K: 300, maxwell: {diffuse_fraction: 1}}\n"
                           "dsmc: {cell_size_mode: fixed, cell_size_m: 0.2, particles_per_cell_target: 20}\n";
  const SimulationConfig cfg = parse_simulation_config(yaml, dir);
  const Geometry geo = make_geometry(read_stl(dir / "cube.stl"));
  const Domain dom = make_domain(cfg.domain, geo.host.bbox, 1e-6);
  const InflowModel inflow = InflowModel::create(dom, cfg.gas);
  DsmcSettings s = resolve_dsmc_settings(cfg, dom);
  DsmcSimulation sim(cfg, geo, inflow, s);
  sim.fill_freestream();
  for (int k = 0; k < 5; ++k) sim.step(false);

  std::vector<int> off, list;
  sim.binning(off, list);
  const std::vector<Vec3> pos = sim.positions();
  const std::vector<std::uint64_t> ids = sim.particle_ids();
  const DeviceGrid& g = sim.grid().grid;
  REQUIRE(off.size() == static_cast<std::size_t>(g.num_cells()) + 1);
  REQUIRE(off.back() == static_cast<int>(sim.num_particles()));
  REQUIRE(list.size() == sim.num_particles());
  std::vector<int> seen(sim.num_particles(), 0);
  for (int c = 0; c < g.num_cells(); ++c) {
    for (int k = off[static_cast<std::size_t>(c)]; k < off[static_cast<std::size_t>(c) + 1]; ++k) {
      const int i = list[static_cast<std::size_t>(k)];
      ++seen[static_cast<std::size_t>(i)];
      REQUIRE(g.cell_of(pos[static_cast<std::size_t>(i)]) == c);
      if (k > off[static_cast<std::size_t>(c)])
        REQUIRE(ids[static_cast<std::size_t>(list[static_cast<std::size_t>(k) - 1])] < ids[static_cast<std::size_t>(i)]);
    }
  }
  for (int v : seen) REQUIRE(v == 1);
  // no particle inside the body
  CHECK(sim.counters().removed_inside_at_fill > 0);
  for (const Vec3& p : pos) REQUIRE(!(std::fabs(p.x) < 0.2 && std::fabs(p.y) < 0.2 && std::fabs(p.z) < 0.2));
  std::set<std::uint64_t> unique(ids.begin(), ids.end());
  CHECK(unique.size() == ids.size());
}
