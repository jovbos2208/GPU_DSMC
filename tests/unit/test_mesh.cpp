// M1.2-M1.4: mesh validation, orientation, device storage.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>

#include "DeviceTest.hpp"
#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/geometry/Primitives.hpp"

using namespace vleo;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

HostMesh build(const TriangleSoup& s, MeshReport& r, bool require_closed = true) {
  MeshOptions o;
  o.require_closed = require_closed;
  return build_host_mesh(s, o, r);
}

void expect_geometry_error(const TriangleSoup& s, const std::string& needle, bool require_closed = true) {
  MeshReport r;
  try {
    (void)build(s, r, require_closed);
    FAIL("mesh accepted");
  } catch (const GeometryError& e) {
    CHECK_THAT(std::string(e.what()), ContainsSubstring(needle));
  }
}

TriangleSoup flipped(TriangleSoup s) {
  for (auto& t : s.triangles) std::swap(t.b, t.c);
  return s;
}

struct NormalAreaOnDevice {
  DeviceMesh m;
  KOKKOS_INLINE_FUNCTION Vec3 operator()(int i) const {
    // area * normal recomputed on the device must equal the stored values
    const Triangle t = m.triangle(i);
    return 0.5 * area_vector_x2(t) - m.area(i) * m.normal(i);
  }
};

}  // namespace

TEST_CASE("Unit cube: topology, orientation, area, volume", "[mesh]") {
  MeshReport r;
  const HostMesh m = build(primitives::box({0, 0, 0}, {1, 1, 1}), r);
  CHECK(r.input_triangles == 12);
  CHECK(r.triangles == 12);
  CHECK(r.vertices == 8);
  CHECK(r.edges == 18);
  CHECK(r.boundary_edges == 0);
  CHECK(r.nonmanifold_edges == 0);
  CHECK(r.inconsistent_edges == 0);
  CHECK(r.closed);
  REQUIRE(r.components.size() == 1);
  CHECK(!r.components[0].flipped);
  CHECK_THAT(r.components[0].signed_volume_m3, WithinRel(1.0, 1e-14));
  CHECK_THAT(r.total_area_m2, WithinRel(6.0, 1e-14));
  CHECK(r.warnings.empty());
  CHECK(r.bbox.lo == Vec3(0, 0, 0));
  CHECK(r.bbox.hi == Vec3(1, 1, 1));
  // outward normals: n . (centroid - centre) > 0
  for (std::size_t i = 0; i < m.size(); ++i) {
    CHECK(dot(m.normals[i], centroid(m.triangles[i]) - Vec3(0.5, 0.5, 0.5)) > 0.0);
    CHECK(m.two_sided[i] == 0);
  }
}

TEST_CASE("Inward-oriented closed component is re-oriented", "[mesh]") {
  TriangleSoup s = flipped(primitives::box({0, 0, 0}, {2, 1, 1}));
  primitives::add_box(s, {5, 5, 5}, {6, 6, 6}, 1);  // second, correctly oriented component
  MeshReport r;
  const HostMesh m = build(s, r);
  REQUIRE(r.components.size() == 2);
  CHECK(r.components[0].flipped);
  CHECK(!r.components[1].flipped);
  CHECK_THAT(r.components[0].signed_volume_m3, WithinRel(2.0, 1e-14));
  CHECK_THAT(r.warnings.back(), ContainsSubstring("component 0 had inward-pointing normals"));
  CHECK(r.stored_normal_disagreements == 12);  // the file normals of the flipped part are wrong too
  for (std::size_t i = 0; i < m.size(); ++i) {
    const Vec3 c = m.component[i] == 0 ? Vec3(1, 0.5, 0.5) : Vec3(5.5, 5.5, 5.5);
    CHECK(dot(m.normals[i], centroid(m.triangles[i]) - c) > 0.0);
  }
  CHECK(m.material_names.size() == 2);
}

TEST_CASE("Scaled sphere: closed, area and volume converge", "[mesh]") {
  MeshReport r;
  (void)build(primitives::uv_sphere({1, -2, 3}, 0.5, 64, 128), r);
  CHECK(r.closed);
  CHECK(r.components.size() == 1);
  CHECK(r.triangles == 2 * 128 * 63);
  CHECK(r.vertices == 2 + 128 * 63);
  CHECK_THAT(r.total_area_m2, WithinRel(4.0 * constants::pi * 0.25, 2e-3));
  CHECK_THAT(r.components[0].signed_volume_m3, WithinRel(4.0 / 3.0 * constants::pi * 0.125, 3e-3));
}

TEST_CASE("Open surfaces: rejected by default, two-sided if allowed", "[mesh]") {
  TriangleSoup plate;
  primitives::add_rectangle(plate, {0, 0, 0}, {1, 0, 0}, {0, 1, 0});
  expect_geometry_error(plate, "1 of 1 component(s) are not closed (4 boundary");
  MeshReport r;
  const HostMesh m = build(plate, r, false);
  CHECK(!r.closed);
  CHECK(r.boundary_edges == 4);
  CHECK(m.two_sided[0] == 1);
  CHECK(m.two_sided[1] == 1);
  CHECK_THAT(r.warnings.back(), ContainsSubstring("treated as two-sided"));

  // A cube with one missing triangle is open as well.
  TriangleSoup cube = primitives::box({0, 0, 0}, {1, 1, 1});
  cube.triangles.pop_back();
  cube.material.pop_back();
  cube.stored_normals.pop_back();
  expect_geometry_error(cube, "3 boundary");
}

TEST_CASE("Inconsistent orientation and non-manifold edges are detected", "[mesh]") {
  TriangleSoup cube = primitives::box({0, 0, 0}, {1, 1, 1});
  std::swap(cube.triangles[3].b, cube.triangles[3].c);  // one flipped triangle
  expect_geometry_error(cube, "3 inconsistently oriented edge(s)");

  TriangleSoup fin = primitives::box({0, 0, 0}, {1, 1, 1});
  // Extra triangle on the edge (0,0,0)-(1,0,0): that edge now has 3 users.
  fin.triangles.push_back({{0, 0, 0}, {1, 0, 0}, {0.5, -1, 0.5}});
  fin.material.push_back(0);
  fin.stored_normals.push_back({});
  expect_geometry_error(fin, "1 non-manifold");
  MeshReport r;
  (void)build(fin, r, false);
  CHECK(r.nonmanifold_edges == 1);
}

TEST_CASE("Degenerate triangles are removed, empty and zero-volume meshes rejected", "[mesh]") {
  TriangleSoup s = primitives::box({0, 0, 0}, {1, 1, 1});
  s.triangles.push_back({{0, 0, 0}, {0.5, 0.5, 0.5}, {1, 1, 1}});  // collinear
  s.triangles.push_back({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});        // point
  s.material.insert(s.material.end(), {0, 0});
  s.stored_normals.insert(s.stored_normals.end(), {Vec3(), Vec3()});
  MeshReport r;
  const HostMesh m = build(s, r);
  CHECK(r.degenerate_removed == 2);
  CHECK(m.size() == 12);
  CHECK(r.closed);
  CHECK_THAT(r.warnings.front(), ContainsSubstring("2 degenerate triangle(s) removed"));

  TriangleSoup only_bad;
  only_bad.triangles.push_back({{0, 0, 0}, {1, 1, 1}, {2, 2, 2}});
  expect_geometry_error(only_bad, "no non-degenerate triangles");

  // Doubled sheet: closed (every edge used twice, opposite directions) but no volume.
  TriangleSoup sheet;
  sheet.triangles = {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}}};
  expect_geometry_error(sheet, "encloses no volume");
}

TEST_CASE("Signed zero coordinates weld; stored normals are only diagnostic", "[mesh]") {
  TriangleSoup s = primitives::box({-1, -1, -1}, {0, 0, 0});
  for (std::size_t i = 0; i < s.triangles.size(); i += 2)  // every other triangle uses -0.0
    for (Vec3* v : {&s.triangles[i].a, &s.triangles[i].b, &s.triangles[i].c})
      if (v->x == 0.0) v->x = -0.0;
  for (auto& n : s.stored_normals) n = -n;  // all file normals wrong
  MeshReport r;
  (void)build(s, r);
  CHECK(r.vertices == 8);
  CHECK(r.closed);
  CHECK(r.stored_normal_disagreements == 12);
  CHECK(!r.components[0].flipped);
}

TEST_CASE("Example satellite geometry", "[mesh]") {
  MeshReport r;
  const HostMesh m = build(read_stl(std::string(VLEO_SOURCE_DIR) + "/examples/geometry/satellite.stl"), r);
  CHECK(r.closed);
  CHECK(r.components.size() == 3);
  for (const auto& c : r.components) CHECK(!c.flipped);
  CHECK(m.material_names == std::vector<std::string>{"bus", "panel_left", "panel_right"});
  CHECK_THAT(r.components[0].signed_volume_m3, WithinRel(1.5, 1e-14));
  CHECK(r.warnings.empty());
}

TEST_CASE("Geometry pipeline: BVH order and device arrays", "[mesh][device]") {
  TriangleSoup s = primitives::uv_sphere({0, 0, 0}, 1.0, 16, 32);
  primitives::add_box(s, {2, 2, 2}, {3, 4, 5}, 1);
  Geometry g = make_geometry(s);
  REQUIRE(g.host.size() == s.triangles.size());
  // Permutation: every input triangle appears exactly once, attributes follow it.
  std::vector<std::uint32_t> src = g.host.source_index;
  std::sort(src.begin(), src.end());
  for (std::size_t i = 0; i < src.size(); ++i) REQUIRE(src[i] == i);
  for (std::size_t i = 0; i < g.host.size(); ++i) {
    const Triangle& t = g.host.triangles[i];
    const Triangle& o = s.triangles[g.host.source_index[i]];
    CHECK((t.a == o.a && t.b == o.b && t.c == o.c));
    CHECK(g.host.material[i] == s.material[g.host.source_index[i]]);
  }
  CHECK(g.device.mesh.num_triangles == static_cast<int>(g.host.size()));
  CHECK(g.device.surface_offset_m == kSurfaceOffsetRelative * 6.0);
  const auto d = test::eval_on_device<Vec3>("normal_area", g.device.mesh.num_triangles, NormalAreaOnDevice{g.device.mesh});
  for (int i = 0; i < g.device.mesh.num_triangles; ++i) CHECK(max_abs_component(d(i)) < 1e-14);
}
