// M1.1: STL import (ASCII and binary) with unit conversion.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>

#include "vleo/geometry/Primitives.hpp"
#include "vleo/geometry/Stl.hpp"

using namespace vleo;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;

namespace {

fs::path temp_dir() {
  const fs::path d = fs::temp_directory_path() / "vleo_test_stl";
  fs::create_directories(d);
  return d;
}

fs::path write_text(const std::string& name, const std::string& text) {
  const fs::path p = temp_dir() / name;
  std::ofstream(p, std::ios::binary) << text;
  return p;
}

void expect_stl_error(const fs::path& p, const std::string& needle) {
  try {
    (void)read_stl(p);
    FAIL("accepted invalid STL " << p);
  } catch (const StlError& e) {
    CHECK_THAT(std::string(e.what()), ContainsSubstring(needle));
  }
}

const char* const kOneTriangle = R"(solid one
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid one
)";

}  // namespace

TEST_CASE("ASCII STL: single triangle", "[stl]") {
  const TriangleSoup s = read_stl(write_text("one.stl", kOneTriangle));
  REQUIRE(s.triangles.size() == 1);
  CHECK(s.format == StlFormat::Ascii);
  CHECK(s.triangles[0].b == Vec3(1, 0, 0));
  CHECK(s.stored_normals[0] == Vec3(0, 0, 1));
  CHECK(s.material[0] == 0);
  REQUIRE(s.solid_names.size() == 1);
  CHECK(s.solid_names[0] == "one");
}

TEST_CASE("ASCII STL: unit conversion, whitespace, multiple solids", "[stl]") {
  const std::string text = std::string("\n\n") + kOneTriangle +
                           "solid second part\n facet normal 0 0 0\n outer loop\n"
                           "\tvertex 1e3 +2 -3.5\n vertex 0 0 0\n vertex 0 1E1 0\n endloop\n endfacet\nendsolid\n";
  const TriangleSoup s = read_stl(write_text("multi.stl", text), 1e-3);  // mm -> m
  REQUIRE(s.triangles.size() == 2);
  CHECK(s.triangles[0].b == Vec3(1e-3, 0, 0));
  CHECK(s.triangles[1].a == Vec3(1.0, 2e-3, -3.5e-3));
  CHECK(s.triangles[1].c == Vec3(0, 1e-2, 0));
  CHECK(s.material[0] == 0);
  CHECK(s.material[1] == 1);
  REQUIRE(s.solid_names.size() == 2);
  CHECK(s.solid_names[1] == "second part");
}

TEST_CASE("ASCII and binary writers round-trip", "[stl]") {
  TriangleSoup box = primitives::box({-0.5, -0.25, 0.0}, {0.5, 0.25, 2.0});
  primitives::add_box(box, {2, 2, 2}, {3, 3, 3}, 1);

  write_stl_ascii(temp_dir() / "box_ascii.stl", box);
  const TriangleSoup a = read_stl(temp_dir() / "box_ascii.stl");
  REQUIRE(a.triangles.size() == box.triangles.size());
  for (std::size_t i = 0; i < a.triangles.size(); ++i) {
    CHECK(a.triangles[i].a == box.triangles[i].a);  // shortest round-trip decimal: exact
    CHECK(a.triangles[i].c == box.triangles[i].c);
    CHECK(a.material[i] == box.material[i]);
  }

  write_stl_binary(temp_dir() / "box_binary.stl", box);
  CHECK(fs::file_size(temp_dir() / "box_binary.stl") == 84 + 50 * box.triangles.size());
  const TriangleSoup b = read_stl(temp_dir() / "box_binary.stl", 100.0);  // e.g. m -> cm
  CHECK(b.format == StlFormat::Binary);
  REQUIRE(b.triangles.size() == box.triangles.size());
  for (std::size_t i = 0; i < b.triangles.size(); ++i) {
    // binary STL is single precision; these coordinates are exact in float
    CHECK(b.triangles[i].b == 100.0 * box.triangles[i].b);
    CHECK(b.material[i] == 0);
  }
}

TEST_CASE("Binary STL whose header starts with 'solid' is detected by size", "[stl]") {
  TriangleSoup s = primitives::box({0, 0, 0}, {1, 1, 1});
  const fs::path p = temp_dir() / "solid_header.stl";
  write_stl_binary(p, s);
  {
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    f.write("solid trap", 10);
  }
  const TriangleSoup r = read_stl(p);
  CHECK(r.format == StlFormat::Binary);
  CHECK(r.triangles.size() == 12);
}

TEST_CASE("Malformed STL files are rejected with a location", "[stl]") {
  expect_stl_error(temp_dir() / "missing.stl", "cannot open");
  expect_stl_error(write_text("empty.stl", ""), "not a valid STL");
  expect_stl_error(write_text("garbage.stl", "hello world\n"), "not a valid STL");
  expect_stl_error(write_text("no_tris.stl", "solid x\nendsolid x\n"), "contains no triangles");
  {
    const std::string t = kOneTriangle;
    expect_stl_error(write_text("no_end.stl", t.substr(0, t.find("endsolid"))), "unexpected end of file");
  }
  {
    std::string t = kOneTriangle;
    t.replace(t.find("vertex 1 0 0"), 12, "vertex 1 x 0");
    expect_stl_error(write_text("bad_number.stl", t), "bad_number.stl:5: invalid number 'x'");
  }
  {
    std::string t = kOneTriangle;
    t.replace(t.find("vertex 1 0 0"), 12, "vertex 1 nan 0");
    expect_stl_error(write_text("nan.stl", t), "invalid number 'nan'");
  }
  {
    std::string t = kOneTriangle;
    t.replace(t.find("endloop"), 7, "endlop");
    expect_stl_error(write_text("typo.stl", t), "expected 'endloop'");
  }
  {
    std::string t = kOneTriangle;
    t.replace(t.find("      vertex 0 1 0\n"), 19, "");
    expect_stl_error(write_text("two_vertices.stl", t), "expected 'vertex x y z'");
  }
  {
    // Binary header announcing more triangles than present.
    TriangleSoup s = primitives::box({0, 0, 0}, {1, 1, 1});
    const fs::path p = temp_dir() / "truncated.stl";
    write_stl_binary(p, s);
    fs::resize_file(p, fs::file_size(p) - 10);
    expect_stl_error(p, "binary size mismatch");
  }
}
