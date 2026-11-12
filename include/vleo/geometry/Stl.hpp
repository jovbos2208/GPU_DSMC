#pragma once

// STL import/export (host side).
//
// Both binary and ASCII STL are supported. The format is detected from the
// file size (binary: 84 + 50 n bytes) and, failing that, from the leading
// "solid" keyword. Facet normals stored in the file are kept only for
// diagnostics; the solver derives normals from the vertex order
// (docs/conventions.md, section 3).
//
// Material ids: in ASCII files every `solid ... endsolid` block gets its own
// material id (0, 1, ... in file order); binary files have a single solid and
// all triangles get material id 0.

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "vleo/geometry/Triangle.hpp"

namespace vleo {

class StlError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class StlFormat { Ascii, Binary };

struct TriangleSoup {
  std::vector<Triangle> triangles;      // in metres after scaling
  std::vector<std::uint32_t> material;  // one per triangle
  std::vector<Vec3> stored_normals;     // facet normals from the file (unscaled, may be zero)
  std::vector<std::string> solid_names; // index = material id
  StlFormat format = StlFormat::Ascii;
};

// Reads an STL file; every coordinate is multiplied by `scale_to_m`.
// Throws StlError on malformed input (with line or triangle number).
TriangleSoup read_stl(const std::filesystem::path& file, double scale_to_m = 1.0);

// Writers (used by tests and the example-geometry generator). Coordinates are
// written as given. The binary writer stores single precision; the ASCII
// writer uses round-trip (shortest) decimal representation.
void write_stl_ascii(const std::filesystem::path& file, const TriangleSoup& soup);
void write_stl_binary(const std::filesystem::path& file, const TriangleSoup& soup);

}  // namespace vleo
