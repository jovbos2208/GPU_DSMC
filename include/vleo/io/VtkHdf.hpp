#pragma once

// VTKHDF output of a Cartesian grid as an unstructured grid of hexahedra.
//
// The file is an HDF5 file in the VTKHDF layout (VTK file format
// specification, "VTKHDF File Format", version 1.0), Type UnstructuredGrid -
// the same data model as a .vtu file. ParaView >= 5.11 opens it directly
// (extension .vtkhdf); scripts/vtkhdf_to_vtu.py converts it to a .vtu file.
//
//   /VTKHDF                      attributes Version = [1, 0], Type = "UnstructuredGrid"
//     NumberOfPoints   [1]       int64
//     NumberOfCells    [1]       int64
//     NumberOfConnectivityIds [1] int64
//     Points           [P, 3]    float64  lattice points of the grid
//     Connectivity     [8 C]     int64    VTK_HEXAHEDRON point order
//     Offsets          [C + 1]   int64
//     Types            [C]       uint8    12 (VTK_HEXAHEDRON)
//     CellData/<name>  [C] or [C, k] float64
//     FieldData/<name> [1]       float64
//
// Only the listed cells are written (e.g. the gas cells of a DSMC grid); all
// (nx+1)(ny+1)(nz+1) lattice points are written. Large datasets are chunked
// and deflate-compressed when the HDF5 library provides the filter.

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "vleo/core/Vec3.hpp"

namespace vleo {

struct VtkCellField {
  std::string name;
  int components = 1;          // 1 (scalar) or 3 (vector)
  std::vector<double> values;  // components * number of written cells, cell-major
};

struct VtkHexGrid {
  Vec3 origin;
  Vec3 spacing;
  int nx = 1, ny = 1, nz = 1;
  std::vector<std::int64_t> cells;  // written cells, index c = ix + nx (iy + ny iz), ascending
  std::vector<VtkCellField> cell_fields;
  std::vector<std::pair<std::string, double>> field_data;  // e.g. {"TimeValue", t}
};

// True if this build can write HDF5 files (CMake option VLEO_HDF5).
bool vtkhdf_available();

// HDF5 library version of this build ("1.10.10"), empty without HDF5.
std::string hdf5_version();

// Writes `grid` to `path` (overwritten). Throws std::runtime_error on failure
// or if HDF5 support is not compiled in.
void write_vtkhdf_hex_grid(const std::filesystem::path& path, const VtkHexGrid& grid);

}  // namespace vleo
