// VTKHDF output of hexahedral grids (DSMC cell states, dsmc.dt_DSMCState_out):
// the file layout follows the VTKHDF 1.0 UnstructuredGrid specification and
// every value is read back exactly with the HDF5 C API.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "vleo/io/VtkHdf.hpp"

#ifdef VLEO_HAVE_HDF5
#include <hdf5.h>
#endif

using namespace vleo;
namespace fs = std::filesystem;

namespace {

VtkHexGrid small_grid() {
  // 3 x 2 x 2 cells of 0.5 x 1 x 2 m; cells 0, 4, 11 written
  VtkHexGrid g;
  g.origin = Vec3(-1.0, 0.0, 2.0);
  g.spacing = Vec3(0.5, 1.0, 2.0);
  g.nx = 3;
  g.ny = 2;
  g.nz = 2;
  g.cells = {0, 4, 11};
  g.cell_fields.push_back({"density", 1, {1.0, 2.5, -3.0}});
  g.cell_fields.push_back({"velocity", 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}});
  g.field_data = {{"TimeValue", 1.25e-4}};
  return g;
}

}  // namespace

#ifdef VLEO_HAVE_HDF5

namespace {

template <class T>
std::vector<T> read_all(hid_t file, const std::string& name, hid_t type) {
  const hid_t set = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  REQUIRE(set >= 0);
  const hid_t space = H5Dget_space(set);
  const hssize_t n = H5Sget_simple_extent_npoints(space);
  std::vector<T> v(static_cast<std::size_t>(n));
  REQUIRE(H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) >= 0);
  H5Sclose(space);
  H5Dclose(set);
  return v;
}

std::vector<hsize_t> dims_of(hid_t file, const std::string& name) {
  const hid_t set = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  const hid_t space = H5Dget_space(set);
  std::vector<hsize_t> d(static_cast<std::size_t>(H5Sget_simple_extent_ndims(space)));
  H5Sget_simple_extent_dims(space, d.data(), nullptr);
  H5Sclose(space);
  H5Dclose(set);
  return d;
}

}  // namespace

TEST_CASE("VTKHDF hexahedral grid: layout and values", "[io]") {
  REQUIRE(vtkhdf_available());
  CHECK(!hdf5_version().empty());
  const fs::path dir = fs::temp_directory_path() / "vleo_test_vtkhdf";
  fs::create_directories(dir);
  const fs::path path = dir / "grid.vtkhdf";
  write_vtkhdf_hex_grid(path, small_grid());
  CHECK(!fs::exists(path.string() + ".part"));

  const hid_t f = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(f >= 0);
  {
    const hid_t g = H5Gopen2(f, "VTKHDF", H5P_DEFAULT);
    REQUIRE(g >= 0);
    std::int64_t version[2] = {0, 0};
    const hid_t av = H5Aopen(g, "Version", H5P_DEFAULT);
    REQUIRE(H5Aread(av, H5T_NATIVE_INT64, version) >= 0);
    H5Aclose(av);
    CHECK(version[0] == 1);
    CHECK(version[1] == 0);
    const hid_t at = H5Aopen(g, "Type", H5P_DEFAULT);
    const hid_t tt = H5Aget_type(at);
    CHECK(H5Tget_class(tt) == H5T_STRING);
    std::string type(H5Tget_size(tt), '\0');
    REQUIRE(H5Aread(at, tt, type.data()) >= 0);
    H5Tclose(tt);
    H5Aclose(at);
    CHECK(type == "UnstructuredGrid");
    H5Gclose(g);
  }
  CHECK(read_all<std::int64_t>(f, "VTKHDF/NumberOfPoints", H5T_NATIVE_INT64) == std::vector<std::int64_t>{4 * 3 * 3});
  CHECK(read_all<std::int64_t>(f, "VTKHDF/NumberOfCells", H5T_NATIVE_INT64) == std::vector<std::int64_t>{3});
  CHECK(read_all<std::int64_t>(f, "VTKHDF/NumberOfConnectivityIds", H5T_NATIVE_INT64) == std::vector<std::int64_t>{24});
  CHECK(dims_of(f, "VTKHDF/Points") == std::vector<hsize_t>{36, 3});
  const auto pts = read_all<double>(f, "VTKHDF/Points", H5T_NATIVE_DOUBLE);
  // point (i, j, k) = (1, 2, 1) -> index 1 + 4 (2 + 3 * 1) = 21
  CHECK(pts[3 * 21 + 0] == -0.5);
  CHECK(pts[3 * 21 + 1] == 2.0);
  CHECK(pts[3 * 21 + 2] == 4.0);
  CHECK(read_all<std::uint8_t>(f, "VTKHDF/Types", H5T_NATIVE_UINT8) == std::vector<std::uint8_t>{12, 12, 12});
  CHECK(read_all<std::int64_t>(f, "VTKHDF/Offsets", H5T_NATIVE_INT64) == std::vector<std::int64_t>{0, 8, 16, 24});
  const auto conn = read_all<std::int64_t>(f, "VTKHDF/Connectivity", H5T_NATIVE_INT64);
  // cell 4 = (ix, iy, iz) = (1, 1, 0): VTK_HEXAHEDRON corner order
  const std::vector<std::int64_t> cell4(conn.begin() + 8, conn.begin() + 16);
  CHECK(cell4 == std::vector<std::int64_t>{5, 6, 10, 9, 17, 18, 22, 21});
  // cell 11 = (2, 1, 1)
  CHECK(conn[16] == 2 + 4 * (1 + 3 * 1));
  CHECK(read_all<double>(f, "VTKHDF/CellData/density", H5T_NATIVE_DOUBLE) == std::vector<double>{1.0, 2.5, -3.0});
  CHECK(dims_of(f, "VTKHDF/CellData/velocity") == std::vector<hsize_t>{3, 3});
  CHECK(read_all<double>(f, "VTKHDF/CellData/velocity", H5T_NATIVE_DOUBLE) ==
        std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  CHECK(read_all<double>(f, "VTKHDF/FieldData/TimeValue", H5T_NATIVE_DOUBLE) == std::vector<double>{1.25e-4});
  H5Fclose(f);
  fs::remove(path);
}

#else

TEST_CASE("VTKHDF output is reported as unavailable without HDF5", "[io]") {
  CHECK(!vtkhdf_available());
  CHECK(hdf5_version().empty());
  CHECK_THROWS(write_vtkhdf_hex_grid(fs::temp_directory_path() / "vleo_no_hdf5.vtkhdf", small_grid()));
}

#endif

TEST_CASE("VTKHDF input validation", "[io]") {
  VtkHexGrid g = small_grid();
  g.cells = {4, 0, 11};  // not ascending
  CHECK_THROWS(write_vtkhdf_hex_grid(fs::temp_directory_path() / "vleo_bad.vtkhdf", g));
  g = small_grid();
  g.cells.push_back(12);  // outside the grid (and field sizes no longer match)
  CHECK_THROWS(write_vtkhdf_hex_grid(fs::temp_directory_path() / "vleo_bad.vtkhdf", g));
  g = small_grid();
  g.cell_fields[0].values.pop_back();
  CHECK_THROWS(write_vtkhdf_hex_grid(fs::temp_directory_path() / "vleo_bad.vtkhdf", g));
}
