#include "vleo/io/VtkHdf.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef VLEO_HAVE_HDF5
#include <hdf5.h>
#endif

namespace vleo {

#ifndef VLEO_HAVE_HDF5

bool vtkhdf_available() { return false; }
std::string hdf5_version() { return {}; }
void write_vtkhdf_hex_grid(const std::filesystem::path&, const VtkHexGrid&) {
  throw std::runtime_error("HDF5 output is not available in this build (configure with -DVLEO_HDF5=ON)");
}

#else

namespace {

constexpr hsize_t kChunkRows = 1u << 16;
constexpr unsigned kDeflateLevel = 4;

// Closes an HDF5 identifier on scope exit.
class Handle {
 public:
  Handle(hid_t id, herr_t (*close)(hid_t), const std::string& what) : id_(id), close_(close) {
    if (id_ < 0) throw std::runtime_error("HDF5: cannot " + what);
  }
  ~Handle() {
    if (id_ >= 0) close_(id_);
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  hid_t get() const { return id_; }

 private:
  hid_t id_;
  herr_t (*close_)(hid_t);
};

void check(herr_t status, const std::string& what) {
  if (status < 0) throw std::runtime_error("HDF5: cannot " + what);
}

bool deflate_available() {
  unsigned config = 0;
  return H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0 && H5Zget_filter_info(H5Z_FILTER_DEFLATE, &config) >= 0 &&
         (config & H5Z_FILTER_CONFIG_ENCODE_ENABLED) != 0;
}

// Dataset of rank 1 (cols == 0) or 2 ([rows, cols]); chunked and compressed if large.
void write_dataset(hid_t parent, const std::string& name, hid_t mem_type, hid_t file_type, const void* data,
                   hsize_t rows, hsize_t cols = 0) {
  const int rank = cols == 0 ? 1 : 2;
  const hsize_t dims[2] = {rows, cols};
  Handle space(H5Screate_simple(rank, dims, nullptr), H5Sclose, "create dataspace for " + name);
  Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "create property list for " + name);
  if (rows > kChunkRows && deflate_available()) {
    const hsize_t chunk[2] = {kChunkRows, cols};
    check(H5Pset_chunk(dcpl.get(), rank, chunk), "set chunking for " + name);
    check(H5Pset_deflate(dcpl.get(), kDeflateLevel), "set compression for " + name);
  }
  Handle set(H5Dcreate2(parent, name.c_str(), file_type, space.get(), H5P_DEFAULT, dcpl.get(), H5P_DEFAULT), H5Dclose,
             "create dataset " + name);
  if (rows > 0) check(H5Dwrite(set.get(), mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "write dataset " + name);
}

void write_scalar_i64(hid_t parent, const std::string& name, std::int64_t v) {
  write_dataset(parent, name, H5T_NATIVE_INT64, H5T_STD_I64LE, &v, 1);
}

void write_string_attribute(hid_t obj, const std::string& name, const std::string& value) {
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "create string type");
  check(H5Tset_size(type.get(), value.size()), "set string size");
  check(H5Tset_strpad(type.get(), H5T_STR_NULLPAD), "set string padding");
  Handle space(H5Screate(H5S_SCALAR), H5Sclose, "create scalar dataspace");
  Handle attr(H5Acreate2(obj, name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose,
              "create attribute " + name);
  check(H5Awrite(attr.get(), type.get(), value.data()), "write attribute " + name);
}

void write_version_attribute(hid_t obj) {
  const std::int64_t version[2] = {1, 0};
  const hsize_t dims[1] = {2};
  Handle space(H5Screate_simple(1, dims, nullptr), H5Sclose, "create dataspace for Version");
  Handle attr(H5Acreate2(obj, "Version", H5T_STD_I64LE, space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose,
              "create attribute Version");
  check(H5Awrite(attr.get(), H5T_NATIVE_INT64, version), "write attribute Version");
}

}  // namespace

bool vtkhdf_available() { return true; }

std::string hdf5_version() {
  unsigned maj = 0, min = 0, rel = 0;
  if (H5get_libversion(&maj, &min, &rel) < 0) return "unknown";
  return std::to_string(maj) + "." + std::to_string(min) + "." + std::to_string(rel);
}

void write_vtkhdf_hex_grid(const std::filesystem::path& path, const VtkHexGrid& g) {
  if (g.nx < 1 || g.ny < 1 || g.nz < 1) throw std::invalid_argument("VTKHDF: grid must have at least one cell");
  const std::int64_t nc_total = static_cast<std::int64_t>(g.nx) * g.ny * g.nz;
  const auto nc = static_cast<std::int64_t>(g.cells.size());
  for (std::size_t k = 0; k < g.cells.size(); ++k)
    if (g.cells[k] < 0 || g.cells[k] >= nc_total || (k > 0 && g.cells[k] <= g.cells[k - 1]))
      throw std::invalid_argument("VTKHDF: cell indices must be ascending and inside the grid");
  for (const auto& f : g.cell_fields)
    if ((f.components != 1 && f.components != 3) ||
        f.values.size() != static_cast<std::size_t>(nc) * static_cast<std::size_t>(f.components))
      throw std::invalid_argument("VTKHDF: field '" + f.name + "' has the wrong size");

  // Lattice points, x fastest.
  const std::int64_t px = g.nx + 1, py = g.ny + 1, pz = g.nz + 1;
  const std::int64_t np = px * py * pz;
  std::vector<double> points(static_cast<std::size_t>(3 * np));
  for (std::int64_t k = 0; k < pz; ++k)
    for (std::int64_t j = 0; j < py; ++j)
      for (std::int64_t i = 0; i < px; ++i) {
        const auto p = static_cast<std::size_t>(3 * (i + px * (j + py * k)));
        points[p + 0] = g.origin.x + static_cast<double>(i) * g.spacing.x;
        points[p + 1] = g.origin.y + static_cast<double>(j) * g.spacing.y;
        points[p + 2] = g.origin.z + static_cast<double>(k) * g.spacing.z;
      }

  // Hexahedra in VTK order: bottom face (z = k) counter-clockwise, then top face.
  std::vector<std::int64_t> conn(static_cast<std::size_t>(8 * nc));
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(nc + 1));
  std::vector<std::uint8_t> types(static_cast<std::size_t>(nc), 12);
  const auto pid = [&](std::int64_t i, std::int64_t j, std::int64_t k) { return i + px * (j + py * k); };
  for (std::int64_t e = 0; e < nc; ++e) {
    const std::int64_t c = g.cells[static_cast<std::size_t>(e)];
    const std::int64_t i = c % g.nx, j = (c / g.nx) % g.ny, k = c / (static_cast<std::int64_t>(g.nx) * g.ny);
    const std::int64_t ids[8] = {
        pid(i, j, k),     pid(i + 1, j, k),     pid(i + 1, j + 1, k),     pid(i, j + 1, k),
        pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1), pid(i, j + 1, k + 1)};
    std::copy(ids, ids + 8, conn.begin() + 8 * e);
    offsets[static_cast<std::size_t>(e)] = 8 * e;
  }
  offsets[static_cast<std::size_t>(nc)] = 8 * nc;

  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);  // errors are reported as exceptions
  const std::filesystem::path tmp = path.string() + ".part";
  {
    Handle file(H5Fcreate(tmp.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose,
                "create file " + tmp.string());
    Handle root(H5Gcreate2(file.get(), "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                "create group VTKHDF");
    write_version_attribute(root.get());
    write_string_attribute(root.get(), "Type", "UnstructuredGrid");
    write_scalar_i64(root.get(), "NumberOfPoints", np);
    write_scalar_i64(root.get(), "NumberOfCells", nc);
    write_scalar_i64(root.get(), "NumberOfConnectivityIds", 8 * nc);
    write_dataset(root.get(), "Points", H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, points.data(), static_cast<hsize_t>(np), 3);
    write_dataset(root.get(), "Connectivity", H5T_NATIVE_INT64, H5T_STD_I64LE, conn.data(),
                  static_cast<hsize_t>(conn.size()));
    write_dataset(root.get(), "Offsets", H5T_NATIVE_INT64, H5T_STD_I64LE, offsets.data(),
                  static_cast<hsize_t>(offsets.size()));
    write_dataset(root.get(), "Types", H5T_NATIVE_UINT8, H5T_STD_U8LE, types.data(),
                  static_cast<hsize_t>(types.size()));
    {
      Handle cd(H5Gcreate2(root.get(), "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                "create group CellData");
      for (const auto& f : g.cell_fields)
        write_dataset(cd.get(), f.name, H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, f.values.data(), static_cast<hsize_t>(nc),
                      f.components == 1 ? 0 : static_cast<hsize_t>(f.components));
    }
    if (!g.field_data.empty()) {
      Handle fd(H5Gcreate2(root.get(), "FieldData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                "create group FieldData");
      for (const auto& [name, value] : g.field_data)
        write_dataset(fd.get(), name, H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, &value, 1);
    }
    check(H5Fflush(file.get(), H5F_SCOPE_GLOBAL), "flush file");
  }
  std::filesystem::rename(tmp, path);
}

#endif  // VLEO_HAVE_HDF5

}  // namespace vleo
