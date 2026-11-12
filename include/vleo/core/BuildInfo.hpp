#pragma once

// Build provenance: git state, compiler, build type, Kokkos version and
// backend. Written into every run's metadata and printed by `--version`.

#include <string>
#include <string_view>
#include <vector>

namespace vleo {

struct GitInfo {
  bool available;  // false if the source tree was not a git checkout at build time
  std::string_view commit;
  std::string_view describe;
  bool dirty;  // tracked files had uncommitted changes at build time
};

struct ConfigureInfo {
  std::string_view solver_version;
  std::string_view build_type;
  std::string_view compiler_id;
  std::string_view compiler_version;
  std::string_view compiler_path;
  std::string_view cxx_flags;
  std::string_view backend;      // cpu | cuda | hip
  std::string_view kokkos_arch;  // enabled Kokkos_ARCH_* options, comma separated
  std::string_view cmake_version;
  std::string_view system;
};

struct BuildInfo {
  GitInfo git;
  ConfigureInfo configure;
  std::string compiler_detail;          // compiler-reported version string(s), incl. CUDA/HIP
  std::string kokkos_version;           // e.g. "5.2.2"
  std::string default_execution_space;  // e.g. "Serial", "OpenMP", "Cuda", "HIP"
  std::string default_host_execution_space;
  std::vector<std::string> enabled_kokkos_backends;
  long cxx_standard;  // value of __cplusplus
};

const BuildInfo& build_info();

// Multi-line human-readable summary (used by --version).
std::string build_info_string();

namespace detail {
const GitInfo& git_info_generated();
const ConfigureInfo& configure_info_generated();
}  // namespace detail

}  // namespace vleo
