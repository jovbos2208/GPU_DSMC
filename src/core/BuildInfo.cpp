#include "vleo/core/BuildInfo.hpp"

#include <sstream>

#include <Kokkos_Core.hpp>

namespace vleo {

namespace {

std::string compiler_detail_string() {
  std::ostringstream os;
#if defined(__clang__)
  os << "clang " << __clang_version__;
#elif defined(__GNUC__)
  os << "gcc " << __VERSION__;
#elif defined(_MSC_VER)
  os << "msvc " << _MSC_FULL_VER;
#else
  os << "unknown compiler";
#endif
#if defined(__CUDACC_VER_MAJOR__)
  os << "; nvcc " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__;
#endif
#if defined(__HIP_PLATFORM_AMD__) && defined(HIP_VERSION_MAJOR)
  os << "; hip " << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "." << HIP_VERSION_PATCH;
#endif
  return os.str();
}

std::string kokkos_version_string() {
  std::ostringstream os;
  os << KOKKOS_VERSION_MAJOR << "." << KOKKOS_VERSION_MINOR << "." << KOKKOS_VERSION_PATCH;
  return os.str();
}

std::vector<std::string> enabled_backends() {
  std::vector<std::string> b;
#if defined(KOKKOS_ENABLE_SERIAL)
  b.emplace_back("Serial");
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  b.emplace_back("OpenMP");
#endif
#if defined(KOKKOS_ENABLE_THREADS)
  b.emplace_back("Threads");
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  b.emplace_back("Cuda");
#endif
#if defined(KOKKOS_ENABLE_HIP)
  b.emplace_back("HIP");
#endif
#if defined(KOKKOS_ENABLE_SYCL)
  b.emplace_back("SYCL");
#endif
  return b;
}

BuildInfo make_build_info() {
  BuildInfo info{
      detail::git_info_generated(),
      detail::configure_info_generated(),
      compiler_detail_string(),
      kokkos_version_string(),
      Kokkos::DefaultExecutionSpace::name(),
      Kokkos::DefaultHostExecutionSpace::name(),
      enabled_backends(),
      static_cast<long>(__cplusplus),
  };
  return info;
}

}  // namespace

const BuildInfo& build_info() {
  static const BuildInfo info = make_build_info();
  return info;
}

std::string build_info_string() {
  const BuildInfo& b = build_info();
  std::ostringstream os;
  os << "vleo_solver " << b.configure.solver_version << "\n"
     << "  git commit       : " << b.git.commit << (b.git.dirty ? " (dirty)" : "") << "\n"
     << "  git describe     : " << b.git.describe << "\n"
     << "  build type       : " << b.configure.build_type << "\n"
     << "  compiler         : " << b.configure.compiler_id << " " << b.configure.compiler_version << " ("
     << b.compiler_detail << ")\n"
     << "  C++ standard     : " << b.cxx_standard << "\n"
     << "  Kokkos version   : " << b.kokkos_version << "\n"
     << "  backend          : " << b.configure.backend << "\n"
     << "  execution space  : " << b.default_execution_space << " (host: " << b.default_host_execution_space
     << ")\n"
     << "  Kokkos backends  : ";
  for (std::size_t i = 0; i < b.enabled_kokkos_backends.size(); ++i)
    os << (i ? ", " : "") << b.enabled_kokkos_backends[i];
  os << "\n"
     << "  Kokkos arch      : " << b.configure.kokkos_arch << "\n";
  return os.str();
}

}  // namespace vleo
