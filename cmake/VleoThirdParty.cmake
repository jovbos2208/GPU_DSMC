# Pinned third-party dependencies. The exact versions and upstream commits are
# recorded in third_party/VERSIONS.md and docs/dependencies.md. Do not update a
# dependency without updating both files and re-running the full test suite.

set(_vleo_tp "${PROJECT_SOURCE_DIR}/third_party")

foreach(_dep kokkos yaml-cpp catch2)
  if(NOT EXISTS "${_vleo_tp}/${_dep}/CMakeLists.txt")
    message(FATAL_ERROR "Missing vendored dependency third_party/${_dep}. "
                        "See docs/dependencies.md for how to restore it.")
  endif()
endforeach()

# --- Kokkos -----------------------------------------------------------------
# Backend selection (Kokkos_ENABLE_SERIAL/OPENMP/CUDA/HIP, Kokkos_ARCH_*) is
# done by the caller, normally through CMakePresets.json.
add_subdirectory("${_vleo_tp}/kokkos" "${PROJECT_BINARY_DIR}/third_party/kokkos" SYSTEM)

# --- yaml-cpp ---------------------------------------------------------------
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${_vleo_tp}/yaml-cpp" "${PROJECT_BINARY_DIR}/third_party/yaml-cpp" EXCLUDE_FROM_ALL SYSTEM)

# --- Catch2 (tests only) ----------------------------------------------------
if(VLEO_BUILD_TESTS)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${_vleo_tp}/catch2" "${PROJECT_BINARY_DIR}/third_party/catch2" EXCLUDE_FROM_ALL SYSTEM)
endif()
