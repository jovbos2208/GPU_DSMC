cmake_minimum_required(VERSION 3.27)

project(VLEOParticleSolver
  VERSION 0.9.0
  DESCRIPTION "GPU TPMC/DSMC solver with Maxwell and CLL gas-surface interaction"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(VLEO_BUILD_TESTS "Build the VLEO test suite" ON)
option(VLEO_ENABLE_WARNINGS "Enable compiler warnings for project targets" ON)
option(VLEO_WARNINGS_AS_ERRORS "Treat compiler warnings as errors for project targets" OFF)

# ---------------------------------------------------------------------------
# Third-party dependencies (pinned, vendored copies; see docs/dependencies.md)
# ---------------------------------------------------------------------------
include(VleoThirdParty)

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------
include(VleoWarnings)
include(VleoBackend)   # derives VLEO_BACKEND (cpu | cuda | hip) from Kokkos

if(VLEO_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(src/core)
add_subdirectory(src/geometry)
add_subdirectory(src/particles)
add_subdirectory(src/gsi)
add_subdirectory(src/tpmc)
add_subdirectory(src/dsmc)
add_subdirectory(src/tally)
add_subdirectory(src/io)

add_executable(vleo_solver src/main.cpp)
target_link_libraries(vleo_solver PRIVATE
  vleo_core
  vleo_geometry
  vleo_particles
  vleo_gsi
  vleo_tpmc
  vleo_dsmc
  vleo_tally
  vleo_io
  vleo_warnings
  Kokkos::kokkos)

if(VLEO_BUILD_TESTS)
  add_subdirectory(tests)
endif()

message(STATUS "VLEO: backend=${VLEO_BACKEND}  build type=${CMAKE_BUILD_TYPE}  "
               "compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
