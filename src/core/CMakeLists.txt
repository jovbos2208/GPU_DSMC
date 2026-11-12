# vleo_core: fundamental types (Vec3, Mat3, frames), physical constants,
# species data, the counter-based RNG and build/provenance information.

# --- Configure-time build information --------------------------------------
if(CMAKE_CONFIGURATION_TYPES)
  set(VLEO_BUILD_TYPE_STRING "multi-config")
else()
  set(VLEO_BUILD_TYPE_STRING "${CMAKE_BUILD_TYPE}")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt_upper)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt_upper}}" VLEO_CXX_FLAGS_STRING)

set(_archs "")
get_cmake_property(_cache_vars CACHE_VARIABLES)
foreach(_v IN LISTS _cache_vars)
  if(_v MATCHES "^Kokkos_ARCH_(.+)$" AND ${_v})
    list(APPEND _archs "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(_archs)
  list(JOIN _archs "," VLEO_KOKKOS_ARCH_STRING)
else()
  set(VLEO_KOKKOS_ARCH_STRING "none")
endif()

set(_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file("${PROJECT_SOURCE_DIR}/cmake/BuildConfig.cpp.in" "${_gen_dir}/BuildConfig.cpp" @ONLY)

# --- Build-time git information --------------------------------------------
set(_git_args
  -DVLEO_SOURCE_DIR=${PROJECT_SOURCE_DIR}
  -DVLEO_GIT_TEMPLATE=${PROJECT_SOURCE_DIR}/cmake/GitInfo.cpp.in
  -DVLEO_GIT_OUTPUT=${_gen_dir}/GitInfo.cpp)
# Run once at configure time so the file exists, then on every build.
execute_process(COMMAND ${CMAKE_COMMAND} ${_git_args} -P ${PROJECT_SOURCE_DIR}/cmake/VleoGitInfo.cmake)
add_custom_target(vleo_git_info
  COMMAND ${CMAKE_COMMAND} ${_git_args} -P ${PROJECT_SOURCE_DIR}/cmake/VleoGitInfo.cmake
  BYPRODUCTS "${_gen_dir}/GitInfo.cpp"
  COMMENT "Updating git provenance information"
  VERBATIM)

add_library(vleo_core STATIC
  BuildInfo.cpp
  SelfTest.cpp
  Species.cpp
  "${_gen_dir}/BuildConfig.cpp"
  "${_gen_dir}/GitInfo.cpp")
add_dependencies(vleo_core vleo_git_info)
target_include_directories(vleo_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(vleo_core PUBLIC Kokkos::kokkos PRIVATE vleo_warnings)
target_compile_features(vleo_core PUBLIC cxx_std_20)
