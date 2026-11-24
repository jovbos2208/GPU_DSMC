# Optional HDF5 (C library) for the DSMC cell-state output (dsmc.dt_DSMCState_out,
# VTKHDF files). VLEO_HDF5:
#   AUTO  use HDF5 if find_package finds it (default)
#   ON    require HDF5
#   OFF   build without HDF5; configurations that request HDF5 output are rejected
set(VLEO_HDF5 "AUTO" CACHE STRING "HDF5 output support: AUTO, ON or OFF")
set_property(CACHE VLEO_HDF5 PROPERTY STRINGS AUTO ON OFF)

set(VLEO_HAVE_HDF5 OFF)
if(NOT VLEO_HDF5 STREQUAL "OFF")
  if(VLEO_HDF5 STREQUAL "ON")
    find_package(HDF5 REQUIRED COMPONENTS C)
  else()
    find_package(HDF5 QUIET COMPONENTS C)
  endif()
  if(HDF5_FOUND)
    set(VLEO_HAVE_HDF5 ON)
    if(NOT TARGET hdf5::hdf5)
      add_library(hdf5::hdf5 INTERFACE IMPORTED)
      target_include_directories(hdf5::hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
      target_link_libraries(hdf5::hdf5 INTERFACE ${HDF5_LIBRARIES})
      target_compile_definitions(hdf5::hdf5 INTERFACE ${HDF5_DEFINITIONS})
    endif()
  endif()
endif()
message(STATUS "VLEO: HDF5 output ${VLEO_HAVE_HDF5} (VLEO_HDF5=${VLEO_HDF5}, HDF5 ${HDF5_VERSION})")
