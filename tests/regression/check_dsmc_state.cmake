# Verifies the DSMC cell-state output of tests/regression/dsmc_state.yaml:
# 8 steps of 1e-6 s with dt_DSMCState_out = 2.5e-6 s -> windows ending at steps
# 3, 5 and 8 (the last one ends with the run), listed in the ParaView series file.
# Usage: cmake -DOUT_DIR=<dir> -P check_dsmc_state.cmake
set(_dir "${OUT_DIR}/dsmc_state")
file(READ "${_dir}/dsmc_state.vtkhdf.series" J)
string(JSON n LENGTH "${J}" files)
if(NOT n EQUAL 3)
  message(FATAL_ERROR "dsmc_state: expected 3 cell-state files, the series lists ${n}")
endif()
math(EXPR last "${n} - 1")
foreach(i RANGE ${last})
  string(JSON name GET "${J}" files ${i} name)
  string(JSON t GET "${J}" files ${i} time)
  if(NOT EXISTS "${_dir}/${name}")
    message(FATAL_ERROR "dsmc_state: ${name} is listed but missing")
  endif()
  file(READ "${_dir}/${name}" head LIMIT 8 HEX)
  if(NOT head STREQUAL "894844460d0a1a0a")
    message(FATAL_ERROR "dsmc_state: ${name} is not an HDF5 file")
  endif()
  message(STATUS "${name}: t = ${t} s")
endforeach()
string(JSON t0 GET "${J}" files 0 time)
string(JSON t2 GET "${J}" files 2 time)
# times are written with round-trip precision, e.g. 3.0000000000000001e-06
if(NOT t0 MATCHES "^(3(\\.0*[0-9]?)?|2\\.9999+[0-9]*)e-0?6$" OR NOT t2 MATCHES "^(8(\\.0*[0-9]?)?|7\\.9999+[0-9]*)e-0?6$")
  message(FATAL_ERROR "dsmc_state: unexpected window end times ${t0}, ${t2}")
endif()
