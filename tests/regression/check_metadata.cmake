# Verifies that vleo_solver wrote complete run metadata (Gate M0).
# Usage: cmake -DOUT_DIR=<dir> -DEXPECTED_BACKEND=<cpu|cuda|hip> -P check_metadata.cmake

foreach(f run_metadata.json resolved_config.yaml input_config.yaml kokkos_configuration.txt)
  if(NOT EXISTS "${OUT_DIR}/${f}")
    message(FATAL_ERROR "missing output file ${OUT_DIR}/${f}")
  endif()
endforeach()

file(READ "${OUT_DIR}/run_metadata.json" J)

function(require_nonempty)
  string(JSON v ERROR_VARIABLE err GET "${J}" ${ARGN})
  if(err)
    message(FATAL_ERROR "metadata: missing ${ARGN}: ${err}")
  endif()
  if("${v}" STREQUAL "")
    message(FATAL_ERROR "metadata: empty value for ${ARGN}")
  endif()
endfunction()

function(require_equal expected)
  string(JSON v ERROR_VARIABLE err GET "${J}" ${ARGN})
  if(err OR NOT "${v}" STREQUAL "${expected}")
    message(FATAL_ERROR "metadata: ${ARGN} = '${v}', expected '${expected}' ${err}")
  endif()
endfunction()

require_equal("vleo-run-metadata/1" schema)
require_equal("configured" status)
require_equal("tpmc" simulation mode)
require_equal("cll" simulation gsi_model)
require_equal("${EXPECTED_BACKEND}" simulation backend)
require_equal("${EXPECTED_BACKEND}" build backend)
require_equal("${EXPECTED_BACKEND}" platform backend)

foreach(path
    "build;solver_version" "build;git;commit" "build;git;describe" "build;git;available" "build;git;dirty"
    "build;build_type" "build;compiler;id" "build;compiler;version" "build;compiler;detail"
    "build;compiler;cxx_standard" "build;kokkos;version" "build;kokkos;default_execution_space"
    "build;kokkos;enabled_backends" "build;kokkos;arch" "build;cmake_version"
    "runtime;timestamp_utc" "runtime;command_line" "runtime;working_directory" "runtime;execution_space"
    "runtime;concurrency" "input;config_file" "simulation;random_seed"
    "resolved_config;simulation;name" "resolved_config;geometry;file" "resolved_config;reference;area_m2"
    "resolved_config;reference;moment_center_m" "resolved_config;gas;species;O"
    "resolved_config;surface;cll;alpha_n" "resolved_config;surface;cll;sigma_t"
    "resolved_config;tpmc;particle_histories" "resolved_config;dsmc;particles_per_cell_target"
    "resolved_config;statistics;confidence_level" "resolved_config;output;format"
    "derived;dynamic_pressure_Pa" "derived;mass_density_kg_m3" "derived;aero_axes_body;drag_e_D"
    "derived;aero_axes_body;lift_e_L" "derived;aero_axes_body;side_e_Y" "output_files" "notes")
  require_nonempty(${path})
endforeach()

string(JSON notes GET "${J}" notes)
if(NOT notes MATCHES "self-test.*\\(passed\\)")
  message(FATAL_ERROR "metadata: backend self-test did not pass: ${notes}")
endif()


# M1: geometry, domain and inflow sections
foreach(path
    "geometry;triangles" "geometry;vertices" "geometry;closed" "geometry;total_area_m2" "geometry;bbox_min_m"
    "geometry;components" "geometry;materials" "geometry;bvh;nodes" "geometry;bvh;max_depth"
    "geometry;surface_offset_m" "inflow;domain_min_m" "inflow;domain_max_m" "inflow;total_rate_per_s"
    "inflow;tpmc_event_rate_factor_per_s" "inflow;channels")
  require_nonempty(${path})
endforeach()
require_equal("36" geometry triangles)
require_equal("ON" geometry closed)
string(JSON n_components LENGTH "${J}" geometry components)
if(NOT n_components EQUAL 3)
  message(FATAL_ERROR "metadata: expected 3 geometry components, got ${n_components}")
endif()

message(STATUS "run metadata complete (backend ${EXPECTED_BACKEND})")
