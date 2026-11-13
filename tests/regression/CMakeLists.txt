# Smoke / regression tests of the vleo_solver executable (labels: smoke).

if(VLEO_BACKEND STREQUAL "cuda")
  set(_platform nvidia)
  set(_foreign_platform amd)
elseif(VLEO_BACKEND STREQUAL "hip")
  set(_platform amd)
  set(_foreign_platform cpu)
else()
  set(_platform cpu)
  set(_foreign_platform nvidia)
endif()

set(_cfg "${PROJECT_SOURCE_DIR}/configs")
set(_out "${CMAKE_CURRENT_BINARY_DIR}/smoke_out")

# 1. Full start-up on the compiled backend: read config, init Kokkos, run the
#    backend self-test, write resolved config + metadata.
add_test(NAME smoke_clean COMMAND ${CMAKE_COMMAND} -E rm -rf "${_out}")
set_tests_properties(smoke_clean PROPERTIES FIXTURES_SETUP smoke_clean LABELS smoke)

add_test(NAME smoke_run
  COMMAND vleo_solver --config "${_cfg}/simulations/tpmc_cll.yaml"
                      --platform "${_cfg}/platforms/${_platform}.yaml"
                      --histories 200000 --quiet
                      --output-dir "${_out}")
set_tests_properties(smoke_run PROPERTIES
  FIXTURES_REQUIRED smoke_clean FIXTURES_SETUP smoke_output LABELS smoke TIMEOUT ${_t120})

# 2. The metadata file is complete.
add_test(NAME smoke_metadata
  COMMAND ${CMAKE_COMMAND} -DOUT_DIR=${_out} -DEXPECTED_BACKEND=${VLEO_BACKEND}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/check_metadata.cmake")
set_tests_properties(smoke_metadata PROPERTIES FIXTURES_REQUIRED smoke_output LABELS smoke)

# 2b. The result file contains all six main quantities with uncertainties.
add_test(NAME smoke_results
  COMMAND ${CMAKE_COMMAND} -DRESULTS=${_out}/results.json -P "${CMAKE_CURRENT_SOURCE_DIR}/check_results.cmake")
set_tests_properties(smoke_results PROPERTIES FIXTURES_REQUIRED smoke_output LABELS smoke)

# 3. The written resolved configuration is itself a valid input.
add_test(NAME smoke_resolved_config_reparses
  COMMAND vleo_solver --check --config "${_out}/resolved_config.yaml")
set_tests_properties(smoke_resolved_config_reparses PROPERTIES FIXTURES_REQUIRED smoke_output LABELS smoke)

# 4. Identical simulation configurations are accepted by every build.
foreach(_sim tpmc_maxwell tpmc_cll tpmc_air dsmc_maxwell dsmc_cll)
  add_test(NAME smoke_check_${_sim}
    COMMAND vleo_solver --check --config "${_cfg}/simulations/${_sim}.yaml"
                        --platform "${_cfg}/platforms/${_platform}.yaml")
  set_tests_properties(smoke_check_${_sim} PROPERTIES LABELS smoke)
endforeach()

# 5. Failure paths.
add_test(NAME smoke_rejects_foreign_platform
  COMMAND vleo_solver --check --config "${_cfg}/simulations/tpmc_cll.yaml"
                      --platform "${_cfg}/platforms/${_foreign_platform}.yaml")
set_tests_properties(smoke_rejects_foreign_platform PROPERTIES
  WILL_FAIL ON LABELS smoke)
add_test(NAME smoke_rejects_foreign_platform_message
  COMMAND vleo_solver --check --config "${_cfg}/simulations/tpmc_cll.yaml"
                      --platform "${_cfg}/platforms/${_foreign_platform}.yaml")
set_tests_properties(smoke_rejects_foreign_platform_message PROPERTIES
  PASS_REGULAR_EXPRESSION "this binary was built for '${VLEO_BACKEND}'" LABELS smoke)

add_test(NAME smoke_rejects_invalid_config
  COMMAND vleo_solver --check --config "${CMAKE_CURRENT_SOURCE_DIR}/invalid_simulation.yaml")
set_tests_properties(smoke_rejects_invalid_config PROPERTIES
  PASS_REGULAR_EXPRESSION "invalid simulation configuration \\(3 issues\\)" LABELS smoke)

add_test(NAME smoke_usage_error COMMAND vleo_solver --no-such-option)
set_tests_properties(smoke_usage_error PROPERTIES WILL_FAIL ON LABELS smoke)

add_test(NAME smoke_version COMMAND vleo_solver --version)
set_tests_properties(smoke_version PROPERTIES
  PASS_REGULAR_EXPRESSION "Kokkos version   : [0-9]+\\.[0-9]+\\.[0-9]+" LABELS smoke)

# 6. Geometry errors are configuration errors (exit code 2) with a clear message.
add_test(NAME smoke_rejects_missing_geometry
  COMMAND vleo_solver --check --config "${CMAKE_CURRENT_SOURCE_DIR}/missing_geometry.yaml")
set_tests_properties(smoke_rejects_missing_geometry PROPERTIES
  PASS_REGULAR_EXPRESSION "cannot open STL file" LABELS smoke)
add_test(NAME smoke_rejects_open_geometry
  COMMAND vleo_solver --check --config "${CMAKE_CURRENT_SOURCE_DIR}/open_geometry.yaml")
set_tests_properties(smoke_rejects_open_geometry PROPERTIES
  PASS_REGULAR_EXPRESSION "are not closed.*require_closed: false" LABELS smoke)

# 7. Gate M2: Maxwell and CLL example runs are reproducible (bitwise identical
#    results for the same configuration, seed and build).
foreach(_sim tpmc_maxwell tpmc_cll)
  foreach(_rep a b)
    add_test(NAME repro_${_sim}_${_rep}
      COMMAND vleo_solver --config "${_cfg}/simulations/${_sim}.yaml" --histories 300000 --quiet
                          --output-dir "${CMAKE_CURRENT_BINARY_DIR}/repro_${_sim}_${_rep}")
    set_tests_properties(repro_${_sim}_${_rep} PROPERTIES FIXTURES_SETUP repro_${_sim} LABELS smoke TIMEOUT ${_t300})
  endforeach()
  add_test(NAME repro_${_sim}_compare
    COMMAND ${CMAKE_COMMAND} -DA=${CMAKE_CURRENT_BINARY_DIR}/repro_${_sim}_a/results.json
                             -DB=${CMAKE_CURRENT_BINARY_DIR}/repro_${_sim}_b/results.json
                             -P "${CMAKE_CURRENT_SOURCE_DIR}/compare_results.cmake")
  set_tests_properties(repro_${_sim}_compare PROPERTIES FIXTURES_REQUIRED repro_${_sim} LABELS smoke)
endforeach()

# 8. Gate M3: monatomic DSMC example runs are reproducible and complete.
foreach(_sim dsmc_maxwell dsmc_cll)
  foreach(_rep a b)
    add_test(NAME repro_${_sim}_${_rep}
      COMMAND vleo_solver --config "${_cfg}/simulations/${_sim}.yaml" --warmup-steps 30 --sample-steps 40 --quiet
                          --output-dir "${CMAKE_CURRENT_BINARY_DIR}/repro_${_sim}_${_rep}")
    set_tests_properties(repro_${_sim}_${_rep} PROPERTIES FIXTURES_SETUP repro_${_sim} LABELS smoke TIMEOUT ${_t300})
  endforeach()
  add_test(NAME repro_${_sim}_compare
    COMMAND ${CMAKE_COMMAND} -DA=${CMAKE_CURRENT_BINARY_DIR}/repro_${_sim}_a/results.json
                             -DB=${CMAKE_CURRENT_BINARY_DIR}/repro_${_sim}_b/results.json
                             -P "${CMAKE_CURRENT_SOURCE_DIR}/compare_results.cmake")
  set_tests_properties(repro_${_sim}_compare PROPERTIES FIXTURES_REQUIRED repro_${_sim} LABELS smoke)
  add_test(NAME repro_${_sim}_results
    COMMAND ${CMAKE_COMMAND} -DRESULTS=${CMAKE_CURRENT_BINARY_DIR}/repro_${_sim}_a/results.json
                             -P "${CMAKE_CURRENT_SOURCE_DIR}/check_results.cmake")
  set_tests_properties(repro_${_sim}_results PROPERTIES FIXTURES_REQUIRED repro_${_sim} LABELS smoke)
endforeach()

# 9. DSMC gas mixtures (M4.2) are accepted.
add_test(NAME smoke_check_dsmc_mixture
  COMMAND vleo_solver --check --config "${CMAKE_CURRENT_SOURCE_DIR}/dsmc_mixture.yaml")
set_tests_properties(smoke_check_dsmc_mixture PROPERTIES LABELS smoke)

# 10. M5.10: shipped examples with fixed seeds against stored references
#     (scripts/check_examples.py, tests/golden/); Release builds only.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_test(NAME examples_fixed_seeds
    COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/check_examples.py" $<TARGET_FILE:vleo_solver>)
  set_tests_properties(examples_fixed_seeds PROPERTIES LABELS "examples;smoke" TIMEOUT 1200)
endif()
