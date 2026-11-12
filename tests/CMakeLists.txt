# Test suite. Run everything with:   ctest --preset <preset>
# (equivalently: ctest --test-dir build/<preset> --output-on-failure)
# Labels: unit, smoke, config, rng.

add_library(vleo_test_main STATIC unit/test_main.cpp)
target_link_libraries(vleo_test_main PUBLIC Catch2::Catch2 Kokkos::kokkos vleo_core)

function(vleo_add_unit_test name)
  cmake_parse_arguments(T "" "" "SOURCES;LIBS;LABELS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE vleo_test_main vleo_warnings ${T_LIBS})
  target_compile_definitions(${name} PRIVATE VLEO_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
                                             VLEO_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME ${name} COMMAND ${name} --order lex --reporter console --colour-mode none)
  set_tests_properties(${name} PROPERTIES LABELS "unit;${T_LABELS}" TIMEOUT 300)
endfunction()

vleo_add_unit_test(test_vec3      SOURCES unit/test_vec3.cpp)
vleo_add_unit_test(test_frames    SOURCES unit/test_frames.cpp)
vleo_add_unit_test(test_geometry  SOURCES unit/test_geometry.cpp LIBS vleo_geometry)
vleo_add_unit_test(test_rng       SOURCES unit/test_rng.cpp LABELS rng)
vleo_add_unit_test(test_config    SOURCES unit/test_config.cpp LIBS vleo_io LABELS config)
vleo_add_unit_test(test_json      SOURCES unit/test_json.cpp LIBS vleo_io)
vleo_add_unit_test(test_build_info SOURCES unit/test_build_info.cpp LIBS vleo_io)

add_subdirectory(regression)
