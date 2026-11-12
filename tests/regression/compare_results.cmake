# Compares the "results" and "std_error" blocks of two results.json files
# textually (bitwise identical numbers). Usage: cmake -DA=<file> -DB=<file> -P compare_results.cmake
file(READ "${A}" JA)
file(READ "${B}" JB)
foreach(block results std_error)
  string(JSON a GET "${JA}" ${block})
  string(JSON b GET "${JB}" ${block})
  if(NOT a STREQUAL b)
    message(FATAL_ERROR "${block} differ between\n  ${A}\n  ${B}\n${a}\n---\n${b}")
  endif()
endforeach()
message(STATUS "results identical")
