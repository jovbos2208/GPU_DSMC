# Script mode (cmake -P): writes the git state of the source tree into a C++
# source file. Run on every build so the reported commit is never stale;
# configure_file() only touches the output when its content changes, so no
# unnecessary recompilation happens.
#
# Required variables: VLEO_SOURCE_DIR, VLEO_GIT_TEMPLATE, VLEO_GIT_OUTPUT

set(VLEO_GIT_COMMIT "unknown")
set(VLEO_GIT_DESCRIBE "unknown")
set(VLEO_GIT_DIRTY "false")
set(VLEO_GIT_AVAILABLE "false")

find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${VLEO_SOURCE_DIR}" rev-parse --verify HEAD
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _commit ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_rc EQUAL 0 AND _commit)
    set(VLEO_GIT_AVAILABLE "true")
    set(VLEO_GIT_COMMIT "${_commit}")

    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${VLEO_SOURCE_DIR}" describe --always --tags --dirty
      RESULT_VARIABLE _rc OUTPUT_VARIABLE _describe ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_rc EQUAL 0)
      set(VLEO_GIT_DESCRIBE "${_describe}")
    endif()

    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${VLEO_SOURCE_DIR}" status --porcelain --untracked-files=no
      RESULT_VARIABLE _rc OUTPUT_VARIABLE _status ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_rc EQUAL 0 AND NOT _status STREQUAL "")
      set(VLEO_GIT_DIRTY "true")
    endif()
  endif()
endif()

configure_file("${VLEO_GIT_TEMPLATE}" "${VLEO_GIT_OUTPUT}" @ONLY)
