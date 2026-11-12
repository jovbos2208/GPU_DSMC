# Interface target carrying the warning flags for project code only
# (third-party code is compiled with its own flags).
add_library(vleo_warnings INTERFACE)

if(VLEO_ENABLE_WARNINGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vleo_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                                                   -Wno-sign-conversion)
    if(VLEO_WARNINGS_AS_ERRORS)
      target_compile_options(vleo_warnings INTERFACE -Werror)
    endif()
  endif()
endif()
