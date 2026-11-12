# vleo_tally: Momentum/energy tallies and statistical reduction (M2).
# Defined in M0 to fix the library structure and dependency graph; it is an
# INTERFACE library until its first translation unit exists.
add_library(vleo_tally INTERFACE)
target_link_libraries(vleo_tally INTERFACE vleo_core)
