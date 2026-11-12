# vleo_gsi: Maxwell and CLL gas-surface interaction kernels (M2).
# Defined in M0 to fix the library structure and dependency graph; it is an
# INTERFACE library until its first translation unit exists.
add_library(vleo_gsi INTERFACE)
target_link_libraries(vleo_gsi INTERFACE vleo_core)
