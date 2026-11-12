# vleo_gsi: Maxwell and CLL gas-surface interaction kernels. Header-only
# (include/vleo/gsi/Scatter.hpp) so that the same inline device code is used by
# TPMC and DSMC on every backend.
add_library(vleo_gsi INTERFACE)
target_link_libraries(vleo_gsi INTERFACE vleo_core)
