# vleo_particles: SoA particle storage, inflow-flux injection and transport (M1).
# Defined in M0 to fix the library structure and dependency graph; it is an
# INTERFACE library until its first translation unit exists.
add_library(vleo_particles INTERFACE)
target_link_libraries(vleo_particles INTERFACE vleo_core vleo_geometry)
