# vleo_dsmc: Collision cells, NTC and VHS collisions (M3).
# Defined in M0 to fix the library structure and dependency graph; it is an
# INTERFACE library until its first translation unit exists.
add_library(vleo_dsmc INTERFACE)
target_link_libraries(vleo_dsmc INTERFACE vleo_core vleo_geometry vleo_particles vleo_gsi vleo_tally)
