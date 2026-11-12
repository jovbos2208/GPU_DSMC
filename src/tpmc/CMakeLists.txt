# vleo_tpmc: steady-state TPMC driver (GSI coupling, batches, convergence
# stop) and result writers.
add_library(vleo_tpmc STATIC Results.cpp Tpmc.cpp)
target_link_libraries(vleo_tpmc PUBLIC vleo_core vleo_geometry vleo_particles vleo_gsi vleo_tally vleo_io
                                PRIVATE vleo_warnings)
