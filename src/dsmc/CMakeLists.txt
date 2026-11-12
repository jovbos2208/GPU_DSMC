# vleo_dsmc: Cartesian collision grid with cut-cell volumes, GPU binning,
# NTC/VHS collisions and the DSMC time loop.
add_library(vleo_dsmc STATIC CollisionGrid.cpp Dsmc.cpp DsmcResults.cpp)
target_link_libraries(vleo_dsmc PUBLIC vleo_core vleo_geometry vleo_particles vleo_gsi vleo_tally vleo_tpmc vleo_io
                                PRIVATE vleo_warnings)
