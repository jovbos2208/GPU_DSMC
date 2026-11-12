# vleo_tally: statistical reduction of per-history tallies (moments, standard
# errors, confidence intervals).
add_library(vleo_tally STATIC Statistics.cpp TimeSeries.cpp)
target_link_libraries(vleo_tally PUBLIC vleo_core vleo_particles PRIVATE vleo_warnings)
