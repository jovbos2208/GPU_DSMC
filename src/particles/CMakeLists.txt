# vleo_particles: SoA particle storage, simulation domain, free-stream inflow
# sampling and collisionless transport with multiple surface interactions.
add_library(vleo_particles STATIC
  Inflow.cpp
  ParticleStorage.cpp
  Transport.cpp)
target_link_libraries(vleo_particles PUBLIC vleo_core vleo_geometry vleo_io PRIVATE vleo_warnings)
