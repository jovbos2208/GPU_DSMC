# vleo_geometry: triangle primitives and basic geometric operations.
# M1 adds STL import, normal checks and the BVH (then this becomes a STATIC library).
add_library(vleo_geometry INTERFACE)
target_link_libraries(vleo_geometry INTERFACE vleo_core)
