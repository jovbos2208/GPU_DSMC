# vleo_geometry: STL import, mesh validation/orientation, ray-triangle
# intersection, BVH (host build, device traversal).
add_library(vleo_geometry STATIC
  Bvh.cpp
  Geometry.cpp
  Mesh.cpp
  Primitives.cpp
  Stl.cpp)
target_link_libraries(vleo_geometry PUBLIC vleo_core PRIVATE vleo_warnings)
