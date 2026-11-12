#include "vleo/geometry/Mesh.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace vleo {

namespace {

struct VertexKey {
  std::uint64_t x, y, z;
  bool operator==(const VertexKey&) const = default;
};

struct VertexKeyHash {
  std::size_t operator()(const VertexKey& k) const noexcept {
    std::uint64_t h = k.x * 0x9E3779B97F4A7C15ull;
    h ^= k.y + 0x7F4A7C159E3779B9ull + (h << 6) + (h >> 2);
    h ^= k.z + 0x94D049BB133111EBull + (h << 6) + (h >> 2);
    return static_cast<std::size_t>(h);
  }
};

VertexKey key_of(const Vec3& v) {
  // + 0.0 maps -0.0 to +0.0 so that both weld together.
  return {std::bit_cast<std::uint64_t>(v.x + 0.0), std::bit_cast<std::uint64_t>(v.y + 0.0),
          std::bit_cast<std::uint64_t>(v.z + 0.0)};
}

struct UnionFind {
  std::vector<std::uint32_t> parent;
  explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
  std::uint32_t find(std::uint32_t a) {
    while (parent[a] != a) a = parent[a] = parent[parent[a]];
    return a;
  }
  void unite(std::uint32_t a, std::uint32_t b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }
};

struct EdgeUse {
  std::uint32_t tri;
  bool forward;  // traversed from the smaller to the larger vertex index
};

}  // namespace

void HostMesh::permute(const std::vector<std::uint32_t>& order) {
  auto apply = [&order](auto& v) {
    auto old = v;
    for (std::size_t i = 0; i < order.size(); ++i) v[i] = old[order[i]];
  };
  apply(triangles);
  apply(normals);
  apply(areas);
  apply(material);
  apply(component);
  apply(two_sided);
  apply(source_index);
}

HostMesh build_host_mesh(const TriangleSoup& soup, const MeshOptions& options, MeshReport& report) {
  report = MeshReport{};
  report.input_triangles = soup.triangles.size();

  // --- 1. Remove degenerate triangles --------------------------------------
  std::vector<Triangle> tris;
  std::vector<std::uint32_t> material, source;
  std::vector<Vec3> stored;
  for (std::size_t i = 0; i < soup.triangles.size(); ++i) {
    if (is_degenerate(soup.triangles[i], options.degenerate_rel_tol)) {
      ++report.degenerate_removed;
      continue;
    }
    tris.push_back(soup.triangles[i]);
    material.push_back(i < soup.material.size() ? soup.material[i] : 0u);
    stored.push_back(i < soup.stored_normals.size() ? soup.stored_normals[i] : Vec3());
    source.push_back(static_cast<std::uint32_t>(i));
  }
  if (report.degenerate_removed > 0)
    report.warnings.push_back(std::to_string(report.degenerate_removed) + " degenerate triangle(s) removed");
  if (tris.empty()) throw GeometryError("mesh contains no non-degenerate triangles");
  const std::size_t n = tris.size();

  // Stored normals vs. vertex order (diagnostic only).
  for (std::size_t i = 0; i < n; ++i)
    if (norm2(stored[i]) > 0.0 && dot(stored[i], area_vector_x2(tris[i])) < 0.0) ++report.stored_normal_disagreements;
  if (report.stored_normal_disagreements > 0)
    report.warnings.push_back(std::to_string(report.stored_normal_disagreements) +
                              " facet normal(s) in the file disagree with the vertex order; the vertex order is used");

  // --- 2. Weld vertices -----------------------------------------------------
  std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> vmap;
  vmap.reserve(3 * n);
  std::vector<std::array<std::uint32_t, 3>> idx(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Vec3* v[3] = {&tris[i].a, &tris[i].b, &tris[i].c};
    for (int k = 0; k < 3; ++k) {
      const auto [it, inserted] = vmap.emplace(key_of(*v[k]), static_cast<std::uint32_t>(vmap.size()));
      idx[i][static_cast<std::size_t>(k)] = it->second;
    }
  }
  report.vertices = vmap.size();

  // --- 3. Edge classification ----------------------------------------------
  std::unordered_map<std::uint64_t, std::vector<EdgeUse>> edges;
  edges.reserve(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      const std::uint32_t a = idx[i][static_cast<std::size_t>(k)];
      const std::uint32_t b = idx[i][static_cast<std::size_t>((k + 1) % 3)];
      const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b);
      edges[key].push_back({static_cast<std::uint32_t>(i), a < b});
    }
  }
  report.edges = edges.size();

  UnionFind uf(n);
  std::vector<std::uint8_t> bad_edge_touch(n, 0);  // triangle touches a boundary/nonmanifold/inconsistent edge
  for (const auto& [key, uses] : edges) {
    for (std::size_t u = 1; u < uses.size(); ++u) uf.unite(uses[0].tri, uses[u].tri);
    bool bad = false;
    if (uses.size() == 1) {
      ++report.boundary_edges;
      bad = true;
    } else if (uses.size() == 2) {
      if (uses[0].forward == uses[1].forward) {
        ++report.inconsistent_edges;
        bad = true;
      }
    } else {
      ++report.nonmanifold_edges;
      bad = true;
    }
    if (bad)
      for (const auto& use : uses) bad_edge_touch[use.tri] = 1;
  }

  // --- 4. Components, closedness, orientation ---------------------------------
  std::vector<std::uint32_t> comp_of(n);
  std::unordered_map<std::uint32_t, std::uint32_t> root_to_comp;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t r = uf.find(static_cast<std::uint32_t>(i));
    const auto [it, inserted] = root_to_comp.emplace(r, static_cast<std::uint32_t>(root_to_comp.size()));
    comp_of[i] = it->second;
  }
  std::vector<ComponentInfo> comps(root_to_comp.size());
  for (auto& c : comps) c.closed = true;

  Aabb box;
  for (const auto& t : tris) box.expand(bounding_box(t));
  const Vec3 ref = box.center();  // reference point for the signed volume (precision)

  for (std::size_t i = 0; i < n; ++i) {
    ComponentInfo& c = comps[comp_of[i]];
    ++c.triangles;
    if (bad_edge_touch[i]) c.closed = false;
    const Vec3 a = tris[i].a - ref, b = tris[i].b - ref, cc = tris[i].c - ref;
    c.signed_volume_m3 += dot(a, cross(b, cc)) / 6.0;
    c.area_m2 += area(tris[i]);
  }

  std::ostringstream errors;
  for (std::size_t ci = 0; ci < comps.size(); ++ci) {
    ComponentInfo& c = comps[ci];
    if (!c.closed) continue;
    // A closed surface must enclose a volume; compare with the isoperimetric scale.
    const double scale = std::pow(c.area_m2, 1.5);
    if (!(std::fabs(c.signed_volume_m3) > 1.0e-9 * scale)) {
      errors << "\n  - component " << ci << " is closed but encloses no volume (e.g. a doubled sheet)";
      continue;
    }
    if (c.signed_volume_m3 < 0.0) {
      c.flipped = true;
      c.signed_volume_m3 = -c.signed_volume_m3;
    }
  }

  report.closed = true;
  for (const auto& c : comps) report.closed = report.closed && c.closed;
  if (!report.closed) {
    std::size_t open = 0;
    for (const auto& c : comps) open += c.closed ? 0 : 1;
    std::ostringstream m;
    m << open << " of " << comps.size() << " component(s) are not closed (" << report.boundary_edges
      << " boundary, " << report.nonmanifold_edges << " non-manifold, " << report.inconsistent_edges
      << " inconsistently oriented edge(s))";
    if (options.require_closed)
      errors << "\n  - " << m.str() << "; set geometry.require_closed: false to treat open surfaces as two-sided";
    else
      report.warnings.push_back(m.str() + "; open components are treated as two-sided surfaces");
  }
  for (std::size_t ci = 0; ci < comps.size(); ++ci)
    if (comps[ci].flipped)
      report.warnings.push_back("component " + std::to_string(ci) +
                                " had inward-pointing normals and was re-oriented");

  const std::string err = errors.str();
  if (!err.empty()) throw GeometryError("invalid geometry:" + err);

  // --- 5. Assemble the host mesh ----------------------------------------------
  HostMesh mesh;
  mesh.triangles.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    Triangle t = tris[i];
    if (comps[comp_of[i]].flipped) std::swap(t.b, t.c);
    mesh.triangles.push_back(t);
    mesh.normals.push_back(unit_normal(t));
    mesh.areas.push_back(area(t));
    mesh.two_sided.push_back(comps[comp_of[i]].closed ? 0 : 1);
  }
  mesh.material = std::move(material);
  mesh.component = std::move(comp_of);
  mesh.source_index = std::move(source);
  mesh.material_names = soup.solid_names;
  if (mesh.material_names.empty()) mesh.material_names.emplace_back("");
  mesh.bbox = box;

  report.triangles = n;
  report.components = std::move(comps);
  report.bbox = box;
  report.total_area_m2 = 0.0;
  for (double a : mesh.areas) report.total_area_m2 += a;
  return mesh;
}

namespace {
template <class T>
Kokkos::View<const T*> to_device(const std::vector<T>& v, const char* label) {
  Kokkos::View<T*> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}
}  // namespace

DeviceMesh upload(const HostMesh& mesh) {
  std::vector<Vec3> a(mesh.size()), b(mesh.size()), c(mesh.size());
  for (std::size_t i = 0; i < mesh.size(); ++i) {
    a[i] = mesh.triangles[i].a;
    b[i] = mesh.triangles[i].b;
    c[i] = mesh.triangles[i].c;
  }
  DeviceMesh d;
  d.v0 = to_device(a, "mesh_v0");
  d.v1 = to_device(b, "mesh_v1");
  d.v2 = to_device(c, "mesh_v2");
  d.normal = to_device(mesh.normals, "mesh_normal");
  d.area = to_device(mesh.areas, "mesh_area");
  d.material = to_device(mesh.material, "mesh_material");
  d.two_sided = to_device(mesh.two_sided, "mesh_two_sided");
  d.num_triangles = static_cast<int>(mesh.size());
  return d;
}

}  // namespace vleo
