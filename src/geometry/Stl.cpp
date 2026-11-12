#include "vleo/geometry/Stl.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>

namespace vleo {

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw StlError("cannot open STL file '" + file.string() + "'");
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

std::uint32_t read_u32_le(const char* p) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

float read_f32_le(const char* p) {
  const std::uint32_t u = read_u32_le(p);
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

void write_u32_le(std::ostream& os, std::uint32_t v) {
  const char b[4] = {static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF),
                     static_cast<char>((v >> 16) & 0xFF), static_cast<char>((v >> 24) & 0xFF)};
  os.write(b, 4);
}

void write_f32_le(std::ostream& os, double d) {
  const auto f = static_cast<float>(d);
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  write_u32_le(os, u);
}

bool looks_like_binary(const std::string& data) {
  if (data.size() < 84) return false;
  const std::uint64_t n = read_u32_le(data.data() + 80);
  return data.size() == 84 + 50 * n;
}

void check_finite(const Vec3& v, const std::string& where) {
  if (!is_finite(v)) throw StlError(where + ": non-finite coordinate");
}

TriangleSoup parse_binary(const std::string& data, double scale, const std::string& name) {
  TriangleSoup soup;
  soup.format = StlFormat::Binary;
  const std::uint32_t n = read_u32_le(data.data() + 80);
  if (n == 0) throw StlError(name + ": binary STL contains no triangles");
  soup.triangles.reserve(n);
  soup.stored_normals.reserve(n);
  soup.material.assign(n, 0u);
  soup.solid_names.emplace_back(std::string(data.data(), strnlen(data.data(), 80)));
  for (std::uint32_t i = 0; i < n; ++i) {
    const char* p = data.data() + 84 + 50ull * i;
    float f[12];
    for (int k = 0; k < 12; ++k) f[k] = read_f32_le(p + 4 * k);
    const Vec3 nrm(f[0], f[1], f[2]);
    Triangle t{Vec3(f[3], f[4], f[5]) * scale, Vec3(f[6], f[7], f[8]) * scale, Vec3(f[9], f[10], f[11]) * scale};
    const std::string where = name + ": triangle " + std::to_string(i);
    check_finite(t.a, where);
    check_finite(t.b, where);
    check_finite(t.c, where);
    soup.triangles.push_back(t);
    soup.stored_normals.push_back(is_finite(nrm) ? nrm : Vec3());
  }
  return soup;
}

// Line-oriented ASCII parser.
class AsciiParser {
 public:
  AsciiParser(const std::string& data, double scale, std::string name)
      : in_(data), scale_(scale), name_(std::move(name)) {}

  TriangleSoup parse() {
    TriangleSoup soup;
    soup.format = StlFormat::Ascii;
    while (next_line()) {
      if (tok_.empty()) continue;
      if (tok_[0] != "solid") fail("expected 'solid'");
      std::string sname;
      for (std::size_t i = 1; i < tok_.size(); ++i) sname += (i > 1 ? " " : "") + tok_[i];
      const auto material = static_cast<std::uint32_t>(soup.solid_names.size());
      soup.solid_names.push_back(sname);
      parse_solid(soup, material);
    }
    if (soup.solid_names.empty()) fail("no 'solid' found");
    if (soup.triangles.empty()) throw StlError(name_ + ": ASCII STL contains no triangles");
    return soup;
  }

 private:
  void parse_solid(TriangleSoup& soup, std::uint32_t material) {
    for (;;) {
      if (!next_nonempty()) fail("unexpected end of file inside solid (missing 'endsolid')");
      if (tok_[0] == "endsolid") return;
      if (tok_[0] != "facet" || tok_.size() != 5 || tok_[1] != "normal") fail("expected 'facet normal nx ny nz'");
      const Vec3 nrm = vec_from(2);
      expect_line({"outer", "loop"});
      Triangle t;
      for (int k = 0; k < 3; ++k) {
        if (!next_nonempty() || tok_[0] != "vertex" || tok_.size() != 4) fail("expected 'vertex x y z'");
        const Vec3 v = vec_from(1) * scale_;
        (k == 0 ? t.a : (k == 1 ? t.b : t.c)) = v;
      }
      expect_line({"endloop"});
      expect_line({"endfacet"});
      soup.triangles.push_back(t);
      soup.stored_normals.push_back(nrm);
      soup.material.push_back(material);
    }
  }

  void expect_line(std::initializer_list<std::string_view> words) {
    if (!next_nonempty() || tok_.size() != words.size()) fail(expected_msg(words));
    std::size_t i = 0;
    for (auto w : words)
      if (tok_[i++] != w) fail(expected_msg(words));
  }

  static std::string expected_msg(std::initializer_list<std::string_view> words) {
    std::string s = "expected '";
    bool first = true;
    for (auto w : words) {
      s += (first ? "" : " ") + std::string(w);
      first = false;
    }
    return s + "'";
  }

  Vec3 vec_from(std::size_t first) {
    Vec3 v;
    for (int k = 0; k < 3; ++k) {
      const std::string& s = tok_[first + static_cast<std::size_t>(k)];
      std::string_view sv(s);
      if (!sv.empty() && sv.front() == '+') sv.remove_prefix(1);
      double d = 0.0;
      const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), d);
      if (ec != std::errc() || ptr != sv.data() + sv.size() || !std::isfinite(d))
        fail("invalid number '" + s + "'");
      v[k] = d;
    }
    return v;
  }

  bool next_line() {
    std::string line;
    if (!std::getline(in_, line)) return false;
    ++line_no_;
    tok_.clear();
    std::istringstream ls(line);
    std::string w;
    while (ls >> w) tok_.push_back(w);
    return true;
  }

  bool next_nonempty() {
    while (next_line())
      if (!tok_.empty()) return true;
    return false;
  }

  [[noreturn]] void fail(const std::string& msg) const {
    throw StlError(name_ + ":" + std::to_string(line_no_) + ": " + msg);
  }

  std::istringstream in_;
  double scale_;
  std::string name_;
  std::vector<std::string> tok_;
  std::size_t line_no_ = 0;
};

}  // namespace

TriangleSoup read_stl(const fs::path& file, double scale_to_m) {
  if (!(scale_to_m > 0.0) || !std::isfinite(scale_to_m)) throw StlError("invalid STL scale factor");
  const std::string data = read_file(file);
  const std::string name = file.string();
  if (looks_like_binary(data)) return parse_binary(data, scale_to_m, name);

  const auto first = data.find_first_not_of(" \t\r\n");
  if (first != std::string::npos && data.compare(first, 5, "solid") == 0)
    return AsciiParser(data, scale_to_m, name).parse();

  if (data.size() >= 84)
    throw StlError(name + ": not a valid STL file (binary size mismatch: header announces " +
                   std::to_string(read_u32_le(data.data() + 80)) + " triangles, file has " +
                   std::to_string(data.size()) + " bytes)");
  throw StlError(name + ": not a valid STL file");
}

void write_stl_ascii(const fs::path& file, const TriangleSoup& soup) {
  std::ofstream os(file, std::ios::binary | std::ios::trunc);
  if (!os) throw StlError("cannot write '" + file.string() + "'");
  auto num = [](double d) {
    char buf[32];
    const auto [p, ec] = std::to_chars(buf, buf + sizeof(buf), d);
    return std::string(buf, p);
  };
  auto vec = [&](const Vec3& v) { return num(v.x) + " " + num(v.y) + " " + num(v.z); };
  const std::size_t n_solids = soup.solid_names.empty() ? 1 : soup.solid_names.size();
  for (std::size_t s = 0; s < n_solids; ++s) {
    const std::string sname = soup.solid_names.empty() ? "vleo" : soup.solid_names[s];
    os << "solid " << sname << "\n";
    for (std::size_t i = 0; i < soup.triangles.size(); ++i) {
      const std::uint32_t m = i < soup.material.size() ? soup.material[i] : 0u;
      if (m != s) continue;
      const Triangle& t = soup.triangles[i];
      const Vec3 n = normalized_or(area_vector_x2(t), Vec3());
      os << "  facet normal " << vec(n) << "\n    outer loop\n"
         << "      vertex " << vec(t.a) << "\n      vertex " << vec(t.b) << "\n      vertex " << vec(t.c) << "\n"
         << "    endloop\n  endfacet\n";
    }
    os << "endsolid " << sname << "\n";
  }
  if (!os) throw StlError("error while writing '" + file.string() + "'");
}

void write_stl_binary(const fs::path& file, const TriangleSoup& soup) {
  std::ofstream os(file, std::ios::binary | std::ios::trunc);
  if (!os) throw StlError("cannot write '" + file.string() + "'");
  char header[80] = {};
  const char* text = "binary STL written by vleo";
  std::memcpy(header, text, std::strlen(text));
  os.write(header, 80);
  write_u32_le(os, static_cast<std::uint32_t>(soup.triangles.size()));
  for (const Triangle& t : soup.triangles) {
    const Vec3 n = normalized_or(area_vector_x2(t), Vec3());
    for (const Vec3& v : {n, t.a, t.b, t.c}) {
      write_f32_le(os, v.x);
      write_f32_le(os, v.y);
      write_f32_le(os, v.z);
    }
    const char attr[2] = {0, 0};
    os.write(attr, 2);
  }
  if (!os) throw StlError("error while writing '" + file.string() + "'");
}

}  // namespace vleo
