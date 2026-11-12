#include "YamlHelpers.hpp"

#include <charconv>
#include <cmath>
#include <set>
#include <sstream>

namespace vleo {

namespace {
std::string join_issues(const std::vector<std::string>& issues, const std::string& context) {
  std::ostringstream os;
  os << "invalid " << context << " (" << issues.size() << (issues.size() == 1 ? " issue" : " issues") << "):";
  for (const auto& i : issues) os << "\n  - " << i;
  return os.str();
}
}  // namespace

ConfigError::ConfigError(std::vector<std::string> issues, const std::string& context)
    : std::runtime_error(join_issues(issues, context)), issues_(std::move(issues)) {}

}  // namespace vleo

namespace vleo::io_detail {

void Issues::add(const std::string& path, const std::string& message, const YAML::Node* where) {
  std::string entry = path.empty() ? "<root>" : path;
  entry += ": " + message;
  if (where && where->IsDefined() && where->Mark().line >= 0) {
    entry += " (line " + std::to_string(where->Mark().line + 1) + ")";
  }
  list_.push_back(std::move(entry));
}

void Issues::throw_if_any(const std::string& context) const {
  if (!list_.empty()) throw ConfigError(list_, context);
}

std::string join_path(const std::string& path, std::string_view key) {
  return path.empty() ? std::string(key) : path + "." + std::string(key);
}

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; };
    if (lower(a[i]) != lower(b[i])) return false;
  }
  return true;
}

bool check_map(const YAML::Node& node, const std::string& path, std::initializer_list<std::string_view> allowed,
               Issues& issues) {
  if (!node.IsMap()) {
    issues.add(path, "must be a mapping (key: value pairs)", &node);
    return false;
  }
  std::set<std::string> seen;
  for (auto it = node.begin(); it != node.end(); ++it) {
    if (!it->first.IsScalar()) {
      issues.add(path, "keys must be plain strings", &it->first);
      continue;
    }
    const std::string key = it->first.Scalar();
    if (!seen.insert(key).second) {
      issues.add(join_path(path, key), "duplicate key", &it->first);
      continue;
    }
    bool ok = false;
    for (auto a : allowed) ok = ok || (a == key);
    if (!ok) {
      std::string list;
      for (auto a : allowed) list += (list.empty() ? "" : ", ") + std::string(a);
      issues.add(join_path(path, key), "unknown key (allowed here: " + list + ")", &it->first);
    }
  }
  return true;
}

std::optional<YAML::Node> child(const YAML::Node& parent, std::string_view key, const std::string& path,
                                Issues& issues, Need need) {
  if (!parent.IsMap()) return std::nullopt;
  const YAML::Node n = parent[std::string(key)];
  if (!n.IsDefined()) {
    if (need == Need::Required) issues.add(join_path(path, key), "required key is missing", &parent);
    return std::nullopt;
  }
  if (n.IsNull()) {
    issues.add(join_path(path, key), "has no value", &n);
    return std::nullopt;
  }
  return n;
}

namespace {

// A scalar is numeric only if it is plain (unquoted).
bool is_plain_scalar(const YAML::Node& n) { return n.IsScalar() && n.Tag() != "!"; }

std::string_view strip_plus(std::string_view s) {
  if (!s.empty() && s.front() == '+') s.remove_prefix(1);
  return s;
}

}  // namespace

std::optional<double> scalar_to_double(const YAML::Node& node) {
  if (!is_plain_scalar(node)) return std::nullopt;
  const std::string_view s = strip_plus(node.Scalar());
  if (s.empty()) return std::nullopt;
  double v = 0.0;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc() || ptr != s.data() + s.size() || !std::isfinite(v)) return std::nullopt;
  return v;
}

std::optional<std::uint64_t> scalar_to_uint64(const YAML::Node& node) {
  if (!is_plain_scalar(node)) return std::nullopt;
  const std::string_view s = strip_plus(node.Scalar());
  if (s.empty()) return std::nullopt;
  std::uint64_t v = 0;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v, 10);
  if (ec != std::errc() || ptr != s.data() + s.size()) return std::nullopt;
  return v;
}

std::optional<double> get_double(const YAML::Node& parent, std::string_view key, const std::string& path,
                                 Issues& issues, Need need) {
  const auto n = child(parent, key, path, issues, need);
  if (!n) return std::nullopt;
  const auto v = scalar_to_double(*n);
  if (!v) {
    issues.add(join_path(path, key),
               "expected a finite real number, got '" + (n->IsScalar() ? n->Scalar() : std::string("<non-scalar>")) +
                   "'",
               &*n);
  }
  return v;
}

std::optional<std::uint64_t> get_uint64(const YAML::Node& parent, std::string_view key, const std::string& path,
                                        Issues& issues, Need need) {
  const auto n = child(parent, key, path, issues, need);
  if (!n) return std::nullopt;
  const auto v = scalar_to_uint64(*n);
  if (!v) {
    issues.add(join_path(path, key),
               "expected a non-negative integer (decimal digits only, at most 2^64-1), got '" +
                   (n->IsScalar() ? n->Scalar() : std::string("<non-scalar>")) + "'",
               &*n);
  }
  return v;
}

std::optional<bool> get_bool(const YAML::Node& parent, std::string_view key, const std::string& path, Issues& issues,
                             Need need) {
  const auto n = child(parent, key, path, issues, need);
  if (!n) return std::nullopt;
  if (is_plain_scalar(*n)) {
    const std::string& s = n->Scalar();
    if (s == "true" || s == "True" || s == "TRUE") return true;
    if (s == "false" || s == "False" || s == "FALSE") return false;
  }
  issues.add(join_path(path, key),
             "expected true or false, got '" + (n->IsScalar() ? n->Scalar() : std::string("<non-scalar>")) + "'",
             &*n);
  return std::nullopt;
}

std::optional<std::string> get_string(const YAML::Node& parent, std::string_view key, const std::string& path,
                                      Issues& issues, Need need, bool allow_empty) {
  const auto n = child(parent, key, path, issues, need);
  if (!n) return std::nullopt;
  if (!n->IsScalar()) {
    issues.add(join_path(path, key), "expected a string", &*n);
    return std::nullopt;
  }
  if (!allow_empty && n->Scalar().empty()) {
    issues.add(join_path(path, key), "must not be empty", &*n);
    return std::nullopt;
  }
  return n->Scalar();
}

std::optional<Vec3> get_vec3(const YAML::Node& parent, std::string_view key, const std::string& path, Issues& issues,
                             Need need) {
  const auto n = child(parent, key, path, issues, need);
  if (!n) return std::nullopt;
  const std::string p = join_path(path, key);
  if (!n->IsSequence() || n->size() != 3) {
    issues.add(p, "expected a list of exactly 3 numbers [x, y, z]", &*n);
    return std::nullopt;
  }
  Vec3 v;
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    const YAML::Node e = (*n)[static_cast<std::size_t>(i)];
    const auto d = scalar_to_double(e);
    if (!d) {
      issues.add(p + "[" + std::to_string(i) + "]",
                 "expected a finite real number, got '" + (e.IsScalar() ? e.Scalar() : std::string("<non-scalar>")) +
                     "'",
                 &e);
      ok = false;
    } else {
      v[i] = *d;
    }
  }
  return ok ? std::optional<Vec3>(v) : std::nullopt;
}

std::string format_double(double v) {
  char buf[64];
  const auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
  return ec == std::errc() ? std::string(buf, ptr) : std::string("nan");
}

std::string format_real(double v) {
  std::string s = format_double(v);
  if (s.find_first_of(".eEn") == std::string::npos) s += ".0";  // 'n' covers inf/nan
  return s;
}

std::optional<double> get_double_in(const YAML::Node& parent, std::string_view key, const std::string& path,
                                    Issues& issues, Need need, const Range& range) {
  const auto v = get_double(parent, key, path, issues, need);
  if (!v) return std::nullopt;
  const YAML::Node n = parent[std::string(key)];
  return check_range(*v, range, join_path(path, key), issues, &n) ? v : std::nullopt;
}

bool check_range(double value, const Range& r, const std::string& path, Issues& issues, const YAML::Node* where) {
  const bool lo_ok = r.lo_inclusive ? value >= r.lo : value > r.lo;
  const bool hi_ok = r.hi_inclusive ? value <= r.hi : value < r.hi;
  if (lo_ok && hi_ok) return true;
  std::string interval = std::string(r.lo_inclusive ? "[" : "(") + format_double(r.lo) + ", " + format_double(r.hi) +
                         (r.hi_inclusive ? "]" : ")");
  issues.add(path, "value " + format_double(value) + " is outside the allowed range " + interval, where);
  return false;
}

}  // namespace vleo::io_detail
