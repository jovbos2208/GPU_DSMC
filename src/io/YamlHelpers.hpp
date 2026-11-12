#pragma once

// Internal helpers for strict, error-collecting YAML validation.
// Not part of the public interface (yaml-cpp stays a private dependency).

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "vleo/core/Vec3.hpp"
#include "vleo/io/ConfigError.hpp"

namespace vleo::io_detail {

class Issues {
 public:
  void add(const std::string& path, const std::string& message, const YAML::Node* where = nullptr);
  bool empty() const { return list_.empty(); }
  std::size_t size() const { return list_.size(); }
  const std::vector<std::string>& list() const { return list_; }
  void throw_if_any(const std::string& context) const;

 private:
  std::vector<std::string> list_;
};

enum class Need { Required, Optional };

std::string join_path(const std::string& path, std::string_view key);

// Verifies that `node` is a mapping whose keys are unique strings from
// `allowed`. Returns false (and records an issue) if it is not a mapping.
bool check_map(const YAML::Node& node, const std::string& path, std::initializer_list<std::string_view> allowed,
               Issues& issues);

// Child lookup. Returns nullopt if absent (recording an issue if required) or
// explicitly null (always an issue).
std::optional<YAML::Node> child(const YAML::Node& parent, std::string_view key, const std::string& path,
                                Issues& issues, Need need);

// Typed getters. Each returns nullopt on absence or on an invalid value; an
// issue is recorded for invalid values and for missing required keys.
std::optional<double> get_double(const YAML::Node& parent, std::string_view key, const std::string& path,
                                 Issues& issues, Need need);
std::optional<std::uint64_t> get_uint64(const YAML::Node& parent, std::string_view key, const std::string& path,
                                        Issues& issues, Need need);
std::optional<bool> get_bool(const YAML::Node& parent, std::string_view key, const std::string& path, Issues& issues,
                             Need need);
std::optional<std::string> get_string(const YAML::Node& parent, std::string_view key, const std::string& path,
                                      Issues& issues, Need need, bool allow_empty = false);
std::optional<Vec3> get_vec3(const YAML::Node& parent, std::string_view key, const std::string& path, Issues& issues,
                             Need need);

// Scalar conversions (no issue recording).
std::optional<double> scalar_to_double(const YAML::Node& node);
std::optional<std::uint64_t> scalar_to_uint64(const YAML::Node& node);

// Range check helper: records an issue and returns false if violated.
struct Range {
  double lo;
  double hi;
  bool lo_inclusive = true;
  bool hi_inclusive = true;
};
bool check_range(double value, const Range& r, const std::string& path, Issues& issues,
                 const YAML::Node* where = nullptr);

// get_double followed by check_range; the issue carries the value's line number.
struct Range;
std::optional<double> get_double_in(const YAML::Node& parent, std::string_view key, const std::string& path,
                                    Issues& issues, Need need, const Range& range);

// Shortest round-trip representation, e.g. "0.1", "1e+14", "300".
std::string format_double(double v);
// Same, but always recognisable as a real number in YAML/JSON ("300.0").
std::string format_real(double v);

template <class E>
struct EnumEntry {
  std::string_view name;
  E value;
};

bool iequals(std::string_view a, std::string_view b);

template <class E, std::size_t N>
std::optional<E> get_enum(const YAML::Node& parent, std::string_view key, const std::string& path, Issues& issues,
                          Need need, const std::array<EnumEntry<E>, N>& table,
                          std::initializer_list<std::pair<std::string_view, std::string_view>> not_yet_supported = {}) {
  const auto s = get_string(parent, key, path, issues, need);
  if (!s) return std::nullopt;
  for (const auto& e : table)
    if (iequals(*s, e.name)) return e.value;
  const std::string p = join_path(path, key);
  const YAML::Node n = parent[std::string(key)];
  for (const auto& [name, reason] : not_yet_supported) {
    if (iequals(*s, name)) {
      issues.add(p, "'" + *s + "' is not supported yet: " + std::string(reason), &n);
      return std::nullopt;
    }
  }
  std::string allowed;
  for (const auto& e : table) allowed += (allowed.empty() ? "" : ", ") + std::string(e.name);
  issues.add(p, "invalid value '" + *s + "' (allowed: " + allowed + ")", &n);
  return std::nullopt;
}

}  // namespace vleo::io_detail
