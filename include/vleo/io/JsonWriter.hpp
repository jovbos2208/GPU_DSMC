#pragma once

// Minimal streaming JSON writer (RFC 8259) with pretty printing.
// Doubles are written in shortest round-trip form; non-finite doubles become
// null because JSON has no representation for them.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "vleo/core/Vec3.hpp"

namespace vleo {

class JsonWriter {
 public:
  explicit JsonWriter(int indent = 2) : indent_(indent) {}

  JsonWriter& begin_object();
  JsonWriter& end_object();
  JsonWriter& begin_array();
  JsonWriter& end_array();

  // Sets the key for the next value; only valid inside an object.
  JsonWriter& key(std::string_view k);

  JsonWriter& value(std::string_view s);
  JsonWriter& value(const char* s) { return value(std::string_view(s)); }
  JsonWriter& value(const std::string& s) { return value(std::string_view(s)); }
  JsonWriter& value(double d);
  JsonWriter& value(std::int64_t i);
  JsonWriter& value(std::uint64_t u);
  JsonWriter& value(int i) { return value(static_cast<std::int64_t>(i)); }
  JsonWriter& value(unsigned i) { return value(static_cast<std::uint64_t>(i)); }
  JsonWriter& value(bool b);
  JsonWriter& value(const Vec3& v);
  JsonWriter& null();

  // Convenience: key(k).value(v)
  template <class T>
  JsonWriter& field(std::string_view k, const T& v) {
    key(k);
    return value(v);
  }

  // The finished document. Throws std::logic_error if containers are open.
  std::string str() const;

  static std::string escape(std::string_view s);

 private:
  struct Level {
    bool is_object;
    std::size_t count;
  };
  void before_value();
  void newline();

  std::string out_;
  std::vector<Level> stack_;
  bool have_key_ = false;
  bool root_written_ = false;
  int indent_;
};

}  // namespace vleo
