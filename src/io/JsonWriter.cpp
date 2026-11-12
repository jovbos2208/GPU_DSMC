#include "vleo/io/JsonWriter.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>

namespace vleo {

std::string JsonWriter::escape(std::string_view s) {
  std::string r;
  r.reserve(s.size() + 2);
  r += '"';
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': r += "\\\""; break;
      case '\\': r += "\\\\"; break;
      case '\b': r += "\\b"; break;
      case '\f': r += "\\f"; break;
      case '\n': r += "\\n"; break;
      case '\r': r += "\\r"; break;
      case '\t': r += "\\t"; break;
      default:
        if (c < 0x20) {
          static constexpr char hex[] = "0123456789abcdef";
          r += "\\u00";
          r += hex[c >> 4];
          r += hex[c & 0xF];
        } else {
          r += ch;  // UTF-8 passes through unchanged
        }
    }
  }
  r += '"';
  return r;
}

void JsonWriter::newline() {
  if (indent_ <= 0) return;
  out_ += '\n';
  out_.append(stack_.size() * static_cast<std::size_t>(indent_), ' ');
}

void JsonWriter::before_value() {
  if (stack_.empty()) {
    if (root_written_) throw std::logic_error("JsonWriter: only one root value allowed");
    root_written_ = true;
    return;
  }
  Level& top = stack_.back();
  if (top.is_object) {
    if (!have_key_) throw std::logic_error("JsonWriter: value inside object without key");
    have_key_ = false;
    ++top.count;  // separator and newline were emitted by key()
  } else {
    if (top.count++ > 0) out_ += ',';
    newline();
  }
}

JsonWriter& JsonWriter::key(std::string_view k) {
  if (stack_.empty() || !stack_.back().is_object) throw std::logic_error("JsonWriter: key outside object");
  if (have_key_) throw std::logic_error("JsonWriter: two keys in a row");
  if (stack_.back().count > 0) out_ += ',';
  newline();
  out_ += escape(k);
  out_ += indent_ > 0 ? ": " : ":";
  have_key_ = true;
  return *this;
}

JsonWriter& JsonWriter::begin_object() {
  before_value();
  out_ += '{';
  stack_.push_back({true, 0});
  return *this;
}

JsonWriter& JsonWriter::end_object() {
  if (stack_.empty() || !stack_.back().is_object || have_key_)
    throw std::logic_error("JsonWriter: mismatched end_object");
  const bool nonempty = stack_.back().count > 0;
  stack_.pop_back();
  if (nonempty) newline();
  out_ += '}';
  return *this;
}

JsonWriter& JsonWriter::begin_array() {
  before_value();
  out_ += '[';
  stack_.push_back({false, 0});
  return *this;
}

JsonWriter& JsonWriter::end_array() {
  if (stack_.empty() || stack_.back().is_object) throw std::logic_error("JsonWriter: mismatched end_array");
  const bool nonempty = stack_.back().count > 0;
  stack_.pop_back();
  if (nonempty) newline();
  out_ += ']';
  return *this;
}

JsonWriter& JsonWriter::value(std::string_view s) {
  before_value();
  out_ += escape(s);
  return *this;
}

JsonWriter& JsonWriter::value(double d) {
  if (!std::isfinite(d)) return null();
  before_value();
  char buf[64];
  const auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), d);
  std::string_view s(buf, static_cast<std::size_t>(ptr - buf));
  out_ += s;
  // Keep doubles recognisable as reals (e.g. "1" -> "1.0") without changing the value.
  if (s.find_first_of(".eE") == std::string_view::npos) out_ += ".0";
  (void)ec;
  return *this;
}

JsonWriter& JsonWriter::value(std::int64_t i) {
  before_value();
  out_ += std::to_string(i);
  return *this;
}

JsonWriter& JsonWriter::value(std::uint64_t u) {
  before_value();
  out_ += std::to_string(u);
  return *this;
}

JsonWriter& JsonWriter::value(bool b) {
  before_value();
  out_ += b ? "true" : "false";
  return *this;
}

JsonWriter& JsonWriter::value(const Vec3& v) {
  // Vectors are written on one line for readability.
  before_value();
  out_ += '[';
  for (int i = 0; i < 3; ++i) {
    if (i) out_ += ", ";
    if (!std::isfinite(v[i])) {
      out_ += "null";
      continue;
    }
    char buf[64];
    const auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v[i]);
    std::string_view s(buf, static_cast<std::size_t>(ptr - buf));
    out_ += s;
    if (s.find_first_of(".eE") == std::string_view::npos) out_ += ".0";
    (void)ec;
  }
  out_ += ']';
  return *this;
}

JsonWriter& JsonWriter::null() {
  before_value();
  out_ += "null";
  return *this;
}

std::string JsonWriter::str() const {
  if (!stack_.empty()) throw std::logic_error("JsonWriter: unclosed containers");
  return out_ + (indent_ > 0 ? "\n" : "");
}

}  // namespace vleo
