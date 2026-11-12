// JSON writer used for run metadata and results.

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <stdexcept>

#include "vleo/io/JsonWriter.hpp"

using namespace vleo;

TEST_CASE("JSON writer produces well-formed nested documents", "[json]") {
  JsonWriter w(0);  // compact
  w.begin_object();
  w.field("a", 1);
  w.field("b", 2.5);
  w.field("c", "text");
  w.field("d", true);
  w.key("e").null();
  w.key("f").begin_array().value(1.0).value(std::uint64_t{18446744073709551615ull}).begin_object().end_object().end_array();
  w.field("g", Vec3(1.0, -0.5, 1e-20));
  w.key("h").begin_array().end_array();
  w.end_object();
  CHECK(w.str() ==
        R"({"a":1,"b":2.5,"c":"text","d":true,"e":null,"f":[1.0,18446744073709551615,{}],"g":[1.0, -0.5, 1e-20],"h":[]})");
}

TEST_CASE("JSON writer escapes strings and maps non-finite numbers to null", "[json]") {
  CHECK(JsonWriter::escape("a\"b\\c\n\t\x01") == R"("a\"b\\c\n\t\u0001")");
  CHECK(JsonWriter::escape("Größe") == "\"Größe\"");
  JsonWriter w(0);
  w.begin_array()
      .value(std::numeric_limits<double>::quiet_NaN())
      .value(std::numeric_limits<double>::infinity())
      .value(Vec3(0.0, std::numeric_limits<double>::infinity(), 3.0))
      .value(0.1)
      .value(-0.0)
      .end_array();
  CHECK(w.str() == "[null,null,[0.0, null, 3.0],0.1,-0.0]");
}

TEST_CASE("JSON writer pretty printing", "[json]") {
  JsonWriter w;
  w.begin_object().field("x", 1).key("y").begin_array().value(2).end_array().end_object();
  CHECK(w.str() == "{\n  \"x\": 1,\n  \"y\": [\n    2\n  ]\n}\n");
}

TEST_CASE("JSON writer rejects misuse", "[json]") {
  JsonWriter a;
  a.begin_object();
  CHECK_THROWS_AS(a.value(1), std::logic_error);  // value without key
  CHECK_THROWS_AS(a.str(), std::logic_error);     // unclosed
  JsonWriter b;
  b.begin_array();
  CHECK_THROWS_AS(b.key("k"), std::logic_error);  // key in array
  CHECK_THROWS_AS(b.end_object(), std::logic_error);
  JsonWriter c;
  c.value(1);
  CHECK_THROWS_AS(c.value(2), std::logic_error);  // two roots
}
