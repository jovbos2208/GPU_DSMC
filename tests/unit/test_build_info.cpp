// M0.5: build provenance (git commit, compiler, build type, Kokkos, backend).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Kokkos_Core.hpp>

#include "vleo/core/BuildInfo.hpp"
#include "vleo/core/SelfTest.hpp"
#include "vleo/io/PlatformConfig.hpp"

using namespace vleo;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("Build information is complete", "[buildinfo]") {
  const BuildInfo& b = build_info();
  CHECK(!b.configure.solver_version.empty());
  CHECK(!b.configure.build_type.empty());
  CHECK(!b.configure.compiler_id.empty());
  CHECK(!b.configure.compiler_version.empty());
  CHECK(!b.compiler_detail.empty());
  CHECK(b.cxx_standard >= 202002L);
  CHECK(b.kokkos_version == std::to_string(KOKKOS_VERSION_MAJOR) + "." + std::to_string(KOKKOS_VERSION_MINOR) + "." +
                                std::to_string(KOKKOS_VERSION_PATCH));
  CHECK(b.default_execution_space == Kokkos::DefaultExecutionSpace::name());
  CHECK(!b.enabled_kokkos_backends.empty());
  CHECK(b.configure.backend == to_string(compiled_backend()));
  // Git: either a 40-hex-digit commit or the explicit "unknown" marker.
  if (b.git.available) {
    CHECK(b.git.commit.size() == 40);
    CHECK(b.git.commit.find_first_not_of("0123456789abcdef") == std::string_view::npos);
  } else {
    CHECK(b.git.commit == "unknown");
  }
}

TEST_CASE("Build information string lists every field", "[buildinfo]") {
  const std::string s = build_info_string();
  for (const char* key : {"git commit", "build type", "compiler", "Kokkos version", "backend", "execution space",
                          "Kokkos backends", "Kokkos arch"}) {
    CHECK_THAT(s, ContainsSubstring(key));
  }
}

TEST_CASE("Backend self-test passes", "[buildinfo][device]") {
  const SelfTestResult r = backend_self_test(2024);
  INFO(r.detail);
  CHECK(r.passed);
}
