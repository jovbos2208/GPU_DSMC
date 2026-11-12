#pragma once

#include <cstdint>
#include <string>

namespace vleo {

struct SelfTestResult {
  bool passed = false;
  std::string detail;
};

// Backend self-test (requires an initialised Kokkos runtime): draws Philox
// words on the default execution space and compares an order-independent
// integer checksum with the same computation on the host. Verifies that
// kernels really execute on the compiled backend and that the counter-based
// RNG is bitwise reproducible across backends.
SelfTestResult backend_self_test(std::uint64_t seed);

}  // namespace vleo
