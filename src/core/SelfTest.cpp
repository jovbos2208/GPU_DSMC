#include "vleo/core/SelfTest.hpp"

#include <sstream>

#include <Kokkos_Core.hpp>

#include "vleo/core/Random.hpp"

namespace vleo {

SelfTestResult backend_self_test(std::uint64_t seed) {
  constexpr int kParticles = 4096;
  constexpr int kDraws = 16;

  std::uint64_t host_sum = 0;
  for (int p = 0; p < kParticles; ++p) {
    RandomStream rng(seed, static_cast<std::uint64_t>(p), 0u, RngStream::Test);
    for (int k = 0; k < kDraws; ++k) host_sum += rng.next_u32();
  }

  std::uint64_t device_sum = 0;
  Kokkos::parallel_reduce(
      "vleo::backend_self_test", Kokkos::RangePolicy<>(0, kParticles),
      KOKKOS_LAMBDA(const int p, std::uint64_t& acc) {
        RandomStream rng(seed, static_cast<std::uint64_t>(p), 0u, RngStream::Test);
        for (int k = 0; k < kDraws; ++k) acc += rng.next_u32();
      },
      device_sum);
  Kokkos::fence();

  SelfTestResult r;
  r.passed = host_sum == device_sum;
  std::ostringstream os;
  os << "backend self-test on " << Kokkos::DefaultExecutionSpace::name() << ": " << kParticles << "x" << kDraws
     << " Philox words, host checksum " << host_sum << ", device checksum " << device_sum
     << (r.passed ? " (passed)" : " (FAILED)");
  r.detail = os.str();
  return r;
}

}  // namespace vleo
