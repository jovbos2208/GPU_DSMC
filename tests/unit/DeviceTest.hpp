#pragma once

// Helpers to evaluate code on the default execution space (the GPU in
// CUDA/HIP builds) and return the results to the host.
//
// Device code in tests is written as functor structs with a
// KOKKOS_INLINE_FUNCTION call operator rather than as KOKKOS_LAMBDA inside
// TEST_CASE bodies: Catch2 test functions have internal linkage, which the
// CUDA extended-lambda rules do not reliably allow.

#include <Kokkos_Core.hpp>

namespace vleo::test {

template <class T, class F>
struct DeviceEval {
  Kokkos::View<T*> out;
  F f;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const { out(i) = f(i); }
};

// Returns host copies of f(i) for i in [0, n), evaluated on the device.
// F must provide `KOKKOS_INLINE_FUNCTION T operator()(int) const`.
template <class T, class F>
Kokkos::View<T*, Kokkos::HostSpace> eval_on_device(const char* label, int n, const F& f) {
  Kokkos::View<T*> out(label, static_cast<std::size_t>(n));
  Kokkos::parallel_for(label, Kokkos::RangePolicy<>(0, n), DeviceEval<T, F>{out, f});
  Kokkos::fence();
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
}

}  // namespace vleo::test
