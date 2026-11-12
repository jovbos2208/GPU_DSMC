// Shared Catch2 main: initialises Kokkos (default execution space of the
// build: Serial/OpenMP, CUDA or HIP) for the duration of the test run.
#include <catch2/catch_session.hpp>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  return Catch::Session().run(argc, argv);
}
