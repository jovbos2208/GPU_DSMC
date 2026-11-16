// Shared Catch2 main: initialises MPI (MPI builds; tests started without
// mpiexec run as a single rank) and Kokkos (default execution space of the
// build: Serial/OpenMP, CUDA or HIP) for the duration of the test run.
#include <catch2/catch_session.hpp>

#include <Kokkos_Core.hpp>

#include "vleo/core/Comm.hpp"

int main(int argc, char* argv[]) {
  vleo::comm::Session mpi(&argc, &argv);
  int rc = 0;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    rc = Catch::Session().run(argc, argv);
  }
  // a failure on any rank fails the test on every rank
  return vleo::comm::any(rc != 0) ? 1 : 0;
}
