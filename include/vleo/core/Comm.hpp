#pragma once

// Minimal MPI layer (M6.1). One MPI rank per GPU. Without VLEO_HAVE_MPI (the
// default build) every function is a no-op for a single rank, so the solver
// code is written once for both cases.
//
// All functions except the queries (enabled, rank, size, local_rank, root,
// library_version) are collective: every rank must call them in the same
// order. Data is exchanged through host memory (no GPU-aware MPI required).
//
// Determinism: the reductions are performed by MPI; the solvers only use them
// where the result is exact (integer counts, a sum with a single non-zero
// contribution) or where rounding differences are acceptable (sums of
// floating-point tallies, docs/parallel.md).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vleo::comm {

// Initialises MPI (if compiled in) and finalises it on destruction. Create one
// instance at the start of main(), before Kokkos.
class Session {
 public:
  Session(int* argc, char*** argv);
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
};

bool enabled();  // compiled with MPI
int rank();
int size();
int local_rank();  // rank on this node (for the GPU assignment)
inline bool root() { return rank() == 0; }
std::string library_version();  // MPI library identification ("" without MPI)

// Aborts all ranks (after an error on one rank that the others cannot see).
[[noreturn]] void abort(int code);

void barrier();
// In-place reductions over all ranks.
void sum(double* values, std::size_t n);
void sum(std::uint64_t* values, std::size_t n);
double max(double value);
double min(double value);
inline double sum(double value) {
  sum(&value, 1);
  return value;
}
inline std::uint64_t sum(std::uint64_t value) {
  sum(&value, 1);
  return value;
}
// Logical OR / AND over all ranks.
bool any(bool value);
inline bool all(bool value) { return !any(!value); }
// Every rank contributes `bytes` bytes; returns the concatenation in rank order.
std::vector<char> allgather(const void* data, std::size_t bytes);
std::vector<int> allgather(int value);
std::vector<std::string> allgather(const std::string& value);  // strings of any length
// Personalised all-to-all exchange of byte blocks: send[k] goes to rank k.
// Returns the received blocks, recv[k] from rank k.
std::vector<std::vector<char>> alltoallv(const std::vector<std::vector<char>>& send);
// Broadcast from rank 0.
void broadcast(void* data, std::size_t bytes);

}  // namespace vleo::comm
