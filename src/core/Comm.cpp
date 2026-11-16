#include "vleo/core/Comm.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef VLEO_HAVE_MPI
#include <mpi.h>
#endif

namespace vleo::comm {

#ifdef VLEO_HAVE_MPI

namespace {

bool g_owner = false;  // this Session called MPI_Init
int g_rank = 0, g_size = 1, g_local_rank = 0;

void check(int rc, const char* what) {
  if (rc != MPI_SUCCESS) throw std::runtime_error(std::string("MPI error in ") + what);
}

bool initialized() {
  int f = 0;
  MPI_Initialized(&f);
  int g = 0;
  MPI_Finalized(&g);
  return f != 0 && g == 0;
}

// MPI counts are int: large buffers are processed in chunks.
constexpr std::size_t kChunk = std::size_t{1} << 28;

}  // namespace

Session::Session(int* argc, char*** argv) {
  if (!initialized()) {
    check(MPI_Init(argc, argv), "MPI_Init");
    g_owner = true;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &g_local_rank);
  MPI_Comm_free(&node);
}

Session::~Session() {
  if (g_owner && initialized()) MPI_Finalize();
  g_owner = false;
}

bool enabled() { return true; }
int rank() { return initialized() ? g_rank : 0; }
int size() { return initialized() ? g_size : 1; }
int local_rank() { return initialized() ? g_local_rank : 0; }

std::string library_version() {
  char buf[MPI_MAX_LIBRARY_VERSION_STRING];
  int len = 0;
  MPI_Get_library_version(buf, &len);
  std::string s(buf, static_cast<std::size_t>(len));
  const auto nl = s.find_first_of("\r\n");
  if (nl != std::string::npos) s.resize(nl);
  while (!s.empty() && (s.back() == ',' || s.back() == ' ' || s.back() == '\0')) s.pop_back();
  return s;
}

void abort(int code) {
  if (initialized() && g_size > 1) MPI_Abort(MPI_COMM_WORLD, code);
  std::exit(code);
}

void barrier() {
  if (size() > 1) MPI_Barrier(MPI_COMM_WORLD);
}

void sum(double* values, std::size_t n) {
  if (size() == 1) return;
  for (std::size_t o = 0; o < n; o += kChunk)
    check(MPI_Allreduce(MPI_IN_PLACE, values + o, static_cast<int>(std::min(kChunk, n - o)), MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD),
          "MPI_Allreduce");
}

void sum(std::uint64_t* values, std::size_t n) {
  if (size() == 1) return;
  for (std::size_t o = 0; o < n; o += kChunk)
    check(MPI_Allreduce(MPI_IN_PLACE, values + o, static_cast<int>(std::min(kChunk, n - o)), MPI_UINT64_T, MPI_SUM,
                        MPI_COMM_WORLD),
          "MPI_Allreduce");
}

double max(double value) {
  if (size() > 1) check(MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce");
  return value;
}

double min(double value) {
  if (size() > 1) check(MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce");
  return value;
}

bool any(bool value) {
  int v = value ? 1 : 0;
  if (size() > 1) check(MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD), "MPI_Allreduce");
  return v != 0;
}

std::vector<char> allgather(const void* data, std::size_t bytes) {
  std::vector<char> out(bytes * static_cast<std::size_t>(size()));
  if (size() == 1) {
    if (bytes > 0) std::memcpy(out.data(), data, bytes);
    return out;
  }
  if (bytes > static_cast<std::size_t>(INT_MAX)) throw std::runtime_error("comm::allgather: block too large");
  check(MPI_Allgather(data, static_cast<int>(bytes), MPI_BYTE, out.data(), static_cast<int>(bytes), MPI_BYTE,
                      MPI_COMM_WORLD),
        "MPI_Allgather");
  return out;
}

std::vector<int> allgather(int value) {
  const std::vector<char> b = allgather(&value, sizeof(int));
  std::vector<int> v(static_cast<std::size_t>(size()));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

std::vector<std::string> allgather(const std::string& value) {
  const std::vector<int> len = allgather(static_cast<int>(value.size()));
  std::size_t width = 0;
  for (int l : len) width = std::max(width, static_cast<std::size_t>(l));
  std::string padded = value;
  padded.resize(width, '\0');
  const std::vector<char> all = allgather(padded.data(), width);
  std::vector<std::string> out;
  for (std::size_t k = 0; k < len.size(); ++k)
    out.emplace_back(all.data() + k * width, static_cast<std::size_t>(len[k]));
  return out;
}

std::vector<std::vector<char>> alltoallv(const std::vector<std::vector<char>>& send) {
  const int p = size();
  if (static_cast<int>(send.size()) != p) throw std::invalid_argument("comm::alltoallv: one block per rank required");
  if (p == 1) return send;
  std::vector<int> scount(static_cast<std::size_t>(p)), rcount(static_cast<std::size_t>(p));
  std::vector<int> sdispl(static_cast<std::size_t>(p)), rdispl(static_cast<std::size_t>(p));
  std::size_t stotal = 0;
  for (int k = 0; k < p; ++k) {
    sdispl[static_cast<std::size_t>(k)] = static_cast<int>(stotal);
    scount[static_cast<std::size_t>(k)] = static_cast<int>(send[static_cast<std::size_t>(k)].size());
    stotal += send[static_cast<std::size_t>(k)].size();
  }
  if (stotal > static_cast<std::size_t>(INT_MAX)) throw std::runtime_error("comm::alltoallv: message too large");
  check(MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD), "MPI_Alltoall");
  std::size_t rtotal = 0;
  for (int k = 0; k < p; ++k) {
    rdispl[static_cast<std::size_t>(k)] = static_cast<int>(rtotal);
    rtotal += static_cast<std::size_t>(rcount[static_cast<std::size_t>(k)]);
  }
  if (rtotal > static_cast<std::size_t>(INT_MAX)) throw std::runtime_error("comm::alltoallv: message too large");
  std::vector<char> sbuf(stotal), rbuf(rtotal);
  for (int k = 0; k < p; ++k) {
    const auto& b = send[static_cast<std::size_t>(k)];
    if (!b.empty()) std::memcpy(sbuf.data() + sdispl[static_cast<std::size_t>(k)], b.data(), b.size());
  }
  check(MPI_Alltoallv(sbuf.data(), scount.data(), sdispl.data(), MPI_BYTE, rbuf.data(), rcount.data(), rdispl.data(),
                      MPI_BYTE, MPI_COMM_WORLD),
        "MPI_Alltoallv");
  std::vector<std::vector<char>> recv(static_cast<std::size_t>(p));
  for (int k = 0; k < p; ++k) {
    const auto off = static_cast<std::size_t>(rdispl[static_cast<std::size_t>(k)]);
    recv[static_cast<std::size_t>(k)].assign(rbuf.begin() + static_cast<std::ptrdiff_t>(off),
                                             rbuf.begin() + static_cast<std::ptrdiff_t>(off) +
                                                 rcount[static_cast<std::size_t>(k)]);
  }
  return recv;
}

std::vector<std::size_t> exchange_counts(const std::vector<std::size_t>& send_bytes) {
  const int p = size();
  if (static_cast<int>(send_bytes.size()) != p) throw std::invalid_argument("comm::exchange_counts: one count per rank");
  if (p == 1) return send_bytes;
  std::vector<std::uint64_t> s(send_bytes.begin(), send_bytes.end()), r(static_cast<std::size_t>(p));
  check(MPI_Alltoall(s.data(), 1, MPI_UINT64_T, r.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD), "MPI_Alltoall");
  return std::vector<std::size_t>(r.begin(), r.end());
}

void alltoallv(const void* send, const std::vector<std::size_t>& send_bytes, void* recv,
               const std::vector<std::size_t>& recv_bytes) {
  const int p = size();
  if (p == 1) {
    if (send_bytes[0] > 0) std::memcpy(recv, send, send_bytes[0]);
    return;
  }
  std::vector<int> sc(static_cast<std::size_t>(p)), sd(static_cast<std::size_t>(p)), rc(static_cast<std::size_t>(p)),
      rd(static_cast<std::size_t>(p));
  std::size_t so = 0, ro = 0;
  for (std::size_t k = 0; k < static_cast<std::size_t>(p); ++k) {
    if (so + send_bytes[k] > static_cast<std::size_t>(INT_MAX) || ro + recv_bytes[k] > static_cast<std::size_t>(INT_MAX))
      throw std::runtime_error("comm::alltoallv: message too large");
    sc[k] = static_cast<int>(send_bytes[k]);
    sd[k] = static_cast<int>(so);
    rc[k] = static_cast<int>(recv_bytes[k]);
    rd[k] = static_cast<int>(ro);
    so += send_bytes[k];
    ro += recv_bytes[k];
  }
  check(MPI_Alltoallv(send, sc.data(), sd.data(), MPI_BYTE, recv, rc.data(), rd.data(), MPI_BYTE, MPI_COMM_WORLD),
        "MPI_Alltoallv");
}

void broadcast(void* data, std::size_t bytes) {
  if (size() == 1) return;
  auto* p = static_cast<char*>(data);
  for (std::size_t o = 0; o < bytes; o += kChunk)
    check(MPI_Bcast(p + o, static_cast<int>(std::min(kChunk, bytes - o)), MPI_BYTE, 0, MPI_COMM_WORLD), "MPI_Bcast");
}

#else  // ------------------------------------------------------------ no MPI

Session::Session(int*, char***) {}
Session::~Session() = default;

bool enabled() { return false; }
int rank() { return 0; }
int size() { return 1; }
int local_rank() { return 0; }
std::string library_version() { return ""; }
void abort(int code) { std::exit(code); }
void barrier() {}
void sum(double*, std::size_t) {}
void sum(std::uint64_t*, std::size_t) {}
double max(double value) { return value; }
double min(double value) { return value; }
bool any(bool value) { return value; }

std::vector<char> allgather(const void* data, std::size_t bytes) {
  std::vector<char> out(bytes);
  if (bytes > 0) std::memcpy(out.data(), data, bytes);
  return out;
}

std::vector<int> allgather(int value) { return {value}; }

std::vector<std::string> allgather(const std::string& value) { return {value}; }

std::vector<std::vector<char>> alltoallv(const std::vector<std::vector<char>>& send) {
  if (send.size() != 1) throw std::invalid_argument("comm::alltoallv: one block per rank required");
  return send;
}

std::vector<std::size_t> exchange_counts(const std::vector<std::size_t>& send_bytes) { return send_bytes; }

void alltoallv(const void* send, const std::vector<std::size_t>& send_bytes, void* recv,
               const std::vector<std::size_t>&) {
  if (!send_bytes.empty() && send_bytes[0] > 0) std::memcpy(recv, send, send_bytes[0]);
}

void broadcast(void*, std::size_t) {}

#endif

}  // namespace vleo::comm
