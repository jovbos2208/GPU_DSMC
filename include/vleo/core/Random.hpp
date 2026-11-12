#pragma once

// Counter-based random numbers (Philox4x32-10, Salmon et al., SC'11).
//
// A RandomStream is a pure function of
//     (seed, particle_id, event_id, stream_id, draw index)
// and carries no global state. Results are therefore independent of thread
// scheduling, kernel launch configuration and backend (CPU / CUDA / HIP):
// every particle history can be replayed bit-for-bit in isolation.
//
// Philox input mapping (key = 64 bit, counter = 128 bit):
//     key[0..1] = seed (low, high 32 bits)
//     ctr[0..1] = particle_id (low, high 32 bits)
//     ctr[2]    = event_id
//     ctr[3]    = (stream_id << 16) | block
// Each block yields four 32-bit words, so one (particle, event, stream)
// triple provides 2^16 blocks = 262144 words (131072 doubles) before the
// stream is exhausted. Exhaustion is reported via exhausted(); further draws
// wrap around and repeat the sequence, which callers must avoid (choose a new
// event_id instead).
//
// Floating-point conversions are exact integer -> double mappings, so uniform
// variates are bitwise identical on all backends. normal() uses log/sqrt/
// cos/sin and may differ in the last ulp between device math libraries.

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "vleo/core/PhysicalConstants.hpp"

namespace vleo {

struct Philox4x32x10 {
  struct Counter {
    std::uint32_t v[4];
  };
  struct Key {
    std::uint32_t v[2];
  };

  static constexpr std::uint32_t kMul0 = 0xD2511F53u;
  static constexpr std::uint32_t kMul1 = 0xCD9E8D57u;
  static constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;  // golden ratio
  static constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;  // sqrt(3) - 1
  static constexpr int kRounds = 10;

  KOKKOS_INLINE_FUNCTION static constexpr Counter round(const Counter& c, const Key& k) {
    const std::uint64_t p0 = static_cast<std::uint64_t>(kMul0) * c.v[0];
    const std::uint64_t p1 = static_cast<std::uint64_t>(kMul1) * c.v[2];
    const auto hi0 = static_cast<std::uint32_t>(p0 >> 32);
    const auto lo0 = static_cast<std::uint32_t>(p0);
    const auto hi1 = static_cast<std::uint32_t>(p1 >> 32);
    const auto lo1 = static_cast<std::uint32_t>(p1);
    return Counter{{hi1 ^ c.v[1] ^ k.v[0], lo1, hi0 ^ c.v[3] ^ k.v[1], lo0}};
  }

  KOKKOS_INLINE_FUNCTION static constexpr Counter generate(Counter c, Key k) {
    c = round(c, k);
    for (int r = 1; r < kRounds; ++r) {
      k.v[0] += kWeyl0;
      k.v[1] += kWeyl1;
      c = round(c, k);
    }
    return c;
  }
};

// Stream identifiers separate the random numbers consumed by different
// physical processes of the same particle and event.
enum class RngStream : std::uint16_t {
  Injection = 1,
  SurfaceInteraction = 2,
  GasCollision = 3,
  CollisionPairSelection = 4,
  InjectionCount = 5,  // DSMC: number of entering particles per (step, channel)
  Test = 0xFFFF,
};

class RandomStream {
 public:
  static constexpr std::uint32_t kBlocksPerStream = 1u << 16;

  KOKKOS_INLINE_FUNCTION RandomStream(std::uint64_t seed, std::uint64_t particle_id, std::uint32_t event_id,
                                      std::uint16_t stream_id)
      : key_{{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)}},
        ctr_{{static_cast<std::uint32_t>(particle_id), static_cast<std::uint32_t>(particle_id >> 32), event_id,
              static_cast<std::uint32_t>(stream_id) << 16}} {}

  KOKKOS_INLINE_FUNCTION RandomStream(std::uint64_t seed, std::uint64_t particle_id, std::uint32_t event_id,
                                      RngStream stream)
      : RandomStream(seed, particle_id, event_id, static_cast<std::uint16_t>(stream)) {}

  // Next raw 32-bit word.
  KOKKOS_INLINE_FUNCTION std::uint32_t next_u32() {
    if (pos_ == 4) refill();
    return buf_.v[pos_++];
  }

  KOKKOS_INLINE_FUNCTION std::uint64_t next_u64() {
    const std::uint64_t hi = next_u32();
    const std::uint64_t lo = next_u32();
    return (hi << 32) | lo;
  }

  // Uniform on [0, 1) with 53 random bits: k * 2^-53, k in [0, 2^53 - 1].
  KOKKOS_INLINE_FUNCTION double uniform() {
    const std::uint32_t a = next_u32();
    const std::uint32_t b = next_u32();
    return uniform_from_words(a, b);
  }

  // Uniform on (0, 1]: safe argument for log(). Smallest value 2^-53.
  KOKKOS_INLINE_FUNCTION double uniform_pos() { return 1.0 - uniform(); }

  // Uniform on the open interval (0, 1): (2k + 1) 2^-53, k in [0, 2^52 - 1],
  // i.e. values in [2^-53, 1 - 2^-53]. Both log(u) and log(1 - u) are finite
  // and non-zero.
  KOKKOS_INLINE_FUNCTION double uniform_open() {
    const std::uint32_t a = next_u32();
    const std::uint32_t b = next_u32();
    return uniform_open_from_words(a, b);
  }

  // Standard normal N(0, 1) via the Box-Muller transform. Values are produced
  // in pairs; the second one is cached. |z| <= sqrt(-2 ln 2^-53) ~= 8.57.
  KOKKOS_INLINE_FUNCTION double normal() {
    if (has_cached_normal_) {
      has_cached_normal_ = false;
      return cached_normal_;
    }
    const double u_pos = uniform_pos();
    const double u = uniform();
    double z0 = 0.0;
    box_muller(u_pos, u, z0, cached_normal_);
    has_cached_normal_ = true;
    return z0;
  }

  // --- Pure conversion functions (exposed for testing edge cases) ----------

  // Maps two 32-bit words to k * 2^-53 with k = (a >> 5) * 2^26 + (b >> 6),
  // i.e. a uniform double on [0, 1 - 2^-53]. Exact on every backend.
  KOKKOS_INLINE_FUNCTION static constexpr double uniform_from_words(std::uint32_t a, std::uint32_t b) {
    return (static_cast<double>(a >> 5) * 67108864.0 + static_cast<double>(b >> 6)) * 0x1.0p-53;
  }

  // (2k + 1) 2^-53 with the 52-bit k = (a >> 6) 2^26 + (b >> 6). Exact.
  KOKKOS_INLINE_FUNCTION static constexpr double uniform_open_from_words(std::uint32_t a, std::uint32_t b) {
    const double k = static_cast<double>(a >> 6) * 67108864.0 + static_cast<double>(b >> 6);
    return (2.0 * k + 1.0) * 0x1.0p-53;
  }

  // Box-Muller transform. u_pos in (0, 1], u in [0, 1).
  KOKKOS_INLINE_FUNCTION static void box_muller(double u_pos, double u, double& z0, double& z1) {
    const double r = Kokkos::sqrt(-2.0 * Kokkos::log(u_pos));
    const double theta = constants::two_pi * u;
    z0 = r * Kokkos::cos(theta);
    z1 = r * Kokkos::sin(theta);
  }

  // True once more than 2^16 blocks were requested, i.e. the sequence of this
  // (particle, event, stream) has started repeating.
  KOKKOS_INLINE_FUNCTION bool exhausted() const { return exhausted_; }

  // Number of Philox blocks generated so far.
  KOKKOS_INLINE_FUNCTION std::uint32_t blocks_used() const { return blocks_used_; }

 private:
  KOKKOS_INLINE_FUNCTION void refill() {
    Philox4x32x10::Counter c = ctr_;
    c.v[3] |= (blocks_used_ & 0xFFFFu);
    buf_ = Philox4x32x10::generate(c, key_);
    pos_ = 0;
    ++blocks_used_;
    if (blocks_used_ > kBlocksPerStream) exhausted_ = true;
  }

  Philox4x32x10::Key key_;
  Philox4x32x10::Counter ctr_;  // ctr_.v[3] holds stream_id << 16, block bits zero
  Philox4x32x10::Counter buf_{{0u, 0u, 0u, 0u}};
  int pos_ = 4;
  std::uint32_t blocks_used_ = 0;
  bool exhausted_ = false;
  bool has_cached_normal_ = false;
  double cached_normal_ = 0.0;
};

}  // namespace vleo
