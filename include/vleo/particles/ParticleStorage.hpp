#pragma once

// Structure-of-arrays particle storage in the default device memory space.
// Every attribute lives in its own contiguous array so that consecutive
// threads access consecutive addresses (coalesced on GPUs).

#include <cstdint>
#include <string>

#include <Kokkos_Core.hpp>

#include "vleo/core/Vec3.hpp"

namespace vleo {

enum class ParticleStatus : std::uint8_t {
  Free = 0,            // slot not in use
  Active = 1,          // being transported
  Exited = 2,          // left the simulation domain (normal end of a history)
  Truncated = 3,       // reached tpmc.max_reflections; history terminated
  LostInsideBody = 4,  // hit a closed surface from the inside (numerical failure)
  Invalid = 5,         // non-finite state or invalid re-emission (numerical failure)
};

inline constexpr int kNumParticleStatus = 6;

const char* to_string(ParticleStatus s);

class ParticleStorage {
 public:
  ParticleStorage() = default;
  explicit ParticleStorage(std::size_t capacity, const std::string& label = "particles");

  std::size_t capacity() const { return x.extent(0); }

  // Sets every slot to status Free and zeroes the counters.
  void clear();

  KOKKOS_INLINE_FUNCTION Vec3 position(std::size_t i) const { return {x(i), y(i), z(i)}; }
  KOKKOS_INLINE_FUNCTION Vec3 velocity(std::size_t i) const { return {vx(i), vy(i), vz(i)}; }
  KOKKOS_INLINE_FUNCTION void set_position(std::size_t i, const Vec3& p) const {
    x(i) = p.x;
    y(i) = p.y;
    z(i) = p.z;
  }
  KOKKOS_INLINE_FUNCTION void set_velocity(std::size_t i, const Vec3& v) const {
    vx(i) = v.x;
    vy(i) = v.y;
    vz(i) = v.z;
  }

  Kokkos::View<double*> x, y, z;     // position [m]
  Kokkos::View<double*> vx, vy, vz;  // velocity [m/s]
  Kokkos::View<double*> weight;      // statistical weight (TPMC: event rate factor [1/s])
  Kokkos::View<std::uint32_t*> species;
  Kokkos::View<std::uint64_t*> particle_id;
  Kokkos::View<std::uint8_t*> status;          // ParticleStatus
  Kokkos::View<std::uint32_t*> interactions;   // surface interactions of the history
};

}  // namespace vleo
