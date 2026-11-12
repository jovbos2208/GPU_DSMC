#include "vleo/particles/ParticleStorage.hpp"

namespace vleo {

const char* to_string(ParticleStatus s) {
  switch (s) {
    case ParticleStatus::Free: return "free";
    case ParticleStatus::Active: return "active";
    case ParticleStatus::Exited: return "exited";
    case ParticleStatus::Truncated: return "truncated";
    case ParticleStatus::LostInsideBody: return "lost_inside_body";
    case ParticleStatus::Invalid: return "invalid";
  }
  return "?";
}

ParticleStorage::ParticleStorage(std::size_t capacity, const std::string& label)
    : x(label + "_x", capacity),
      y(label + "_y", capacity),
      z(label + "_z", capacity),
      vx(label + "_vx", capacity),
      vy(label + "_vy", capacity),
      vz(label + "_vz", capacity),
      weight(label + "_weight", capacity),
      species(label + "_species", capacity),
      particle_id(label + "_id", capacity),
      status(label + "_status", capacity),
      interactions(label + "_interactions", capacity) {}

void ParticleStorage::clear() {
  Kokkos::deep_copy(status, static_cast<std::uint8_t>(ParticleStatus::Free));
  Kokkos::deep_copy(interactions, 0u);
}

}  // namespace vleo
