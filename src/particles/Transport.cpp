#include "vleo/particles/Transport.hpp"

namespace vleo {

TransportCounters count_particles(const ParticleStorage& p, std::size_t n) {
  std::uint64_t sums[detail::CountKernel::kSlots] = {};
  detail::CountKernel k;
  k.status = p.status;
  k.interactions = p.interactions;
  Kokkos::parallel_reduce("vleo::count_particles", Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, n), k,
                          sums);
  TransportCounters c;
  c.active = sums[static_cast<int>(ParticleStatus::Active)] + sums[static_cast<int>(ParticleStatus::Free)];
  c.exited = sums[static_cast<int>(ParticleStatus::Exited)];
  c.truncated = sums[static_cast<int>(ParticleStatus::Truncated)];
  c.lost_inside_body = sums[static_cast<int>(ParticleStatus::LostInsideBody)];
  c.invalid = sums[static_cast<int>(ParticleStatus::Invalid)];
  c.surface_interactions = sums[kNumParticleStatus];
  c.histories_with_hits = sums[kNumParticleStatus + 1];
  c.max_interactions = sums[kNumParticleStatus + 2];
  c.injected = n;
  return c;
}

}  // namespace vleo
