#pragma once

// Collisionless particle transport (TPMC histories) through the domain with
// surface interactions and multiple reflections.
//
// One thread follows one history from injection to its end:
//   loop:
//     t_exit = distance to the domain boundary
//     hit    = closest surface hit with 0 < t <= t_exit (BVH)
//     no hit            -> status Exited (position on the boundary)
//     back-face hit on a closed (one-sided) surface -> LostInsideBody
//     interactions == max_interactions -> Truncated (the hit is not processed)
//     otherwise: new velocity from the interaction model, origin offset by
//       surface_offset along the gas-side normal, triangle excluded from the
//       next search, interactions += 1
//
// The interaction model is a template parameter with
//   KOKKOS_INLINE_FUNCTION Vec3 operator()(const SurfaceEvent&, RandomStream&) const;
// It must return a velocity pointing into the gas (v . n_gas > 0); anything
// else ends the history with status Invalid. Random numbers of the k-th
// interaction (k = 0, 1, ...) of history `id` come from the stream
// (seed, id, k + 1, RngStream::SurfaceInteraction); injection uses event 0.

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "vleo/core/Random.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/particles/Domain.hpp"
#include "vleo/particles/Inflow.hpp"
#include "vleo/particles/ParticleStorage.hpp"

namespace vleo {

struct SurfaceEvent {
  Vec3 position;       // hit point
  Vec3 velocity_in;    // incident velocity (wall at rest)
  Vec3 normal_gas;     // unit normal pointing into the gas on the side that was hit
  int triangle;        // mesh triangle index (BVH order)
  std::uint32_t species;
  std::uint64_t particle_id;
  std::uint32_t interaction_index;  // 0 for the first interaction of the history
};

// Specular reflection: reference model for transport tests (identical to
// Maxwell with diffuse_fraction = 0). The physical models follow in M2.
struct SpecularInteraction {
  KOKKOS_INLINE_FUNCTION Vec3 operator()(const SurfaceEvent& e, RandomStream&) const {
    return reflect(e.velocity_in, e.normal_gas);
  }
};

// Result counters of one transport launch. Every history ends in exactly one
// final state, so injected == exited + truncated + lost_inside_body + invalid
// holds by construction; active must be zero.
struct TransportCounters {
  std::uint64_t injected = 0;
  std::uint64_t active = 0;
  std::uint64_t exited = 0;
  std::uint64_t truncated = 0;
  std::uint64_t lost_inside_body = 0;
  std::uint64_t invalid = 0;
  std::uint64_t surface_interactions = 0;  // processed interactions (all histories)
  std::uint64_t histories_with_hits = 0;   // histories with >= 1 interaction
  std::uint64_t max_interactions = 0;      // largest per-history count

  bool balanced() const {
    return active == 0 && injected == exited + truncated + lost_inside_body + invalid;
  }
  TransportCounters& operator+=(const TransportCounters& o) {
    injected += o.injected;
    active += o.active;
    exited += o.exited;
    truncated += o.truncated;
    lost_inside_body += o.lost_inside_body;
    invalid += o.invalid;
    surface_interactions += o.surface_interactions;
    histories_with_hits += o.histories_with_hits;
    max_interactions = o.max_interactions > max_interactions ? o.max_interactions : max_interactions;
    return *this;
  }
};

namespace detail {

// Fills slots [0, n) with new histories first_id + i.
struct InjectKernel {
  ParticleStorage p;
  DeviceInflow inflow;
  std::uint64_t seed;
  std::uint64_t first_id;
  double weight;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const std::uint64_t id = first_id + i;
    RandomStream rng(seed, id, 0u, RngStream::Injection);
    const InflowSample s = inflow.sample(rng);
    p.set_position(i, s.position);
    p.set_velocity(i, s.velocity);
    p.species(i) = s.species;
    p.particle_id(i) = id;
    p.weight(i) = weight;
    p.interactions(i) = 0u;
    p.status(i) = static_cast<std::uint8_t>(ParticleStatus::Active);
  }
};

template <class Interaction>
struct TransportKernel {
  ParticleStorage p;
  DeviceGeometry geo;
  Domain domain;
  Interaction interaction;
  std::uint64_t seed;
  std::uint32_t max_interactions;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    if (p.status(i) != static_cast<std::uint8_t>(ParticleStatus::Active)) return;
    Vec3 pos = p.position(i);
    Vec3 vel = p.velocity(i);
    const std::uint64_t id = p.particle_id(i);
    const std::uint32_t species = p.species(i);
    std::uint32_t k = p.interactions(i);
    int ignore = -1;
    ParticleStatus status = ParticleStatus::Active;

    while (status == ParticleStatus::Active) {
      if (!is_finite(pos) || !is_finite(vel) || norm2(vel) == 0.0) {
        status = ParticleStatus::Invalid;
        break;
      }
      const double t_exit = domain.exit_distance(pos, vel);
      const RayHit hit = geo.closest_hit(pos, vel, t_exit, ignore);
      if (!hit.valid()) {
        pos = pos + t_exit * vel;
        status = ParticleStatus::Exited;
        break;
      }
      pos = pos + hit.t * vel;
      const Vec3 n = geo.mesh.normal(hit.triangle);
      const bool front = dot(vel, n) < 0.0;
      if (!front && geo.mesh.two_sided(hit.triangle) == 0) {
        status = ParticleStatus::LostInsideBody;
        break;
      }
      if (k >= max_interactions) {
        status = ParticleStatus::Truncated;
        break;
      }
      const Vec3 n_gas = front ? n : -n;
      SurfaceEvent ev{pos, vel, n_gas, hit.triangle, species, id, k};
      RandomStream rng(seed, id, k + 1u, RngStream::SurfaceInteraction);
      vel = interaction(ev, rng);
      ++k;
      if (!(dot(vel, n_gas) > 0.0)) {
        status = ParticleStatus::Invalid;
        break;
      }
      pos = pos + geo.surface_offset_m * n_gas;
      ignore = hit.triangle;
    }
    p.set_position(i, pos);
    p.set_velocity(i, vel);
    p.interactions(i) = k;
    p.status(i) = static_cast<std::uint8_t>(status);
  }
};

// Array reduction over the final particle states.
struct CountKernel {
  using value_type = std::uint64_t[];
  using size_type = std::size_t;
  static constexpr int kSlots = kNumParticleStatus + 3;  // statuses, interactions, with_hits, max
  size_type value_count = kSlots;

  Kokkos::View<const std::uint8_t*> status;
  Kokkos::View<const std::uint32_t*> interactions;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, value_type sums) const {
    const int s = status(i) < kNumParticleStatus ? status(i) : static_cast<int>(ParticleStatus::Invalid);
    sums[s] += 1;
    const std::uint64_t k = interactions(i);
    sums[kNumParticleStatus] += k;
    sums[kNumParticleStatus + 1] += k > 0 ? 1 : 0;
    if (k > sums[kNumParticleStatus + 2]) sums[kNumParticleStatus + 2] = k;
  }
  KOKKOS_INLINE_FUNCTION void init(value_type sums) const {
    for (int k = 0; k < kSlots; ++k) sums[k] = 0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (int k = 0; k < kSlots - 1; ++k) dst[k] += src[k];
    if (src[kSlots - 1] > dst[kSlots - 1]) dst[kSlots - 1] = src[kSlots - 1];
  }
};

}  // namespace detail

// Counts the final states of slots [0, n).
TransportCounters count_particles(const ParticleStorage& p, std::size_t n);

struct TransportSetup {
  DeviceGeometry geometry;
  Domain domain;
  DeviceInflow inflow;
  std::uint64_t seed = 0;
  std::uint32_t max_interactions = 100;
  double weight = 1.0;  // stored per particle
};

// Transports the Active particles in slots [0, n) of `p` to completion
// (positions, velocities, ids and species must be set) and counts all n slots.
template <class Interaction>
TransportCounters transport_particles(const TransportSetup& s, const Interaction& interaction, ParticleStorage& p,
                                      std::size_t n) {
  using policy = Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>;
  Kokkos::parallel_for("vleo::transport", policy(0, n),
                       detail::TransportKernel<Interaction>{p, s.geometry, s.domain, interaction, s.seed,
                                                            s.max_interactions});
  return count_particles(p, n);
}

// Injects histories [first_id, first_id + n) into slots [0, n) of `p`
// (n <= capacity), transports them to completion and returns the counters.
template <class Interaction>
TransportCounters run_histories(const TransportSetup& s, const Interaction& interaction, ParticleStorage& p,
                                std::uint64_t first_id, std::size_t n) {
  using policy = Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>;
  Kokkos::parallel_for("vleo::inject", policy(0, n),
                       detail::InjectKernel{p, s.inflow, s.seed, first_id, s.weight});
  return transport_particles(s, interaction, p, n);
}

}  // namespace vleo
