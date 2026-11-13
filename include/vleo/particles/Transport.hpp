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
//   KOKKOS_INLINE_FUNCTION InteractionResult operator()(const SurfaceEvent&, RandomStream&) const;
//   KOKKOS_INLINE_FUNCTION Vec3 wall_velocity(int triangle) const;
// It must return a velocity pointing into the gas (v . n_gas > 0); anything
// else ends the history with status Invalid. Random numbers of the k-th
// interaction (k = 0, 1, ...) of history `id` come from the stream
// (seed, id, k + 1, RngStream::SurfaceInteraction); injection uses event 0.
//
// Event balance (implementation plan, section 5), for every processed
// interaction e of a particle with mass m:
//     dp_e = m (v_i - v_r)                       momentum transferred to the body
//     dl_e = (r_e - r_0) x dp_e                  angular momentum about r_0
//     dE_e = m/2 (|v_i - v_w|^2 - |v_r - v_w|^2) + (e_int,i - e_int,r)
//            translational plus internal (rotational + vibrational) energy to the wall
// Per history the sums are accumulated in registers and written once to the
// history tally array (no atomics, deterministic). Optionally the same event
// values are added atomically to per-triangle surface tallies.

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "vleo/core/InternalEnergy.hpp"
#include "vleo/core/Random.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/particles/Domain.hpp"
#include "vleo/particles/Inflow.hpp"
#include "vleo/particles/ParticleStorage.hpp"

namespace vleo {

struct SurfaceEvent {
  Vec3 position;       // hit point
  Vec3 velocity_in;    // incident velocity (body frame)
  Vec3 normal_gas;     // unit normal pointing into the gas on the side that was hit
  int triangle;        // mesh triangle index (BVH order)
  std::uint32_t species;
  double mass_kg;
  std::uint64_t particle_id;
  std::uint32_t interaction_index;  // 0 for the first interaction of the history
  double e_rot_in;                  // internal energies of the incident particle [J]
  double e_vib_in;
  SpeciesModes modes;               // mass and internal modes of the species
};

struct InteractionResult {
  Vec3 velocity;
  double e_rot;
  double e_vib;
};

// Specular reflection at a wall at rest: reference model for transport tests
// (identical to Maxwell with diffuse_fraction = 0).
struct SpecularInteraction {
  KOKKOS_INLINE_FUNCTION InteractionResult operator()(const SurfaceEvent& e, RandomStream&) const {
    return {reflect(e.velocity_in, e.normal_gas), e.e_rot_in, e.e_vib_in};
  }
  KOKKOS_INLINE_FUNCTION Vec3 wall_velocity(int) const { return Vec3(); }
};

// Layout of the per-history tally array (n_histories x kHistoryTallies).
enum HistoryTally : int {
  kTallyPx = 0, kTallyPy, kTallyPz,  // momentum to the body [kg m/s]
  kTallyLx, kTallyLy, kTallyLz,      // angular momentum about r_0 [kg m^2/s]
  kTallyE,                           // energy to the wall [J]
  kTallyEPos,                        // sum of positive event energies [J]
  kTallyENeg,                        // sum of negative event energies [J]
  kHistoryTallies
};

// Layout of the per-triangle tally array (n_triangles x kSurfaceTallies).
enum SurfaceTally : int {
  kSurfPx = 0, kSurfPy, kSurfPz,
  kSurfE, kSurfEPos, kSurfENeg,
  kSurfHits,
  kSurfaceTallies
};

struct TallyTargets {
  Kokkos::View<double**, Kokkos::LayoutLeft> history;  // empty: no history tallies
  Kokkos::View<double**, Kokkos::LayoutLeft> surface;  // empty: no surface tallies
  Vec3 moment_center;
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
    p.e_rot(i) = s.e_rot;
    p.e_vib(i) = s.e_vib;
    p.status(i) = static_cast<std::uint8_t>(ParticleStatus::Active);
  }
};

// State of one history while it is followed (registers).
struct HistoryState {
  Vec3 pos, vel;
  double e_rot = 0.0, e_vib = 0.0;
  std::uint64_t id = 0;
  std::uint32_t species = 0;
  std::uint32_t k = 0;  // interactions so far
  ParticleStatus status = ParticleStatus::Active;
  Vec3 dp_sum, dl_sum;
  double de_sum = 0.0, de_pos = 0.0, de_neg = 0.0;
};

// Follows a history to its end (shared by the stored-particle and the fused kernels).
template <class Interaction>
KOKKOS_INLINE_FUNCTION void follow_history(HistoryState& h, const DeviceGeometry& geo, const Domain& domain,
                                           const Interaction& interaction, std::uint64_t seed,
                                           std::uint32_t max_interactions,
                                           const Kokkos::View<const double*>& species_mass,
                                           const Kokkos::View<const SpeciesModes*>& species_modes,
                                           const TallyTargets& tally) {
  int ignore = -1;
  const double mass = species_mass.extent(0) > h.species ? species_mass(h.species) : 1.0;
  SpeciesModes modes;
  if (species_modes.extent(0) > h.species) modes = species_modes(h.species);
  modes.mass = mass;
  const bool surface_tally = tally.surface.extent(0) > 0;

  while (h.status == ParticleStatus::Active) {
    if (!is_finite(h.pos) || !is_finite(h.vel) || norm2(h.vel) == 0.0) {
      h.status = ParticleStatus::Invalid;
      break;
    }
    const double t_exit = domain.exit_distance(h.pos, h.vel);
    const RayHit hit = geo.closest_hit(h.pos, h.vel, t_exit, ignore);
    if (!hit.valid()) {
      h.pos = h.pos + t_exit * h.vel;
      h.status = ParticleStatus::Exited;
      break;
    }
    h.pos = h.pos + hit.t * h.vel;
    const Vec3 n = geo.mesh.normal(hit.triangle);
    const bool front = dot(h.vel, n) < 0.0;
    if (!front && geo.mesh.two_sided(hit.triangle) == 0) {
      h.status = ParticleStatus::LostInsideBody;
      break;
    }
    if (h.k >= max_interactions) {
      h.status = ParticleStatus::Truncated;
      break;
    }
    const Vec3 n_gas = front ? n : -n;
    SurfaceEvent ev{h.pos, h.vel, n_gas, hit.triangle, h.species, mass, h.id, h.k, h.e_rot, h.e_vib, modes};
    RandomStream rng(seed, h.id, h.k + 1u, RngStream::SurfaceInteraction);
    const Vec3 v_in = h.vel;
    const double e_int_in = h.e_rot + h.e_vib;
    const InteractionResult res = interaction(ev, rng);
    h.vel = res.velocity;
    h.e_rot = res.e_rot;
    h.e_vib = res.e_vib;
    ++h.k;
    if (!(dot(h.vel, n_gas) > 0.0) || !is_finite(h.vel)) {
      h.status = ParticleStatus::Invalid;
      break;
    }
    // Event balance (accumulated only for valid interactions).
    const Vec3 vw = interaction.wall_velocity(hit.triangle);
    const Vec3 dp = mass * (v_in - h.vel);
    const double de = 0.5 * mass * (norm2(v_in - vw) - norm2(h.vel - vw)) + (e_int_in - h.e_rot - h.e_vib);
    h.dp_sum += dp;
    h.dl_sum += cross(h.pos - tally.moment_center, dp);
    h.de_sum += de;
    (de >= 0.0 ? h.de_pos : h.de_neg) += de;
    if (surface_tally) {
      const int t = hit.triangle;
      Kokkos::atomic_add(&tally.surface(t, kSurfPx), dp.x);
      Kokkos::atomic_add(&tally.surface(t, kSurfPy), dp.y);
      Kokkos::atomic_add(&tally.surface(t, kSurfPz), dp.z);
      Kokkos::atomic_add(&tally.surface(t, kSurfE), de);
      Kokkos::atomic_add(&tally.surface(t, de >= 0.0 ? kSurfEPos : kSurfENeg), de);
      Kokkos::atomic_add(&tally.surface(t, kSurfHits), 1.0);
    }
    h.pos = h.pos + geo.surface_offset_m * n_gas;
    ignore = hit.triangle;
  }
}

KOKKOS_INLINE_FUNCTION void write_history_tally(const TallyTargets& tally, std::size_t i, const HistoryState& h) {
  if (tally.history.extent(0) > i) {
    tally.history(i, kTallyPx) = h.dp_sum.x;
    tally.history(i, kTallyPy) = h.dp_sum.y;
    tally.history(i, kTallyPz) = h.dp_sum.z;
    tally.history(i, kTallyLx) = h.dl_sum.x;
    tally.history(i, kTallyLy) = h.dl_sum.y;
    tally.history(i, kTallyLz) = h.dl_sum.z;
    tally.history(i, kTallyE) = h.de_sum;
    tally.history(i, kTallyEPos) = h.de_pos;
    tally.history(i, kTallyENeg) = h.de_neg;
  }
}

template <class Interaction>
struct TransportKernel {
  ParticleStorage p;
  DeviceGeometry geo;
  Domain domain;
  Interaction interaction;
  std::uint64_t seed;
  std::uint32_t max_interactions;
  Kokkos::View<const double*> species_mass;
  Kokkos::View<const SpeciesModes*> species_modes;
  TallyTargets tally;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    if (p.status(i) != static_cast<std::uint8_t>(ParticleStatus::Active)) return;
    HistoryState h;
    h.pos = p.position(i);
    h.vel = p.velocity(i);
    h.id = p.particle_id(i);
    h.species = p.species(i);
    h.k = p.interactions(i);
    h.e_rot = p.e_rot(i);
    h.e_vib = p.e_vib(i);
    follow_history(h, geo, domain, interaction, seed, max_interactions, species_mass, species_modes, tally);
    p.set_position(i, h.pos);
    p.set_velocity(i, h.vel);
    p.interactions(i) = h.k;
    p.e_rot(i) = h.e_rot;
    p.e_vib(i) = h.e_vib;
    p.status(i) = static_cast<std::uint8_t>(h.status);
    write_history_tally(tally, i, h);
  }
};

// TPMC: injection and transport fused (M5.3). The entering molecule is
// sampled in registers and followed immediately; only the final status, the
// interaction count and the history tallies are stored. Same random streams
// as InjectKernel + TransportKernel, hence bitwise identical results.
template <class Interaction>
struct FusedHistoryKernel {
  Kokkos::View<std::uint8_t*> status;
  Kokkos::View<std::uint32_t*> interactions;
  DeviceInflow inflow;
  DeviceGeometry geo;
  Domain domain;
  Interaction interaction;
  std::uint64_t seed;
  std::uint64_t first_id;
  std::uint32_t max_interactions;
  Kokkos::View<const double*> species_mass;
  Kokkos::View<const SpeciesModes*> species_modes;
  TallyTargets tally;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    HistoryState h;
    h.id = first_id + i;
    {
      RandomStream rng(seed, h.id, 0u, RngStream::Injection);
      const InflowSample s = inflow.sample(rng);
      h.pos = s.position;
      h.vel = s.velocity;
      h.species = s.species;
      h.e_rot = s.e_rot;
      h.e_vib = s.e_vib;
    }
    follow_history(h, geo, domain, interaction, seed, max_interactions, species_mass, species_modes, tally);
    interactions(i) = h.k;
    status(i) = static_cast<std::uint8_t>(h.status);
    write_history_tally(tally, i, h);
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
  Kokkos::View<const double*> species_mass;  // per species index; empty: mass 1 (no physical tallies)
  Kokkos::View<const SpeciesModes*> species_modes;  // per species; empty: no internal energy
  TallyTargets tally;
};

// Transports the Active particles in slots [0, n) of `p` to completion
// (positions, velocities, ids and species must be set) and counts all n slots.
template <class Interaction>
TransportCounters transport_particles(const TransportSetup& s, const Interaction& interaction, ParticleStorage& p,
                                      std::size_t n) {
  using policy = Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>;
  Kokkos::parallel_for("vleo::transport", policy(0, n),
                       detail::TransportKernel<Interaction>{p, s.geometry, s.domain, interaction, s.seed,
                                                            s.max_interactions, s.species_mass, s.species_modes,
                                                            s.tally});
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
