#pragma once

// Free-stream inflow through the six faces of the domain box.
//
// For a drifting Maxwellian (number density n, temperature T, bulk velocity
// u) the number flux of species with mass m through a face with inward unit
// normal e is
//     Gamma = n c / (2 sqrt(pi)) * F(s),   F(s) = exp(-s^2) + sqrt(pi) s (1 + erf s),
// with the most probable thermal speed c = sqrt(2 k T / m) and the speed ratio
// s = (u . e) / c. Entering molecules have the flux-weighted normal velocity
// distribution  f(z) ~ z exp(-(z - s)^2), z = v_n / c > 0, and Maxwellian
// tangential components (mean u_t, standard deviation c / sqrt 2).
//
// Each (face, species) pair is a channel selected with probability
// Gamma_channel * A_face / sum. The normal speed is drawn with an exact
// rejection sampler (valid for every s, see sample_flux_normal()).

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "vleo/core/InternalEnergy.hpp"
#include "vleo/core/Random.hpp"
#include "vleo/io/Config.hpp"
#include "vleo/particles/Domain.hpp"

namespace vleo {

// F(s) above, evaluated without catastrophic cancellation for s << 0.
double flux_factor(double s);

// Number flux [1/(m^2 s)] for number density n, thermal speed c, speed ratio s.
double number_flux(double n, double c, double s);

// CDF of the dimensionless flux-weighted normal speed z (used by tests).
double flux_normal_cdf(double z, double s);

// Exact sampler for f(z) ~ z exp(-(z - s)^2), z > 0.
//  s >= 0: proposal h(z) = (|z - s| + s) exp(-(z - s)^2) >= f(z), a mixture of
//          |w| exp(-w^2) and s exp(-w^2) (w = z - s > -s), accept z / (|w| + s).
//          Acceptance >= 0.73 for all s >= 0 (minimum near s = 1.2).
//  s <  0: proposal (z - s) exp(-(z - s)^2) >= f(z), sampled by inversion,
//          accept z / (z - s). Acceptance 0.45 at s = -0.5, ~1/(2 s^2) for s << 0;
//          such faces receive a vanishing share of the particles.
// p_a = mass of the |w| component / total proposal mass and exp_s2 = exp(-s^2)
// are precomputed per channel.
KOKKOS_INLINE_FUNCTION double sample_flux_normal(double s, double p_a, double exp_s2, RandomStream& rng) {
  constexpr double inv_sqrt2 = 0.70710678118654752440;
  if (s < 0.0) {
    for (;;) {
      const double w = Kokkos::sqrt(s * s - Kokkos::log(rng.uniform_pos()));  // w >= |s|
      const double z = w + s;
      if (rng.uniform() * w < z) return z;
    }
  }
  const double mass_a = 1.0 - 0.5 * exp_s2;  // integral of |w| e^{-w^2} over w > -s
  for (;;) {
    double w;
    if (rng.uniform() < p_a) {
      if (rng.uniform() * mass_a < 0.5) {
        w = Kokkos::sqrt(-Kokkos::log(rng.uniform_pos()));  // w >= 0
      } else {
        // w in (-s, 0]: r = -w has density r e^{-r^2} on [0, s)
        w = -Kokkos::sqrt(-Kokkos::log(1.0 - rng.uniform() * (1.0 - exp_s2)));
      }
    } else {
      do {
        w = rng.normal() * inv_sqrt2;  // N(0, 1/2)  <->  e^{-w^2}
      } while (w <= -s);
    }
    const double z = w + s;
    if (z > 0.0 && rng.uniform() * (Kokkos::fabs(w) + s) < z) return z;
  }
}

struct InflowChannel {
  int face = 0;               // 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z (face at lo/hi of the axis)
  std::uint32_t species = 0;  // index into GasSection::species
  double rate_per_s = 0.0;    // molecules per second entering through this face
  double thermal_speed = 0.0; // c = sqrt(2 k T / m)
  double s = 0.0;             // speed ratio along the inward normal
  double p_a = 0.0;           // sampler parameter (s >= 0)
  double exp_s2 = 0.0;
  // internal energy of entering molecules (equilibrium at the free-stream temperature)
  int rot_dof = 0;
  int vib = 0;
  double theta_v = 0.0;
  double temperature = 0.0;
};

struct InflowModel {
  std::vector<InflowChannel> channels;
  double total_rate_per_s = 0.0;
  Domain domain;
  Vec3 bulk_velocity;

  static InflowModel create(const Domain& domain, const GasSection& gas);
};

struct InflowSample {
  Vec3 position;
  Vec3 velocity;
  std::uint32_t species;
  double e_rot = 0.0;
  double e_vib = 0.0;
};

struct DeviceInflow {
  Kokkos::View<const double*> cdf;  // cumulative channel rates
  Kokkos::View<const InflowChannel*> channels;
  int num_channels = 0;
  double total_rate = 0.0;
  Domain domain;
  Vec3 bulk_velocity;

  KOKKOS_INLINE_FUNCTION InflowSample sample(RandomStream& rng) const {
    const double x = rng.uniform() * total_rate;
    int ch = num_channels - 1;
    for (int i = 0; i < num_channels; ++i) {
      if (x < cdf(i)) {
        ch = i;
        break;
      }
    }
    // Guard against selecting a trailing zero-rate channel through rounding.
    while (ch > 0 && channels(ch).rate_per_s == 0.0) --ch;
    return sample_channel(ch, rng);
  }

  // Sample entering through channel `ch` (used by DSMC, which draws the
  // number of entering particles per channel and time step itself).
  KOKKOS_INLINE_FUNCTION InflowSample sample_channel(int ch, RandomStream& rng) const {
    const InflowChannel& c = channels(ch);

    const int axis = c.face / 2;
    const double sign = (c.face % 2 == 0) ? 1.0 : -1.0;
    const Aabb& box = domain.box;
    InflowSample out;
    out.species = c.species;
    const double z = sample_flux_normal(c.s, c.p_a, c.exp_s2, rng);
    const double sigma_t = c.thermal_speed * 0.70710678118654752440;
    for (int k = 0; k < 3; ++k) {
      if (k == axis) {
        out.velocity[k] = sign * z * c.thermal_speed;
        out.position[k] = sign > 0.0 ? box.lo[k] : box.hi[k];
      } else {
        out.velocity[k] = bulk_velocity[k] + sigma_t * rng.normal();
        out.position[k] = box.lo[k] + (box.hi[k] - box.lo[k]) * rng.uniform();
      }
    }
    out.e_rot = sample_rotational_energy(c.rot_dof, c.temperature, rng);
    out.e_vib = sample_vibrational_energy(c.vib, c.theta_v, c.temperature, rng);
    return out;
  }
};

DeviceInflow upload(const InflowModel& model);

}  // namespace vleo
