#pragma once

// Gas-surface interaction kernels (Maxwell and Cercignani-Lampis-Lord).
//
// One implementation for TPMC and DSMC on every backend. Everything is
// evaluated in the frame moving with the wall:
//     c_i = v_i - v_w,   c_r = (scattered relative velocity),   v_r = c_r + v_w
// in the local orthonormal basis (t1, t2, n) with n the unit normal pointing
// into the gas (incident particles have c_i . n < 0). With s = sqrt(k T_w / m):
//
// Maxwell (diffuse_fraction f_d, R ~ U[0,1)):
//   R >= f_d: specular   c_r = c_i - 2 (c_i . n) n
//   R <  f_d: diffuse    c_r,t1 = s Z1, c_r,t2 = s Z2, c_r,n = s sqrt(-2 ln R1),  R1 ~ U(0,1)
//
// CLL (alpha_n in [0,1] normal energy, sigma_t in [0,2] tangential momentum
// accommodation; tangential energy accommodation alpha_t = sigma_t (2 - sigma_t)):
//   c_r,t = (1 - sigma_t) c_i,t + s sqrt(sigma_t (2 - sigma_t)) (Z1, Z2)
//   c_r,n = sqrt( (sqrt(1 - alpha_n) |c_i,n| + s sqrt(alpha_n) Z3)^2 + s^2 alpha_n Z4^2 )
// Limits: CLL(0, 0) = specular, CLL(1, 1) = diffuse (same distribution as Maxwell f_d = 1).
//
// Random numbers per interaction: Maxwell draws R first, then (diffuse
// branch) Z1, Z2 (one Box-Muller pair) and R1; CLL draws Z1..Z4 (two pairs).

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "vleo/core/Frames.hpp"
#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Random.hpp"
#include "vleo/core/Vec3.hpp"
#include "vleo/gsi/GSIModel.hpp"

namespace vleo {

struct SurfaceProperties {
  GSIModel model = GSIModel::Maxwell;
  double temperature_K = 0.0;
  // Maxwell
  double diffuse_fraction = 0.0;
  // CLL
  double alpha_n = 0.0;
  double sigma_t = 0.0;
  // Velocity of the wall in the body frame (zero for a rigid body at rest in its own frame).
  Vec3 wall_velocity;
};

struct ScatteringResult {
  Vec3 velocity_out;          // body frame
  double internal_energy_out; // J; monatomic gases: unchanged (0)
};

// Scattered velocity in the wall frame for a given incident relative velocity.
KOKKOS_INLINE_FUNCTION Vec3 scatter_relative(const Vec3& c_in, const Vec3& n, double particle_mass,
                                             const SurfaceProperties& surface, RandomStream& rng) {
  const double s = Kokkos::sqrt(constants::k_B * surface.temperature_K / particle_mass);
  if (surface.model == GSIModel::Maxwell) {
    if (rng.uniform() >= surface.diffuse_fraction) return reflect(c_in, n);
    const OrthonormalBasis b = make_basis_from_normal(n);
    const double z1 = rng.normal();
    const double z2 = rng.normal();
    const double cn = s * Kokkos::sqrt(-2.0 * Kokkos::log(rng.uniform_open()));
    return b.to_global(Vec3(s * z1, s * z2, cn));
  }
  // CLL
  const OrthonormalBasis b = make_basis_from_normal(n);
  const Vec3 l = b.to_local(c_in);
  const double sig = surface.sigma_t;
  const double an = surface.alpha_n;
  const double st = s * Kokkos::sqrt(sig * (2.0 - sig));
  const double ct1 = (1.0 - sig) * l.x + st * rng.normal();
  const double ct2 = (1.0 - sig) * l.y + st * rng.normal();
  const double a = Kokkos::sqrt(1.0 - an) * Kokkos::fabs(l.z);
  const double sa = s * Kokkos::sqrt(an);
  double cn = 0.0;
  // c_n = 0 has probability zero; it is redrawn so that c_r . n > 0 always.
  do {
    const double z3 = rng.normal();
    const double z4 = rng.normal();
    const double p = a + sa * z3;
    cn = Kokkos::sqrt(p * p + sa * sa * z4 * z4);
  } while (!(cn > 0.0));
  return b.to_global(Vec3(ct1, ct2, cn));
}

// Common GSI interface (implementation plan, section 4).
//   velocity_in     incident velocity (body frame)
//   wall_velocity   velocity of the surface element (body frame)
//   outward_normal  unit normal pointing into the gas on the side that was hit
KOKKOS_INLINE_FUNCTION ScatteringResult scatter_particle(const Vec3& velocity_in, const Vec3& wall_velocity,
                                                         const Vec3& outward_normal, double particle_mass,
                                                         double internal_energy_in, const SurfaceProperties& surface,
                                                         RandomStream& rng) {
  const Vec3 c_in = velocity_in - wall_velocity;
  const Vec3 c_r = scatter_relative(c_in, outward_normal, particle_mass, surface, rng);
  return {c_r + wall_velocity, internal_energy_in};
}

}  // namespace vleo
