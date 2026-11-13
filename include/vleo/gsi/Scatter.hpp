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
// Internal energy (molecules, M4.8) is handled after the velocity:
//   Maxwell specular: e_rot, e_vib unchanged; diffuse: both resampled from
//     the equilibrium distributions at T_w (full accommodation).
//   CLL rotation (2 DOF, x = e_rot / (k T_w)):
//     x' = (sqrt((1 - alpha_rot) x) + sqrt(alpha_rot / 2) Z5)^2 + (alpha_rot / 2) Z6^2
//     so that <e_rot'> = (1 - alpha_rot) e_rot + alpha_rot k T_w; alpha_rot = 1 is diffuse.
//   CLL vibration: with probability alpha_vib resampled at T_w, else unchanged.

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "vleo/core/Frames.hpp"
#include "vleo/core/InternalEnergy.hpp"
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
  double alpha_rot = 0.0;  // rotational energy accommodation
  double alpha_vib = 0.0;  // vibrational energy accommodation (probability)
  // Velocity of the wall in the body frame (zero for a rigid body at rest in its own frame).
  Vec3 wall_velocity;
};

struct ScatteringResult {
  Vec3 velocity_out;  // body frame
  double e_rot_out;   // J
  double e_vib_out;   // J
};

// Internal energy after the interaction (called after the velocity draws).
KOKKOS_INLINE_FUNCTION void scatter_internal(bool diffuse_maxwell, const SpeciesModes& sm, const SurfaceProperties& surface,
                                             double& e_rot, double& e_vib, RandomStream& rng) {
  const double Tw = surface.temperature_K;
  if (surface.model == GSIModel::Maxwell) {
    if (!diffuse_maxwell) return;
    e_rot = sample_rotational_energy(sm.rot_dof, Tw, rng);
    e_vib = sample_vibrational_energy(sm.vib, sm.theta_v, Tw, rng);
    return;
  }
  if (sm.rot_dof == 2) {
    const double a = surface.alpha_rot;
    const double kT = constants::k_B * Tw;
    const double x = e_rot / kT;
    const double p = Kokkos::sqrt((1.0 - a) * x) + Kokkos::sqrt(0.5 * a) * rng.normal();
    const double z = rng.normal();
    e_rot = (p * p + 0.5 * a * z * z) * kT;
  }
  if (sm.vib && rng.uniform() < surface.alpha_vib) e_vib = sample_vibrational_energy(sm.vib, sm.theta_v, Tw, rng);
}

// Scattered velocity in the wall frame for a given incident relative velocity.
// `diffuse` reports whether the Maxwell model took the diffuse branch.
KOKKOS_INLINE_FUNCTION Vec3 scatter_relative(const Vec3& c_in, const Vec3& n, double particle_mass,
                                             const SurfaceProperties& surface, RandomStream& rng, bool& diffuse) {
  const double s = Kokkos::sqrt(constants::k_B * surface.temperature_K / particle_mass);
  diffuse = false;
  if (surface.model == GSIModel::Maxwell) {
    if (rng.uniform() >= surface.diffuse_fraction) return reflect(c_in, n);
    diffuse = true;
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
//   species         mass and internal modes of the particle
KOKKOS_INLINE_FUNCTION ScatteringResult scatter_particle(const Vec3& velocity_in, const Vec3& wall_velocity,
                                                         const Vec3& outward_normal, const SpeciesModes& species,
                                                         double e_rot_in, double e_vib_in,
                                                         const SurfaceProperties& surface, RandomStream& rng) {
  const Vec3 c_in = velocity_in - wall_velocity;
  bool diffuse = false;
  const Vec3 c_r = scatter_relative(c_in, outward_normal, species.mass, surface, rng, diffuse);
  double e_rot = e_rot_in, e_vib = e_vib_in;
  if (species.rot_dof > 0 || species.vib) scatter_internal(diffuse, species, surface, e_rot, e_vib, rng);
  return {c_r + wall_velocity, e_rot, e_vib};
}

// Monatomic convenience overload (no internal energy).
KOKKOS_INLINE_FUNCTION ScatteringResult scatter_particle(const Vec3& velocity_in, const Vec3& wall_velocity,
                                                         const Vec3& outward_normal, double particle_mass,
                                                         const SurfaceProperties& surface, RandomStream& rng) {
  SpeciesModes sm;
  sm.mass = particle_mass;
  return scatter_particle(velocity_in, wall_velocity, outward_normal, sm, 0.0, 0.0, surface, rng);
}

}  // namespace vleo
