#pragma once

// Variable Hard Sphere (VHS) collision model (Bird 1994, section 4.3).
//
//   sigma_T(g) = pi d_ref^2 (2 k T_ref / (m_r g^2))^(omega - 1/2) / Gamma(5/2 - omega)
//
// with the reduced mass m_r = m_p m_q / (m_p + m_q). Post-collision relative
// velocities are isotropic in the centre-of-mass frame (VHS); |g| is preserved,
// so momentum and energy are conserved exactly up to rounding.
//
// Equilibrium relations used for parameters and diagnostics (single species):
//   <sigma_T g> = 2 sqrt(pi) d_ref^2 sqrt(2 k T_ref / m_r) (T / T_ref)^(1 - omega)
//   collision rate per molecule  nu = n <sigma_T g>
//   mean free path  lambda = c_mean / nu,   c_mean = sqrt(8 k T / (pi m))

#include <cmath>

#include <Kokkos_Core.hpp>

#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Random.hpp"
#include "vleo/core/Species.hpp"
#include "vleo/core/Vec3.hpp"

namespace vleo {

// Pair data in the form used on the device: sigma_T g = coeff * g^(2 - 2 omega).
struct VhsPair {
  double coeff = 0.0;      // pi d^2 (2 k T_ref / m_r)^(omega - 1/2) / Gamma(5/2 - omega)
  double exponent = 1.0;   // 2 - 2 omega
  double m_p = 0.0, m_q = 0.0;

  KOKKOS_INLINE_FUNCTION double sigma_g(double g) const { return g > 0.0 ? coeff * Kokkos::pow(g, exponent) : 0.0; }
  KOKKOS_INLINE_FUNCTION double sigma(double g) const { return g > 0.0 ? sigma_g(g) / g : 0.0; }
};

// Pair parameters for species p and q (collision diameter and omega averaged).
inline VhsPair make_vhs_pair(double m_p, const VhsParameters& p, double m_q, const VhsParameters& q) {
  const double d = 0.5 * (p.d_ref_m + q.d_ref_m);
  const double omega = 0.5 * (p.omega + q.omega);
  const double T_ref = 0.5 * (p.T_ref_K + q.T_ref_K);
  const double m_r = m_p * m_q / (m_p + m_q);
  VhsPair v;
  v.coeff = constants::pi * d * d * std::pow(2.0 * constants::k_B * T_ref / m_r, omega - 0.5) / std::tgamma(2.5 - omega);
  v.exponent = 2.0 - 2.0 * omega;
  v.m_p = m_p;
  v.m_q = m_q;
  return v;
}

// <sigma_T g> in an equilibrium gas at temperature T.
inline double vhs_mean_sigma_g(double m_p, const VhsParameters& p, double m_q, const VhsParameters& q, double T) {
  const double d = 0.5 * (p.d_ref_m + q.d_ref_m);
  const double omega = 0.5 * (p.omega + q.omega);
  const double T_ref = 0.5 * (p.T_ref_K + q.T_ref_K);
  const double m_r = m_p * m_q / (m_p + m_q);
  return 2.0 * std::sqrt(constants::pi) * d * d * std::sqrt(2.0 * constants::k_B * T_ref / m_r) *
         std::pow(T / T_ref, 1.0 - omega);
}

// Mean free path of a single-species equilibrium gas (lambda = c_mean / nu).
inline double vhs_mean_free_path(double m, const VhsParameters& p, double n, double T) {
  const double c_mean = std::sqrt(8.0 * constants::k_B * T / (constants::pi * m));
  return c_mean / (n * vhs_mean_sigma_g(m, p, m, p, T));
}

// Mean collision time of a single-species equilibrium gas (1 / nu).
inline double vhs_mean_collision_time(double m, const VhsParameters& p, double n, double T) {
  return 1.0 / (n * vhs_mean_sigma_g(m, p, m, p, T));
}

// Uniformly distributed unit vector.
KOKKOS_INLINE_FUNCTION Vec3 isotropic_direction(RandomStream& rng) {
  const double cos_t = 2.0 * rng.uniform() - 1.0;
  const double sin_t = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - cos_t * cos_t));
  const double phi = constants::two_pi * rng.uniform();
  return {sin_t * Kokkos::cos(phi), sin_t * Kokkos::sin(phi), cos_t};
}

// Elastic VHS collision: isotropic scattering in the centre-of-mass frame.
KOKKOS_INLINE_FUNCTION void vhs_collide(Vec3& v_p, Vec3& v_q, double m_p, double m_q, RandomStream& rng) {
  const double mt = m_p + m_q;
  const Vec3 v_cm = (m_p * v_p + m_q * v_q) / mt;
  const double g = norm(v_p - v_q);
  const Vec3 g_new = g * isotropic_direction(rng);
  v_p = v_cm + (m_q / mt) * g_new;
  v_q = v_cm - (m_p / mt) * g_new;
}

}  // namespace vleo
