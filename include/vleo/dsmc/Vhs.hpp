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

#include "vleo/core/Frames.hpp"
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
  double alpha = 1.0;      // VSS scattering exponent (1: isotropic VHS scattering)

  KOKKOS_INLINE_FUNCTION double sigma_g(double g) const { return g > 0.0 ? coeff * Kokkos::pow(g, exponent) : 0.0; }
  KOKKOS_INLINE_FUNCTION double sigma(double g) const { return g > 0.0 ? sigma_g(g) / g : 0.0; }
};

// Pair parameters for species p and q (collision diameter and omega averaged).
// vss = true uses the VSS diameters and scattering exponents.
inline VhsPair make_vhs_pair(double m_p, const VhsParameters& p, double m_q, const VhsParameters& q,
                             bool vss = false) {
  const double d = vss ? 0.5 * (p.d_vss_m + q.d_vss_m) : 0.5 * (p.d_ref_m + q.d_ref_m);
  const double omega = 0.5 * (p.omega + q.omega);
  const double T_ref = 0.5 * (p.T_ref_K + q.T_ref_K);
  const double m_r = m_p * m_q / (m_p + m_q);
  VhsPair v;
  v.coeff = constants::pi * d * d * std::pow(2.0 * constants::k_B * T_ref / m_r, omega - 0.5) / std::tgamma(2.5 - omega);
  v.exponent = 2.0 - 2.0 * omega;
  v.m_p = m_p;
  v.m_q = m_q;
  v.alpha = vss ? 0.5 * (p.alpha + q.alpha) : 1.0;
  return v;
}

// Parameters with the diameter of the chosen collision model (VSS: d_vss).
inline VhsParameters for_model(VhsParameters p, bool vss) {
  if (vss) p.d_ref_m = p.d_vss_m;
  return p;
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

// Isotropic scattering with a prescribed post-collision relative speed
// (inelastic collisions: translational energy exchanged with internal modes).
KOKKOS_INLINE_FUNCTION void vhs_collide_with_speed(Vec3& v_p, Vec3& v_q, double m_p, double m_q, double g_new_mag,
                                                   RandomStream& rng) {
  const double mt = m_p + m_q;
  const Vec3 v_cm = (m_p * v_p + m_q * v_q) / mt;
  const Vec3 g_new = g_new_mag * isotropic_direction(rng);
  v_p = v_cm + (m_q / mt) * g_new;
  v_q = v_cm - (m_p / mt) * g_new;
}

// VSS scattering (Koura & Matsumoto 1991): deflection cos(chi) = 2 R^(1/alpha) - 1
// about the pre-collision relative velocity, uniform azimuth; post-collision
// relative speed g_new_mag. alpha = 1 reproduces isotropic scattering.
KOKKOS_INLINE_FUNCTION void vss_collide_with_speed(Vec3& v_p, Vec3& v_q, double m_p, double m_q, double g_new_mag,
                                                   double alpha, RandomStream& rng) {
  const double mt = m_p + m_q;
  const Vec3 v_cm = (m_p * v_p + m_q * v_q) / mt;
  const Vec3 g = v_p - v_q;
  const double gm = norm(g);
  Vec3 g_new;
  if (!(gm > 0.0)) {
    g_new = g_new_mag * isotropic_direction(rng);
  } else {
    const double cos_chi = 2.0 * Kokkos::pow(rng.uniform(), 1.0 / alpha) - 1.0;
    const double sin_chi = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - cos_chi * cos_chi));
    const double eps = constants::two_pi * rng.uniform();
    const Vec3 e = g / gm;
    const OrthonormalBasis b = make_basis_from_normal(e);
    g_new = g_new_mag * (cos_chi * e + sin_chi * (Kokkos::cos(eps) * b.t1 + Kokkos::sin(eps) * b.t2));
  }
  v_p = v_cm + (m_q / mt) * g_new;
  v_q = v_cm - (m_p / mt) * g_new;
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
