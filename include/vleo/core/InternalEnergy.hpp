#pragma once

// Internal energy of molecules (M4.5): data model and sampling primitives.
//
// Data model: every particle carries a rotational energy e_rot [J]
// (continuous, rot_dof = 2) and a vibrational energy e_vib = i k theta_v [J]
// (quantised harmonic oscillator, level i >= 0). Atoms have rot_dof = 0 and
// no vibrational mode; their internal energies stay exactly zero.
//
// Equilibrium distributions at temperature T:
//   rotation, 2 DOF:  e_rot ~ Exp(k T)                 (e = -k T ln U)
//   vibration (SHO):  P(i) = (1 - q) q^i, q = exp(-theta_v / T)   (i = floor(ln U / ln q))

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/core/Random.hpp"

namespace vleo {

struct SpeciesModes {
  double mass = 0.0;
  int rot_dof = 0;               // 0 or 2
  double rot_relax_prob = 0.0;   // 1 / Z_rot (DSMC)
  int vib = 0;                   // 1: one SHO mode
  double theta_v = 0.0;          // [K]
  double vib_relax_prob = 0.0;   // 1 / Z_vib (DSMC)
  double omega = 0.5;            // VHS exponent (for Larsen-Borgnakke)
};

KOKKOS_INLINE_FUNCTION double sample_rotational_energy(int rot_dof, double T, RandomStream& rng) {
  if (rot_dof != 2 || !(T > 0.0)) return 0.0;
  return -constants::k_B * T * Kokkos::log(rng.uniform_open());
}

KOKKOS_INLINE_FUNCTION int sample_vibrational_level(double theta_v, double T, RandomStream& rng) {
  if (!(theta_v > 0.0) || !(T > 0.0)) return 0;
  const double ln_q = -theta_v / T;  // < 0
  const double level = Kokkos::floor(Kokkos::log(rng.uniform_open()) / ln_q);
  return level < 1.0e6 ? static_cast<int>(level) : 1000000;
}

KOKKOS_INLINE_FUNCTION double sample_vibrational_energy(int vib, double theta_v, double T, RandomStream& rng) {
  if (!vib) return 0.0;
  return sample_vibrational_level(theta_v, T, rng) * constants::k_B * theta_v;
}

// Larsen-Borgnakke, rotation (2 DOF) of one molecule with the translational
// collision energy Et (VHS pair exponent omega): the share x = e_rot / (Et +
// e_rot) follows Beta(1, 5/2 - omega), sampled exactly as x = 1 - U^(1/(5/2-omega)).
KOKKOS_INLINE_FUNCTION void lb_rotation(double& Et, double& e_rot, double omega, RandomStream& rng) {
  const double Ec = Et + e_rot;
  const double x = 1.0 - Kokkos::pow(rng.uniform_open(), 1.0 / (2.5 - omega));
  e_rot = x * Ec;
  Et = Ec - e_rot;
}

// Quantum Larsen-Borgnakke for a harmonic oscillator (Bird 2013, Bergemann &
// Boyd 1994): level i uniform in [0, i_max], i_max = floor(Ec / (k theta_v)),
// accepted with probability (1 - i k theta_v / Ec)^(3/2 - omega).
KOKKOS_INLINE_FUNCTION void lb_vibration(double& Et, double& e_vib, double theta_v, double omega, RandomStream& rng) {
  const double Ec = Et + e_vib;
  const double quantum = constants::k_B * theta_v;
  const double i_max = Kokkos::floor(Ec / quantum);
  double e_new = 0.0;
  for (int attempt = 0; attempt < 100000; ++attempt) {
    const double i = Kokkos::floor(rng.uniform() * (i_max + 1.0));
    const double ev = Kokkos::fmin(i, i_max) * quantum;
    if (rng.uniform() < Kokkos::pow(1.0 - ev / Ec, 1.5 - omega)) {
      e_new = ev;
      break;
    }
  }
  e_vib = e_new;
  Et = Ec - e_vib;
}

}  // namespace vleo
