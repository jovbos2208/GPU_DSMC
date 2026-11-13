#pragma once

// Gas species database (host side), SI units. Every value carries its source
// (docs/species.md). Data sets:
//   [Bird94]  G. A. Bird, Molecular Gas Dynamics and the Direct Simulation of
//             Gas Flows, Oxford 1994, Table A1 (VHS and VSS columns, T_ref = 273 K)
//   [SPARTA]  SPARTA DSMC code, data/air.species and data/air.vss, release
//             27Aug2026 (used for the SPARTA comparison cases, M4.10)
//   [IUPAC]   standard atomic weights (conventional values)

#include <optional>
#include <span>
#include <string_view>

namespace vleo {

// Collision data. Cross-section (VHS and VSS share the form)
//   sigma_T(g) = pi d^2 (2 k T_ref / (m_r g^2))^(omega - 1/2) / Gamma(5/2 - omega)
// with d = d_vhs for the VHS model and d = d_vss for the VSS model. The VSS
// deflection follows cos(chi) = 2 R^(1/alpha) - 1 (alpha = 1: isotropic, VHS).
struct VhsParameters {
  bool available = false;  // false: species cannot be used in DSMC
  double d_ref_m = 0.0;    // VHS reference diameter
  double omega = 0.5;      // viscosity-temperature exponent
  double T_ref_K = 273.0;  // reference temperature
  double d_vss_m = 0.0;    // VSS reference diameter
  double alpha = 1.0;      // VSS scattering exponent
  std::string_view source;
};

// Internal energy modes (Larsen-Borgnakke, docs/species.md).
struct InternalModes {
  int rot_dof = 0;                     // rotational degrees of freedom (0 or 2)
  double rot_collision_number = 0.0;   // Z_rot (constant): 1 / probability of rotational relaxation
  int vib_dof = 0;                     // 0, or 2 for one harmonic-oscillator mode
  double vib_temperature_K = 0.0;      // characteristic vibrational temperature
  double vib_collision_number = 0.0;   // Z_vib (constant)
  std::string_view source;
};

struct SpeciesInfo {
  std::string_view name;
  double molar_mass_g_per_mol;  // = mass in unified atomic mass units
  double mass_kg;
  bool monatomic;
  // Supported by the current solver release. Molecular species need their
  // internal energy modes; they are rejected until those are validated so
  // their internal energy is never silently ignored.
  bool supported;
  VhsParameters vhs;     // collision data for DSMC
  InternalModes modes;   // internal energy (molecules)
};

// All species known to the solver, supported or not.
std::span<const SpeciesInfo> known_species();

// Lookup by exact (case-sensitive) name, e.g. "O", "N", "He", "Ar", "N2", "O2".
std::optional<SpeciesInfo> find_species(std::string_view name);

}  // namespace vleo
