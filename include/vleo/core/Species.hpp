#pragma once

// Gas species database (host side). Molecular masses in SI units.

#include <optional>
#include <span>
#include <string_view>

namespace vleo {

// Variable Hard Sphere parameters (Bird 1994): sigma_T(g) =
// pi d_ref^2 (2 k T_ref / (m_r g^2))^(omega - 1/2) / Gamma(5/2 - omega).
struct VhsParameters {
  bool available = false;  // false: species cannot be used in DSMC (yet)
  double d_ref_m = 0.0;    // reference diameter
  double omega = 0.5;      // viscosity-temperature exponent
  double T_ref_K = 273.0;  // reference temperature
  std::string_view source;
};

struct SpeciesInfo {
  std::string_view name;
  double molar_mass_g_per_mol;  // = mass in unified atomic mass units
  double mass_kg;
  bool monatomic;
  // Supported by the current solver release. Molecular species need internal
  // energy modes (rotation/vibration, milestone M4); they are recognised but
  // rejected so their internal energy is never silently ignored.
  bool supported;
  VhsParameters vhs;  // collision data for DSMC
};

// All species known to the solver, supported or not.
std::span<const SpeciesInfo> known_species();

// Lookup by exact (case-sensitive) name, e.g. "O", "N", "He", "N2", "O2".
std::optional<SpeciesInfo> find_species(std::string_view name);

}  // namespace vleo
