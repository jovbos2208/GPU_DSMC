#pragma once

// Gas species database (host side). Molecular masses in SI units.

#include <optional>
#include <span>
#include <string_view>

namespace vleo {

struct SpeciesInfo {
  std::string_view name;
  double molar_mass_g_per_mol;  // = mass in unified atomic mass units
  double mass_kg;
  bool monatomic;
  // Supported by the current solver release. Molecular species need internal
  // energy modes (rotation/vibration, milestone M4); they are recognised but
  // rejected so their internal energy is never silently ignored.
  bool supported;
};

// All species known to the solver, supported or not.
std::span<const SpeciesInfo> known_species();

// Lookup by exact (case-sensitive) name, e.g. "O", "N", "He", "N2", "O2".
std::optional<SpeciesInfo> find_species(std::string_view name);

}  // namespace vleo
