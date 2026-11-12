#include "vleo/core/Species.hpp"

#include <array>

#include "vleo/core/PhysicalConstants.hpp"

namespace vleo {

namespace {

// Standard atomic weights (IUPAC, conventional values).
constexpr SpeciesInfo make(std::string_view name, double molar_mass, bool monatomic, bool supported,
                           VhsParameters vhs = {}) {
  return SpeciesInfo{name, molar_mass, molar_mass * constants::amu, monatomic, supported, vhs};
}

// VHS data: G. A. Bird, Molecular Gas Dynamics and the Direct Simulation of
// Gas Flows, Oxford 1994, Table A1 (T_ref = 273 K). Atomic O and N are not in
// that table; their collision data (with sources) follow in milestone M4.1,
// until then they are TPMC-only.
constexpr char kBird94[] = "Bird 1994, Table A1";
constexpr std::array<SpeciesInfo, 6> kSpecies{{
    make("O", 15.999, true, true),
    make("N", 14.007, true, true),
    make("He", 4.002602, true, true, {true, 2.33e-10, 0.66, 273.0, kBird94}),
    make("Ar", 39.948, true, true, {true, 4.17e-10, 0.81, 273.0, kBird94}),
    make("N2", 28.014, false, false, {true, 4.17e-10, 0.74, 273.0, kBird94}),
    make("O2", 31.998, false, false, {true, 4.07e-10, 0.77, 273.0, kBird94}),
}};

}  // namespace

std::span<const SpeciesInfo> known_species() { return kSpecies; }

std::optional<SpeciesInfo> find_species(std::string_view name) {
  for (const auto& s : kSpecies)
    if (s.name == name) return s;
  return std::nullopt;
}

}  // namespace vleo
