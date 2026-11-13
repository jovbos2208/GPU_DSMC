#include "vleo/core/Species.hpp"

#include <array>

#include "vleo/core/PhysicalConstants.hpp"

namespace vleo {

namespace {

constexpr char kBird94[] = "Bird 1994, Table A1";
constexpr char kSparta[] = "SPARTA 27Aug2026, data/air.vss";
constexpr char kSpartaSpecies[] = "SPARTA 27Aug2026, data/air.species";

constexpr SpeciesInfo make(std::string_view name, double molar_mass, bool monatomic, bool supported,
                           VhsParameters vhs, InternalModes modes = {}) {
  return SpeciesInfo{name, molar_mass, molar_mass * constants::amu, monatomic, supported, vhs, modes};
}

// Molar masses: [IUPAC]. Collision data and internal modes: see the sources.
//   He, Ar, N2, O2 collision data: [Bird94] (VHS d, omega; VSS d, alpha).
//   O, N collision data: [SPARTA] air.vss (d = 3.0e-10 m, omega = 0.80, alpha = 1.0, T_ref = 273.15 K);
//     atomic species are not in [Bird94] Table A1.
//   N2, O2 internal modes: [SPARTA] air.species: rotational relaxation probability 0.2
//     (Z_rot = 5), vibrational relaxation probability 1.90114e-5 (N2) / 5.58659e-5 (O2),
//     characteristic vibrational temperature 3371 K (N2) / 2256 K (O2).
constexpr std::array<SpeciesInfo, 6> kSpecies{{
    make("O", 15.999, true, true, {true, 3.0e-10, 0.80, 273.15, 3.0e-10, 1.0, kSparta}),
    make("N", 14.007, true, true, {true, 3.0e-10, 0.80, 273.15, 3.0e-10, 1.0, kSparta}),
    make("He", 4.002602, true, true, {true, 2.33e-10, 0.66, 273.0, 2.30e-10, 1.26, kBird94}),
    make("Ar", 39.948, true, true, {true, 4.17e-10, 0.81, 273.0, 4.11e-10, 1.40, kBird94}),
    make("N2", 28.014, false, true, {true, 4.17e-10, 0.74, 273.0, 4.11e-10, 1.36, kBird94},
         {2, 5.0, 2, 3371.0, 1.0 / 1.90114e-5, kSpartaSpecies}),
    make("O2", 31.998, false, true, {true, 4.07e-10, 0.77, 273.0, 4.01e-10, 1.40, kBird94},
         {2, 5.0, 2, 2256.0, 1.0 / 5.58659e-5, kSpartaSpecies}),
}};

}  // namespace

std::span<const SpeciesInfo> known_species() { return kSpecies; }

std::optional<SpeciesInfo> find_species(std::string_view name) {
  for (const auto& s : kSpecies)
    if (s.name == name) return s;
  return std::nullopt;
}

}  // namespace vleo
