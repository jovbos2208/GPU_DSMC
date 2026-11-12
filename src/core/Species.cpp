#include "vleo/core/Species.hpp"

#include <array>

#include "vleo/core/PhysicalConstants.hpp"

namespace vleo {

namespace {

// Standard atomic weights (IUPAC, conventional values).
constexpr SpeciesInfo make(std::string_view name, double molar_mass, bool monatomic, bool supported) {
  return SpeciesInfo{name, molar_mass, molar_mass * constants::amu, monatomic, supported};
}

constexpr std::array<SpeciesInfo, 5> kSpecies{{
    make("O", 15.999, true, true),
    make("N", 14.007, true, true),
    make("He", 4.002602, true, true),
    make("N2", 28.014, false, false),
    make("O2", 31.998, false, false),
}};

}  // namespace

std::span<const SpeciesInfo> known_species() { return kSpecies; }

std::optional<SpeciesInfo> find_species(std::string_view name) {
  for (const auto& s : kSpecies)
    if (s.name == name) return s;
  return std::nullopt;
}

}  // namespace vleo
