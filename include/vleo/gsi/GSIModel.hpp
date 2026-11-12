#pragma once

#include <cstdint>

namespace vleo {

// Gas-surface interaction model. Parameter conventions (docs/conventions.md):
//   Maxwell: diffuse_fraction f_d in [0, 1], wall temperature T_w > 0.
//   CLL:     alpha_n in [0, 1] (normal ENERGY accommodation),
//            sigma_t in [0, 2] (tangential MOMENTUM accommodation).
//            Tangential energy accommodation is alpha_t = sigma_t (2 - sigma_t).
enum class GSIModel : std::uint8_t {
  Maxwell,
  CLL,
};

}  // namespace vleo
