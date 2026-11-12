#pragma once

// Physical constants and unit conversions. The solver works exclusively in SI
// base units internally (m, kg, s, K, N, J, W). Conversions happen only at the
// input boundary (e.g. geometry length units) - see docs/conventions.md.

namespace vleo::constants {

// Boltzmann constant [J/K] (exact, SI 2019).
inline constexpr double k_B = 1.380649e-23;

// Avogadro constant [1/mol] (exact, SI 2019).
inline constexpr double N_A = 6.02214076e23;

// Unified atomic mass unit [kg] (CODATA 2022).
inline constexpr double amu = 1.66053906892e-27;

inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double two_pi = 2.0 * pi;

}  // namespace vleo::constants

namespace vleo::units {

// Length units accepted for geometry input, as factors to metres.
inline constexpr double metre = 1.0;
inline constexpr double centimetre = 1.0e-2;
inline constexpr double millimetre = 1.0e-3;

}  // namespace vleo::units
