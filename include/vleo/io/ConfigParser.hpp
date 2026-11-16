#pragma once

#include <filesystem>
#include <string_view>

#include "vleo/io/Config.hpp"
#include "vleo/io/ConfigError.hpp"

namespace vleo {

// Reads and validates a simulation configuration file. Relative paths inside
// the file (geometry.file, surface.material_file) are resolved against the
// directory of the file. Throws ConfigError listing all problems found.
SimulationConfig load_simulation_config(const std::filesystem::path& file);

// Same, from YAML text; relative paths are resolved against `base_dir`
// (made absolute against the current working directory).
SimulationConfig parse_simulation_config(std::string_view yaml_text,
                                         const std::filesystem::path& base_dir = std::filesystem::current_path());

// Attitude sweep (M6.8): replaces the free-stream direction, keeping its
// speed. The flight direction in the body frame is
//   d = (cos(alpha) cos(beta), sin(beta), sin(alpha) cos(beta))
// (alpha = beta = 0: flight along +x) and gas.bulk_velocity_body_mps = -|V| d.
// Derived quantities (aerodynamic axes, ...) are recomputed. Throws
// ConfigError if the lift axis becomes parallel to the flow.
void set_attitude(SimulationConfig& cfg, double alpha_deg, double beta_deg);

}  // namespace vleo
