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

}  // namespace vleo
