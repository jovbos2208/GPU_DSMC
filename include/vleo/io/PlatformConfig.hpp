#pragma once

// Platform (hardware) configuration: configs/platforms/*.yaml.
// Kept separate from the physical simulation configuration so the same
// simulation file runs unchanged on CPU, NVIDIA and AMD builds.

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "vleo/io/ConfigError.hpp"

namespace vleo {

enum class Backend { CPU, CUDA, HIP };

std::string_view to_string(Backend b);

// Backend this binary was compiled for.
Backend compiled_backend();

struct PlatformConfig {
  std::filesystem::path source_file;
  Backend backend = Backend::CPU;
  int device_id = 0;     // GPU index (ignored for cpu)
  int num_threads = 0;   // host threads for OpenMP; 0 = Kokkos default
  std::string description;
};

PlatformConfig load_platform_config(const std::filesystem::path& file);
PlatformConfig parse_platform_config(std::string_view yaml_text);

// Throws ConfigError if the platform file requests a backend other than the
// compiled one (a YAML entry cannot turn an NVIDIA binary into an AMD one).
void check_platform_matches_build(const PlatformConfig& p);

}  // namespace vleo
