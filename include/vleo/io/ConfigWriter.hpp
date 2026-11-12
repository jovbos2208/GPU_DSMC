#pragma once

#include <string>

#include "vleo/io/Config.hpp"
#include "vleo/io/JsonWriter.hpp"

namespace vleo {

// Resolved configuration as YAML: every schema key with its effective value
// (defaults filled in, geometry path absolute,
// a surface.material_file inlined). The output is itself a valid input file
// and parses back to an identical SimulationConfig (round trip).
std::string to_resolved_yaml(const SimulationConfig& cfg);

// Writes the resolved configuration as a JSON object value (same content and
// key names as to_resolved_yaml), at the writer's current position.
void write_resolved_json(JsonWriter& w, const SimulationConfig& cfg);

// Writes the derived quantities (reference density, dynamic pressure, aero
// axes, ...) as a JSON object value.
void write_derived_json(JsonWriter& w, const SimulationConfig& cfg);

}  // namespace vleo
