#include "vleo/io/Config.hpp"

#include "vleo/core/PhysicalConstants.hpp"

namespace vleo {

std::string_view to_string(SolverMode v) { return v == SolverMode::TPMC ? "tpmc" : "dsmc"; }

std::string_view to_string(LengthUnit v) {
  switch (v) {
    case LengthUnit::Metre: return "m";
    case LengthUnit::Centimetre: return "cm";
    case LengthUnit::Millimetre: return "mm";
  }
  return "?";
}

std::string_view to_string(OutputFormat v) {
  switch (v) {
    case OutputFormat::JSON: return "json";
    case OutputFormat::CSV: return "csv";
    case OutputFormat::HDF5: return "hdf5";
  }
  return "?";
}

std::string_view to_string(CollisionModel) { return "VHS"; }

std::string_view to_string(ParameterMode v) { return v == ParameterMode::Automatic ? "automatic" : "fixed"; }

std::string_view to_string(GSIModel v) { return v == GSIModel::Maxwell ? "maxwell" : "cll"; }

double length_unit_to_metre(LengthUnit u) {
  switch (u) {
    case LengthUnit::Metre: return units::metre;
    case LengthUnit::Centimetre: return units::centimetre;
    case LengthUnit::Millimetre: return units::millimetre;
  }
  return 0.0;
}

}  // namespace vleo
