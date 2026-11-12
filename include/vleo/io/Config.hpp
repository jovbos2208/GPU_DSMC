#pragma once

// Validated, fully resolved simulation configuration (host side).
// The YAML schema is documented in docs/configuration.md. Every field below is
// set after a successful parse: either from the input or from the documented
// default. No field is ever left uninitialised.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vleo/core/Frames.hpp"
#include "vleo/core/Vec3.hpp"
#include "vleo/gsi/GSIModel.hpp"

namespace vleo {

enum class SolverMode { TPMC, DSMC };
enum class LengthUnit { Metre, Centimetre, Millimetre };
enum class OutputFormat { JSON, CSV, HDF5 };
enum class CollisionModel { VHS };
enum class ParameterMode { Automatic, Fixed };

std::string_view to_string(SolverMode v);
std::string_view to_string(LengthUnit v);
std::string_view to_string(OutputFormat v);
std::string_view to_string(CollisionModel v);
std::string_view to_string(ParameterMode v);
std::string_view to_string(GSIModel v);

double length_unit_to_metre(LengthUnit u);

struct SimulationSection {
  std::string name;
  SolverMode mode = SolverMode::TPMC;
  std::uint64_t random_seed = 0;
};

struct GeometrySection {
  std::string file;                     // as written in the input
  std::filesystem::path resolved_file;  // absolute, relative paths resolved against the config file directory
  LengthUnit units = LengthUnit::Metre;
};

struct ReferenceSection {
  double area_m2 = 0.0;
  double length_m = 0.0;
  Vec3 moment_center_m;
  Vec3 lift_axis_body;
};

struct SpeciesFraction {
  std::string name;
  double mole_fraction = 0.0;  // number fraction as given; all fractions sum to 1 within 1e-6
  double mass_kg = 0.0;
};

struct GasSection {
  double number_density_m3 = 0.0;
  double translational_temperature_K = 0.0;
  Vec3 bulk_velocity_body_mps;
  std::vector<SpeciesFraction> species;  // in input order
};

struct MaxwellParameters {
  double diffuse_fraction = 0.0;
};

struct CLLParameters {
  double alpha_n = 0.0;
  double sigma_t = 0.0;
};

struct SurfaceSection {
  GSIModel model = GSIModel::Maxwell;
  double temperature_K = 0.0;
  std::optional<MaxwellParameters> maxwell;  // always set if model == Maxwell
  std::optional<CLLParameters> cll;          // always set if model == CLL
  std::filesystem::path material_file;       // empty unless loaded via surface.material_file
};

struct TPMCSection {
  std::uint64_t particle_histories = 1'000'000;
  std::uint32_t max_reflections = 100;
};

struct DSMCSection {
  CollisionModel collision_model = CollisionModel::VHS;
  ParameterMode timestep_mode = ParameterMode::Automatic;
  std::optional<double> timestep_s;  // set iff timestep_mode == Fixed
  ParameterMode cell_size_mode = ParameterMode::Automatic;
  std::optional<double> cell_size_m;  // set iff cell_size_mode == Fixed
  std::uint32_t particles_per_cell_target = 30;
};

struct StatisticsSection {
  double confidence_level = 0.95;
  bool report_uncertainty = true;
};

struct OutputSection {
  std::string directory = "results/";
  OutputFormat format = OutputFormat::JSON;
  bool save_surface_fields = false;
};

// Quantities computed from the validated input (not part of the input schema).
struct DerivedQuantities {
  double mole_fraction_sum = 1.0;       // sum_s x_s as given (|sum - 1| <= 1e-6)
  double mean_molecular_mass_kg = 0.0;  // sum_s x_s m_s / sum_s x_s
  double mass_density_kg_m3 = 0.0;      // n * mean mass
  double freestream_speed_mps = 0.0;    // |V_inf|
  double dynamic_pressure_Pa = 0.0;     // 0.5 rho V^2
  double geometry_scale_to_m = 1.0;
  AeroAxes aero_axes;              // body-frame components of e_D, e_Y, e_L
  bool lift_axis_was_projected = false;  // input lift axis was not exactly orthogonal to V_inf
};

struct SimulationConfig {
  std::filesystem::path source_file;  // absolute; empty if parsed from a string
  SimulationSection simulation;
  GeometrySection geometry;
  ReferenceSection reference;
  GasSection gas;
  SurfaceSection surface;
  TPMCSection tpmc;
  DSMCSection dsmc;
  StatisticsSection statistics;
  OutputSection output;
  DerivedQuantities derived;
};

}  // namespace vleo
