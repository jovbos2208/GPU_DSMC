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
#include "vleo/core/Species.hpp"
#include "vleo/core/Vec3.hpp"
#include "vleo/gsi/GSIModel.hpp"

namespace vleo {

enum class SolverMode { TPMC, DSMC };
enum class LengthUnit { Metre, Centimetre, Millimetre };
enum class OutputFormat { JSON, CSV, HDF5 };
enum class CollisionModel { VHS, VSS };
enum class ParameterMode { Automatic, Fixed };
enum class ConvergenceQuantity { CD, CL, CY, CMx, CMy, CMz, HeatRate };

std::string_view to_string(SolverMode v);
std::string_view to_string(LengthUnit v);
std::string_view to_string(OutputFormat v);
std::string_view to_string(CollisionModel v);
std::string_view to_string(ParameterMode v);
std::string_view to_string(GSIModel v);
std::string_view to_string(ConvergenceQuantity v);

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
  // Every connected component must be a closed, consistently oriented
  // 2-manifold. If false, open components are allowed and treated as
  // two-sided surfaces (e.g. a flat plate).
  bool require_closed = true;
};

// Simulation domain: axis-aligned box around the body. Exactly one of
// margin_m (box = mesh bounding box grown by the margin on every side) or the
// explicit box_min_m/box_max_m pair is set after parsing; if neither is given
// the margin defaults to kDefaultDomainMarginFraction * largest mesh extent.
struct DomainSection {
  std::optional<double> margin_m;
  std::optional<Vec3> box_min_m;
  std::optional<Vec3> box_max_m;
};
inline constexpr double kDefaultDomainMarginFraction = 0.05;

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
  VhsParameters vhs;
  InternalModes modes;
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
  double alpha_rot = 0.0;  // rotational energy accommodation (default: alpha_n)
  double alpha_vib = 0.0;  // vibrational energy accommodation (default: alpha_n)
};

struct SurfaceSection {
  GSIModel model = GSIModel::Maxwell;
  double temperature_K = 0.0;
  std::optional<MaxwellParameters> maxwell;  // always set if model == Maxwell
  std::optional<CLLParameters> cll;          // always set if model == CLL
  std::filesystem::path material_file;       // empty unless loaded via surface.material_file
};

// Optional early stop: after every batch the standard error of `quantity` is
// compared with the target (exactly one of relative_error / absolute_error);
// the run stops once it is reached and at least min_histories were run.
// tpmc.particle_histories remains the upper limit.
struct ConvergenceCriterion {
  ConvergenceQuantity quantity = ConvergenceQuantity::CD;
  std::optional<double> relative_error;  // std_error / |value|
  std::optional<double> absolute_error;  // std_error in the unit of the quantity
  std::uint64_t min_histories = 0;
};

struct TPMCSection {
  std::uint64_t particle_histories = 1'000'000;
  std::uint32_t max_reflections = 100;
  std::uint64_t batch_size = 1u << 20;  // histories processed per kernel launch
  std::optional<ConvergenceCriterion> convergence;
};

struct DSMCSection {
  CollisionModel collision_model = CollisionModel::VHS;
  ParameterMode timestep_mode = ParameterMode::Automatic;
  std::optional<double> timestep_s;  // set iff timestep_mode == Fixed
  ParameterMode cell_size_mode = ParameterMode::Automatic;
  std::optional<double> cell_size_m;  // set iff cell_size_mode == Fixed
  bool cell_size_pilot = true;        // automatic cells: refine with pilot runs (local lambda)
  std::uint32_t particles_per_cell_target = 30;
  std::optional<std::uint64_t> warmup_steps;  // unset: automatic (docs/dsmc.md)
  std::uint64_t sample_steps = 1000;
  std::uint32_t blocks = 20;                  // blocks for the block-averaged standard error
  std::optional<std::uint64_t> max_particles; // unset: automatic
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
  DomainSection domain;
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
