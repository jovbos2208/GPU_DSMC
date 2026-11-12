// M0.8-M0.10: configuration schema, validation, resolved output.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <string>

#include "vleo/core/Species.hpp"
#include "vleo/io/ConfigParser.hpp"
#include "vleo/io/ConfigWriter.hpp"
#include "vleo/io/PlatformConfig.hpp"

using namespace vleo;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const fs::path kSource = VLEO_SOURCE_DIR;
const fs::path kData = VLEO_TEST_DATA_DIR;

// Minimal valid configuration: only required keys.
const char* const kMinimal = R"(
simulation:
  name: minimal
  mode: tpmc
  random_seed: 1
geometry:
  file: body.stl
reference:
  area_m2: 2.0
  length_m: 0.5
  moment_center_m: [0.1, 0.0, 0.0]
  lift_axis_body: [0.0, 0.0, 1.0]
gas:
  number_density_m3: 1.0e15
  translational_temperature_K: 1000.0
  bulk_velocity_body_mps: [-7500.0, 0.0, 0.0]
  species:
    O: 1.0
surface:
  model: maxwell
  temperature_K: 300.0
  maxwell:
    diffuse_fraction: 1.0
)";

// Replaces the first occurrence of `from` in the minimal config by `to`.
std::string mutate(const std::string& from, const std::string& to) {
  std::string s = kMinimal;
  const auto pos = s.find(from);
  REQUIRE(pos != std::string::npos);
  s.replace(pos, from.size(), to);
  return s;
}

// Expects the text to be rejected with an issue mentioning `needle`.
void expect_invalid(const std::string& yaml, const std::string& needle) {
  try {
    (void)parse_simulation_config(yaml, kData);
    FAIL("configuration was accepted but should be rejected (expected: " << needle << ")");
  } catch (const ConfigError& e) {
    INFO(e.what());
    REQUIRE(!e.issues().empty());
    CHECK_THAT(std::string(e.what()), ContainsSubstring(needle));
  }
}

}  // namespace

// ----------------------------------------------------------------------------
// Positive tests
// ----------------------------------------------------------------------------

TEST_CASE("Shipped simulation configurations are valid", "[config]") {
  const auto name = GENERATE(as<std::string>{}, "tpmc_maxwell", "tpmc_cll", "dsmc_maxwell", "dsmc_cll");
  INFO(name);
  const SimulationConfig c = load_simulation_config(kSource / "configs" / "simulations" / (name + ".yaml"));
  CHECK(c.simulation.name == name);
  CHECK(c.simulation.mode == (name.starts_with("tpmc") ? SolverMode::TPMC : SolverMode::DSMC));
  CHECK(c.surface.model == (name.ends_with("cll") ? GSIModel::CLL : GSIModel::Maxwell));
  CHECK(c.source_file.is_absolute());
  CHECK(c.geometry.resolved_file.is_absolute());
  CHECK(c.gas.species.size() == 3);
}

TEST_CASE("Shipped platform configurations are valid", "[config]") {
  CHECK(load_platform_config(kSource / "configs/platforms/cpu.yaml").backend == Backend::CPU);
  CHECK(load_platform_config(kSource / "configs/platforms/nvidia.yaml").backend == Backend::CUDA);
  CHECK(load_platform_config(kSource / "configs/platforms/amd.yaml").backend == Backend::HIP);
}

TEST_CASE("Minimal configuration gets all documented defaults", "[config]") {
  const SimulationConfig c = parse_simulation_config(kMinimal, kData);
  CHECK(c.simulation.name == "minimal");
  CHECK(c.simulation.random_seed == 1);
  CHECK(c.geometry.units == LengthUnit::Metre);
  CHECK(c.geometry.resolved_file == (fs::absolute(kData) / "body.stl").lexically_normal());
  CHECK(c.reference.moment_center_m == Vec3(0.1, 0.0, 0.0));
  REQUIRE(c.surface.maxwell);
  CHECK(!c.surface.cll);
  CHECK(c.surface.maxwell->diffuse_fraction == 1.0);
  // Defaults (docs/configuration.md)
  CHECK(c.tpmc.particle_histories == 1'000'000);
  CHECK(c.tpmc.max_reflections == 100);
  CHECK(c.dsmc.collision_model == CollisionModel::VHS);
  CHECK(c.dsmc.timestep_mode == ParameterMode::Automatic);
  CHECK(!c.dsmc.timestep_s);
  CHECK(c.dsmc.cell_size_mode == ParameterMode::Automatic);
  CHECK(!c.dsmc.cell_size_m);
  CHECK(c.dsmc.particles_per_cell_target == 30);
  CHECK(c.statistics.confidence_level == 0.95);
  CHECK(c.statistics.report_uncertainty);
  CHECK(c.output.directory == "results/");
  CHECK(c.output.format == OutputFormat::JSON);
  CHECK(!c.output.save_surface_fields);
}

TEST_CASE("Derived reference quantities", "[config]") {
  const SimulationConfig c = parse_simulation_config(kMinimal, kData);
  const double m_O = find_species("O")->mass_kg;
  CHECK_THAT(m_O, WithinRel(15.999 * 1.66053906892e-27, 1e-15));
  CHECK_THAT(c.derived.mean_molecular_mass_kg, WithinRel(m_O, 1e-15));
  CHECK_THAT(c.derived.mass_density_kg_m3, WithinRel(1e15 * m_O, 1e-15));
  CHECK_THAT(c.derived.freestream_speed_mps, WithinRel(7500.0, 1e-15));
  CHECK_THAT(c.derived.dynamic_pressure_Pa, WithinRel(0.5 * 1e15 * m_O * 7500.0 * 7500.0, 1e-14));
  CHECK(c.derived.aero_axes.drag == Vec3(-1.0, 0.0, 0.0));
  CHECK(c.derived.aero_axes.lift == Vec3(0.0, 0.0, 1.0));
  CHECK(c.derived.aero_axes.side == Vec3(0.0, -1.0, 0.0));
  CHECK(!c.derived.lift_axis_was_projected);

  const SimulationConfig p = parse_simulation_config(
      mutate("lift_axis_body: [0.0, 0.0, 1.0]", "lift_axis_body: [0.3, 0.0, 1.0]"), kData);
  CHECK(p.derived.lift_axis_was_projected);
  CHECK(approx_equal(p.derived.aero_axes.lift, Vec3(0.0, 0.0, 1.0), 1e-15));
}

TEST_CASE("Species mixtures use the database masses and normalised fractions", "[config]") {
  const SimulationConfig c =
      parse_simulation_config(mutate("    O: 1.0", "    O: 0.8\n    N: 0.15\n    He: 0.0500001"), kData);
  REQUIRE(c.gas.species.size() == 3);
  CHECK(c.gas.species[0].name == "O");  // input order preserved
  CHECK(c.gas.species[1].name == "N");
  CHECK(c.gas.species[2].name == "He");
  CHECK(c.gas.species[2].mole_fraction == 0.0500001);  // kept as given
  CHECK(c.gas.species[1].mass_kg == find_species("N")->mass_kg);
  CHECK_THAT(c.derived.mole_fraction_sum, WithinAbs(1.0000001, 1e-15));
  const double m = (0.8 * find_species("O")->mass_kg + 0.15 * find_species("N")->mass_kg +
                    0.0500001 * find_species("He")->mass_kg) /
                   1.0000001;
  CHECK_THAT(c.derived.mean_molecular_mass_kg, WithinRel(m, 1e-14));
  CHECK_THAT(c.derived.mass_density_kg_m3, WithinRel(1e15 * m, 1e-14));
}

TEST_CASE("Optional values are parsed", "[config]") {
  std::string y = kMinimal;
  y += R"(
tpmc:
  particle_histories: 123456789012
  max_reflections: 7
dsmc:
  collision_model: vhs
  timestep_mode: fixed
  timestep_s: 1.0e-6
  cell_size_mode: fixed
  cell_size_m: 0.01
  particles_per_cell_target: 50
statistics:
  confidence_level: 0.99
  report_uncertainty: false
output:
  directory: out/run1
  format: HDF5
  save_surface_fields: true
)";
  y = std::string(y).replace(y.find("file: body.stl"), 14, "file: /abs/body.stl\n  units: mm");
  const SimulationConfig c = parse_simulation_config(y, kData);
  CHECK(c.geometry.units == LengthUnit::Millimetre);
  CHECK(c.derived.geometry_scale_to_m == 1e-3);
  CHECK(c.geometry.resolved_file == fs::path("/abs/body.stl"));
  CHECK(c.tpmc.particle_histories == 123456789012ull);
  CHECK(c.tpmc.max_reflections == 7);
  CHECK(c.dsmc.timestep_mode == ParameterMode::Fixed);
  CHECK(*c.dsmc.timestep_s == 1e-6);
  CHECK(*c.dsmc.cell_size_m == 0.01);
  CHECK(c.dsmc.particles_per_cell_target == 50);
  CHECK(c.statistics.confidence_level == 0.99);
  CHECK(!c.statistics.report_uncertainty);
  CHECK(c.output.directory == "out/run1");
  CHECK(c.output.format == OutputFormat::HDF5);
  CHECK(c.output.save_surface_fields);
}

TEST_CASE("CLL surface and boundary parameter values", "[config]") {
  const std::string cll = mutate("  model: maxwell\n  temperature_K: 300.0\n  maxwell:\n    diffuse_fraction: 1.0",
                                 "  model: cll\n  temperature_K: 300.0\n  cll:\n    alpha_n: 0.0\n    sigma_t: 2.0");
  const SimulationConfig c = parse_simulation_config(cll, kData);
  REQUIRE(c.surface.cll);
  CHECK(c.surface.cll->alpha_n == 0.0);
  CHECK(c.surface.cll->sigma_t == 2.0);
  CHECK(parse_simulation_config(mutate("diffuse_fraction: 1.0", "diffuse_fraction: 0"), kData)
            .surface.maxwell->diffuse_fraction == 0.0);
}

TEST_CASE("Surface can be loaded from a material file", "[config]") {
  const std::string y = mutate("  model: maxwell\n  temperature_K: 300.0\n  maxwell:\n    diffuse_fraction: 1.0",
                               "  material_file: material_cll.yaml");
  const SimulationConfig c = parse_simulation_config(y, kData);
  CHECK(c.surface.model == GSIModel::CLL);
  CHECK(c.surface.temperature_K == 350.0);
  REQUIRE(c.surface.cll);
  CHECK(c.surface.cll->alpha_n == 0.9);
  CHECK(c.surface.material_file == (fs::absolute(kData) / "material_cll.yaml").lexically_normal());

  // The shipped example material is valid too.
  const SimulationConfig e = parse_simulation_config(
      mutate("  model: maxwell\n  temperature_K: 300.0\n  maxwell:\n    diffuse_fraction: 1.0",
             "  material_file: " + (kSource / "configs/materials/example_surface.yaml").string()),
      kData);
  CHECK(e.surface.model == GSIModel::CLL);
}

TEST_CASE("Resolved configuration round-trips exactly", "[config]") {
  const auto name = GENERATE(as<std::string>{}, "tpmc_maxwell", "tpmc_cll", "dsmc_maxwell", "dsmc_cll", "minimal");
  INFO(name);
  const SimulationConfig a = name == "minimal"
                                 ? parse_simulation_config(kMinimal, kData)
                                 : load_simulation_config(kSource / "configs/simulations" / (name + ".yaml"));
  const std::string yaml = to_resolved_yaml(a);
  INFO(yaml);
  const SimulationConfig b = parse_simulation_config(yaml, "/");
  // Re-emitting the re-parsed configuration gives identical text, and every
  // value (including all defaults) survived bit-exactly.
  CHECK(to_resolved_yaml(b) == yaml);
  CHECK(b.reference.area_m2 == a.reference.area_m2);
  CHECK(b.gas.number_density_m3 == a.gas.number_density_m3);
  CHECK(b.gas.bulk_velocity_body_mps == a.gas.bulk_velocity_body_mps);
  CHECK(b.geometry.resolved_file == a.geometry.resolved_file);
  CHECK(b.derived.dynamic_pressure_Pa == a.derived.dynamic_pressure_Pa);
  for (const auto& key : {"simulation:", "geometry:", "reference:", "gas:", "surface:", "tpmc:", "dsmc:",
                          "statistics:", "output:", "particle_histories:", "confidence_level:"}) {
    CHECK_THAT(yaml, ContainsSubstring(key));
  }
}

TEST_CASE("Resolved configuration is also available as JSON", "[config]") {
  const SimulationConfig c = parse_simulation_config(kMinimal, kData);
  JsonWriter w;
  write_resolved_json(w, c);
  const std::string j = w.str();
  CHECK_THAT(j, ContainsSubstring("\"particle_histories\": 1000000"));
  CHECK_THAT(j, ContainsSubstring("\"moment_center_m\": [0.1, 0.0, 0.0]"));
  CHECK_THAT(j, ContainsSubstring("\"mode\": \"tpmc\""));
}

TEST_CASE("Platform backend must match the build", "[config]") {
  const PlatformConfig own = parse_platform_config(std::string("platform:\n  backend: ") +
                                                   std::string(to_string(compiled_backend())) + "\n");
  CHECK_NOTHROW(check_platform_matches_build(own));
  const Backend other = compiled_backend() == Backend::CUDA ? Backend::HIP : Backend::CUDA;
  const PlatformConfig foreign =
      parse_platform_config(std::string("platform:\n  backend: ") + std::string(to_string(other)) + "\n");
  CHECK_THROWS_AS(check_platform_matches_build(foreign), ConfigError);
}

// ----------------------------------------------------------------------------
// Negative tests: every rule of the schema has at least one rejecting case.
// ----------------------------------------------------------------------------

TEST_CASE("Rejected: syntax and structure", "[config][negative]") {
  expect_invalid("", "configuration is empty");
  expect_invalid("simulation: [unclosed", "YAML syntax error");
  expect_invalid("- a\n- b\n", "must be a mapping");
  expect_invalid(std::string(kMinimal) + "platform:\n  backend: cpu\n", "platform: unknown key");
  expect_invalid(std::string(kMinimal) + "simulation:\n  name: twice\n", "duplicate key");
  expect_invalid(mutate("  mode: tpmc", "  mdoe: tpmc"), "simulation.mdoe: unknown key");
  expect_invalid(mutate("geometry:\n  file: body.stl", "geometry: body.stl"), "geometry: must be a mapping");
}

TEST_CASE("Rejected: missing required keys", "[config][negative]") {
  const auto [from, to, needle] = GENERATE(table<std::string, std::string, std::string>({
      {"  name: minimal\n", "", "simulation.name: required key is missing"},
      {"  mode: tpmc\n", "", "simulation.mode: required key is missing"},
      {"  random_seed: 1\n", "", "simulation.random_seed: required key is missing"},
      {"  file: body.stl\n", "  units: m\n", "geometry.file: required key is missing"},
      {"  area_m2: 2.0\n", "", "reference.area_m2: required key is missing"},
      {"  length_m: 0.5\n", "", "reference.length_m: required key is missing"},
      {"  moment_center_m: [0.1, 0.0, 0.0]\n", "", "reference.moment_center_m: required key is missing"},
      {"  lift_axis_body: [0.0, 0.0, 1.0]\n", "", "reference.lift_axis_body: required key is missing"},
      {"  number_density_m3: 1.0e15\n", "", "gas.number_density_m3: required key is missing"},
      {"  translational_temperature_K: 1000.0\n", "", "gas.translational_temperature_K: required key is missing"},
      {"  bulk_velocity_body_mps: [-7500.0, 0.0, 0.0]\n", "", "gas.bulk_velocity_body_mps: required key is missing"},
      {"  species:\n    O: 1.0\n", "", "gas.species: required key is missing"},
      {"  model: maxwell\n", "", "surface.model: required key is missing"},
      {"  temperature_K: 300.0\n", "", "surface.temperature_K: required key is missing"},
      {"  maxwell:\n    diffuse_fraction: 1.0\n", "", "surface.maxwell: required when model is 'maxwell'"},
      {"    diffuse_fraction: 1.0\n", "    {}\n", "surface.maxwell"},
  }));
  INFO(needle);
  expect_invalid(mutate(from, to), needle);
}

TEST_CASE("Rejected: missing sections", "[config][negative]") {
  const auto section = GENERATE(as<std::string>{}, "simulation", "geometry", "reference", "gas", "surface");
  std::string y = kMinimal;
  const auto start = y.find("\n" + section + ":");
  REQUIRE(start != std::string::npos);
  auto end = y.find("\n", start + 1);
  while (end != std::string::npos && end + 1 < y.size() && y[end + 1] == ' ') end = y.find("\n", end + 1);
  y.erase(start, (end == std::string::npos ? y.size() : end) - start);
  expect_invalid(y, section + ": required key is missing");
}

TEST_CASE("Rejected: wrong types", "[config][negative]") {
  const auto [from, to, needle] = GENERATE(table<std::string, std::string, std::string>({
      {"random_seed: 1", "random_seed: -1", "simulation.random_seed: expected a non-negative integer"},
      {"random_seed: 1", "random_seed: 1.5", "simulation.random_seed: expected a non-negative integer"},
      {"random_seed: 1", "random_seed: 99999999999999999999", "at most 2^64-1"},
      {"random_seed: 1", "random_seed: '1'", "simulation.random_seed: expected a non-negative integer"},
      {"area_m2: 2.0", "area_m2: two", "reference.area_m2: expected a finite real number"},
      {"area_m2: 2.0", "area_m2: \"2.0\"", "reference.area_m2: expected a finite real number"},
      {"area_m2: 2.0", "area_m2: [2.0]", "reference.area_m2: expected a finite real number"},
      {"area_m2: 2.0", "area_m2: .nan", "reference.area_m2: expected a finite real number"},
      {"area_m2: 2.0", "area_m2: .inf", "reference.area_m2: expected a finite real number"},
      {"area_m2: 2.0", "area_m2: inf", "reference.area_m2: expected a finite real number"},
      {"area_m2: 2.0", "area_m2:", "reference.area_m2: has no value"},
      {"[0.1, 0.0, 0.0]", "[0.1, 0.0]", "reference.moment_center_m: expected a list of exactly 3 numbers"},
      {"[0.1, 0.0, 0.0]", "[0.1, 0.0, 0.0, 1.0]", "reference.moment_center_m: expected a list of exactly 3"},
      {"[0.1, 0.0, 0.0]", "[0.1, x, 0.0]", "reference.moment_center_m[1]: expected a finite real number"},
      {"[0.1, 0.0, 0.0]", "0.1", "reference.moment_center_m: expected a list"},
      {"name: minimal", "name: [a, b]", "simulation.name: expected a string"},
      {"name: minimal", "name: \"\"", "simulation.name: must not be empty"},
      {"name: minimal", "name: my run", "simulation.name: may only contain"},
      {"    O: 1.0", "    O: lots", "gas.species.O: mole fraction must be a finite real number"},
      {"    O: 1.0", "    - O", "gas.species: must be a non-empty mapping"},
  }));
  INFO(needle);
  expect_invalid(mutate(from, to), needle);
}

TEST_CASE("Rejected: values out of range", "[config][negative]") {
  const auto [from, to, needle] = GENERATE(table<std::string, std::string, std::string>({
      {"area_m2: 2.0", "area_m2: 0.0", "reference.area_m2: value 0 is outside the allowed range (0, inf]"},
      {"area_m2: 2.0", "area_m2: -1", "reference.area_m2: value -1 is outside"},
      {"length_m: 0.5", "length_m: 0", "reference.length_m: value 0 is outside"},
      {"lift_axis_body: [0.0, 0.0, 1.0]", "lift_axis_body: [0.0, 0.0, 0.0]",
       "reference.lift_axis_body: must be a non-zero vector"},
      {"lift_axis_body: [0.0, 0.0, 1.0]", "lift_axis_body: [-2.0, 0.0, 0.0]",
       "reference.lift_axis_body: is (numerically) parallel"},
      {"number_density_m3: 1.0e15", "number_density_m3: 0", "gas.number_density_m3: value 0 is outside"},
      {"translational_temperature_K: 1000.0", "translational_temperature_K: -5",
       "gas.translational_temperature_K: value -5 is outside"},
      {"[-7500.0, 0.0, 0.0]", "[0, 0, 0]", "gas.bulk_velocity_body_mps: must be non-zero"},
      {"  temperature_K: 300.0", "  temperature_K: 0", "surface.temperature_K: value 0 is outside"},
      {"diffuse_fraction: 1.0", "diffuse_fraction: 1.0001", "surface.maxwell.diffuse_fraction: value 1.0001"},
      {"diffuse_fraction: 1.0", "diffuse_fraction: -0.1", "surface.maxwell.diffuse_fraction: value -0.1"},
      {"    O: 1.0", "    O: 0.9", "gas.species: mole fractions must sum to 1"},
      {"    O: 1.0", "    O: 0.6\n    N: 0.6", "gas.species: mole fractions must sum to 1"},
      {"    O: 1.0", "    O: 1.2\n    N: -0.2", "gas.species.N: value -0.2 is outside the allowed range [0, 1]"},
  }));
  INFO(needle);
  expect_invalid(mutate(from, to), needle);
}

TEST_CASE("Rejected: CLL parameters out of range", "[config][negative]") {
  const auto [alpha_n, sigma_t, needle] = GENERATE(table<std::string, std::string, std::string>({
      {"1.01", "1.0", "surface.cll.alpha_n: value 1.01 is outside the allowed range [0, 1]"},
      {"-0.01", "1.0", "surface.cll.alpha_n: value -0.01"},
      {"0.5", "2.01", "surface.cll.sigma_t: value 2.01 is outside the allowed range [0, 2]"},
      {"0.5", "-0.1", "surface.cll.sigma_t: value -0.1"},
  }));
  INFO(needle);
  expect_invalid(mutate("  model: maxwell\n  temperature_K: 300.0\n  maxwell:\n    diffuse_fraction: 1.0",
                        "  model: cll\n  temperature_K: 300.0\n  cll:\n    alpha_n: " + alpha_n +
                            "\n    sigma_t: " + sigma_t),
                 needle);
  expect_invalid(mutate("  model: maxwell", "  model: cll"), "surface.cll: required when model is 'cll'");
  expect_invalid(mutate("    diffuse_fraction: 1.0", "    diffuse_fraction: 1.0\n  cll:\n    alpha_n: 0.5"),
                 "surface.cll.sigma_t: required key is missing");
  expect_invalid(mutate("    diffuse_fraction: 1.0", "    diffuse_fraction: 1.0\n    alpha_n: 0.5"),
                 "surface.maxwell.alpha_n: unknown key");
}

TEST_CASE("Rejected: invalid enumerations and unsupported features", "[config][negative]") {
  const auto [from, to, needle] = GENERATE(table<std::string, std::string, std::string>({
      {"mode: tpmc", "mode: pic", "simulation.mode: invalid value 'pic' (allowed: tpmc, dsmc)"},
      {"model: maxwell", "model: specular", "surface.model: invalid value 'specular' (allowed: maxwell, cll)"},
      {"file: body.stl", "file: body.stl\n  units: km", "geometry.units: invalid value 'km'"},
      {"file: body.stl", "file: body.stl\n  units: in", "geometry.units: 'in' is not supported yet"},
      {"    O: 1.0", "    Xe: 1.0", "gas.species.Xe: unknown species (supported: O, N, He)"},
      {"    O: 1.0", "    N2: 1.0", "gas.species.N2: molecular species are not supported yet"},
      {"    O: 1.0", "    O2: 0.5\n    O: 0.5", "gas.species.O2: molecular species are not supported yet"},
  }));
  INFO(needle);
  expect_invalid(mutate(from, to), needle);
}

TEST_CASE("Rejected: invalid optional sections", "[config][negative]") {
  const auto [extra, needle] = GENERATE(table<std::string, std::string>({
      {"tpmc:\n  particle_histories: 0\n", "tpmc.particle_histories: must be at least 1"},
      {"tpmc:\n  particle_histories: 2.0e6\n", "tpmc.particle_histories: expected a non-negative integer"},
      {"tpmc:\n  max_reflections: 0\n", "tpmc.max_reflections: must be in [1, 4294967295]"},
      {"tpmc:\n  max_reflections: 4294967296\n", "tpmc.max_reflections: must be in [1, 4294967295]"},
      {"tpmc:\n  histories: 10\n", "tpmc.histories: unknown key"},
      {"tpmc: 5\n", "tpmc: must be a mapping"},
      {"tpmc:\n", "tpmc: has no value"},
      {"dsmc:\n  collision_model: VSS\n", "dsmc.collision_model: 'VSS' is not supported yet"},
      {"dsmc:\n  collision_model: hard_sphere\n", "dsmc.collision_model: invalid value"},
      {"dsmc:\n  timestep_mode: fixed\n", "dsmc.timestep_s: required key is missing"},
      {"dsmc:\n  timestep_mode: fixed\n  timestep_s: -1e-6\n", "dsmc.timestep_s: value -1e-06 is outside"},
      {"dsmc:\n  timestep_s: 1e-6\n", "dsmc.timestep_s: only allowed when timestep_mode is 'fixed'"},
      {"dsmc:\n  cell_size_mode: fixed\n", "dsmc.cell_size_m: required key is missing"},
      {"dsmc:\n  cell_size_m: 0.1\n", "dsmc.cell_size_m: only allowed when cell_size_mode is 'fixed'"},
      {"dsmc:\n  timestep_mode: sometimes\n", "dsmc.timestep_mode: invalid value 'sometimes'"},
      {"dsmc:\n  particles_per_cell_target: 0\n", "dsmc.particles_per_cell_target: must be in"},
      {"statistics:\n  confidence_level: 1.0\n", "statistics.confidence_level: value 1 is outside the allowed range (0, 1)"},
      {"statistics:\n  confidence_level: 0\n", "statistics.confidence_level: value 0 is outside"},
      {"statistics:\n  report_uncertainty: yes\n", "statistics.report_uncertainty: expected true or false"},
      {"statistics:\n  report_uncertainty: 1\n", "statistics.report_uncertainty: expected true or false"},
      {"output:\n  format: xml\n", "output.format: invalid value 'xml' (allowed: json, csv, hdf5)"},
      {"output:\n  directory: \"\"\n", "output.directory: must not be empty"},
      {"output:\n  save_surface_fields: maybe\n", "output.save_surface_fields: expected true or false"},
  }));
  INFO(needle);
  expect_invalid(std::string(kMinimal) + extra, needle);
}

TEST_CASE("Rejected: material file problems", "[config][negative]") {
  const std::string surface = "  model: maxwell\n  temperature_K: 300.0\n  maxwell:\n    diffuse_fraction: 1.0";
  expect_invalid(mutate(surface, "  material_file: does_not_exist.yaml"), "surface.material_file: cannot open");
  expect_invalid(mutate(surface, "  material_file: material_cll.yaml\n  temperature_K: 300.0"),
                 "surface.temperature_K: unknown key");
  expect_invalid(mutate(surface, "  material_file: material_bad.yaml"),
                 "surface(material_bad.yaml).surface.cll.alpha_n: value 1.5 is outside");
  expect_invalid(mutate(surface, "  material_file: material_nested.yaml"),
                 "material files cannot reference other material files");
}

TEST_CASE("All issues are reported at once, with line numbers", "[config][negative]") {
  std::string y = mutate("area_m2: 2.0", "area_m2: -2.0");
  y = std::string(y).replace(y.find("mode: tpmc"), 10, "mode: xyz");
  y = std::string(y).replace(y.find("diffuse_fraction: 1.0"), 21, "diffuse_fraction: 3");
  try {
    (void)parse_simulation_config(y, kData);
    FAIL("should be rejected");
  } catch (const ConfigError& e) {
    INFO(e.what());
    CHECK(e.issues().size() == 3);
    CHECK_THAT(std::string(e.what()), ContainsSubstring("(3 issues)"));
    CHECK_THAT(std::string(e.what()), ContainsSubstring("simulation.mode: invalid value 'xyz' (allowed: tpmc, dsmc) (line 4)"));
  }
}

TEST_CASE("Rejected: platform configurations", "[config][negative]") {
  auto reject = [](const std::string& y, const std::string& needle) {
    try {
      (void)parse_platform_config(y);
      FAIL("accepted: " << y);
    } catch (const ConfigError& e) {
      CHECK_THAT(std::string(e.what()), ContainsSubstring(needle));
    }
  };
  reject("", "platform configuration is empty");
  reject("platform:\n  backend: sycl\n", "platform.backend: invalid value 'sycl' (allowed: cpu, cuda, hip)");
  reject("platform:\n  device_id: 0\n", "platform.backend: required key is missing");
  reject("platform:\n  backend: cuda\n  device_id: -1\n", "platform.device_id: expected a non-negative integer");
  reject("platform:\n  backend: cuda\n  gpu: 0\n", "platform.gpu: unknown key");
  reject("backend: cuda\n", "backend: unknown key");
  CHECK_THROWS_AS(load_platform_config(kData / "no_such_platform.yaml"), ConfigError);
  CHECK_THROWS_AS(load_simulation_config(kData / "no_such_config.yaml"), ConfigError);
}

// ----------------------------------------------------------------------------
// M1 additions: domain, geometry.require_closed, tpmc.batch_size
// ----------------------------------------------------------------------------

TEST_CASE("Domain, require_closed and batch_size", "[config]") {
  const SimulationConfig d = parse_simulation_config(kMinimal, kData);
  CHECK(d.geometry.require_closed);
  CHECK(!d.domain.margin_m);
  CHECK(!d.domain.box_min_m);
  CHECK(d.tpmc.batch_size == (1u << 20));

  const SimulationConfig m = parse_simulation_config(
      std::string(kMinimal) + "domain:\n  margin_m: 0.25\ntpmc:\n  batch_size: 4096\n", kData);
  CHECK(*m.domain.margin_m == 0.25);
  CHECK(m.tpmc.batch_size == 4096);

  const SimulationConfig b = parse_simulation_config(
      mutate("file: body.stl", "file: body.stl\n  require_closed: false") +
          "domain:\n  box_min_m: [-1, -2, -3]\n  box_max_m: [1, 2, 3]\n",
      kData);
  CHECK(!b.geometry.require_closed);
  CHECK(*b.domain.box_min_m == Vec3(-1, -2, -3));
  CHECK(*b.domain.box_max_m == Vec3(1, 2, 3));
  // Round trip keeps the domain block.
  const std::string yaml = to_resolved_yaml(b);
  CHECK_THAT(yaml, ContainsSubstring("box_max_m: [1.0, 2.0, 3.0]"));
  CHECK_THAT(yaml, ContainsSubstring("require_closed: false"));
  CHECK(to_resolved_yaml(parse_simulation_config(yaml, "/")) == yaml);
}

TEST_CASE("Rejected: invalid domain and M1 options", "[config][negative]") {
  const auto [extra, needle] = GENERATE(table<std::string, std::string>({
      {"domain:\n  margin_m: 0\n", "domain.margin_m: value 0 is outside"},
      {"domain:\n  margin_m: 1\n  box_min_m: [0, 0, 0]\n", "give either margin_m or box_min_m/box_max_m"},
      {"domain:\n  box_min_m: [0, 0, 0]\n", "domain.box_max_m: required key is missing"},
      {"domain:\n  box_min_m: [0, 0, 0]\n  box_max_m: [1, 0, 1]\n", "box_min_m must be smaller than box_max_m"},
      {"domain:\n  size: 3\n", "domain.size: unknown key"},
      {"tpmc:\n  batch_size: 0\n", "tpmc.batch_size: must be in [1, 2147483648]"},
      {"tpmc:\n  batch_size: 2147483649\n", "tpmc.batch_size: must be in [1, 2147483648]"},
  }));
  INFO(needle);
  expect_invalid(std::string(kMinimal) + extra, needle);
  expect_invalid(mutate("file: body.stl", "file: body.stl\n  require_closed: no"),
                 "geometry.require_closed: expected true or false");
}
