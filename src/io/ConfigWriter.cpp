#include "vleo/io/ConfigWriter.hpp"

#include <yaml-cpp/yaml.h>

#include "YamlHelpers.hpp"

namespace vleo {

namespace {

// Both writers walk the configuration through this small interface so that
// the YAML and JSON outputs cannot drift apart.
template <class Sink>
void walk(const SimulationConfig& c, Sink& s) {
  s.begin("simulation");
  s.str("name", c.simulation.name);
  s.str("mode", to_string(c.simulation.mode));
  s.u64("random_seed", c.simulation.random_seed);
  s.end();

  s.begin("geometry");
  s.str("file", c.geometry.resolved_file.string());
  s.str("units", to_string(c.geometry.units));
  s.boolean("require_closed", c.geometry.require_closed);
  s.end();

  if (c.domain.margin_m || c.domain.box_min_m) {
    s.begin("domain");
    if (c.domain.margin_m) s.dbl("margin_m", *c.domain.margin_m);
    if (c.domain.box_min_m) s.vec("box_min_m", *c.domain.box_min_m);
    if (c.domain.box_max_m) s.vec("box_max_m", *c.domain.box_max_m);
    s.end();
  }

  s.begin("reference");
  s.dbl("area_m2", c.reference.area_m2);
  s.dbl("length_m", c.reference.length_m);
  s.vec("moment_center_m", c.reference.moment_center_m);
  s.vec("lift_axis_body", c.reference.lift_axis_body);
  s.end();

  s.begin("gas");
  s.dbl("number_density_m3", c.gas.number_density_m3);
  s.dbl("translational_temperature_K", c.gas.translational_temperature_K);
  s.vec("bulk_velocity_body_mps", c.gas.bulk_velocity_body_mps);
  s.begin("species");
  for (const auto& sp : c.gas.species) s.dbl(sp.name, sp.mole_fraction);
  s.end();
  s.end();

  s.begin("surface");
  s.str("model", to_string(c.surface.model));
  s.dbl("temperature_K", c.surface.temperature_K);
  if (c.surface.maxwell) {
    s.begin("maxwell");
    s.dbl("diffuse_fraction", c.surface.maxwell->diffuse_fraction);
    s.end();
  }
  if (c.surface.cll) {
    s.begin("cll");
    s.dbl("alpha_n", c.surface.cll->alpha_n);
    s.dbl("sigma_t", c.surface.cll->sigma_t);
    s.end();
  }
  s.end();

  s.begin("tpmc");
  s.u64("particle_histories", c.tpmc.particle_histories);
  s.u64("max_reflections", c.tpmc.max_reflections);
  s.u64("batch_size", c.tpmc.batch_size);
  s.end();

  s.begin("dsmc");
  s.str("collision_model", to_string(c.dsmc.collision_model));
  s.str("timestep_mode", to_string(c.dsmc.timestep_mode));
  if (c.dsmc.timestep_s) s.dbl("timestep_s", *c.dsmc.timestep_s);
  s.str("cell_size_mode", to_string(c.dsmc.cell_size_mode));
  if (c.dsmc.cell_size_m) s.dbl("cell_size_m", *c.dsmc.cell_size_m);
  s.u64("particles_per_cell_target", c.dsmc.particles_per_cell_target);
  s.end();

  s.begin("statistics");
  s.dbl("confidence_level", c.statistics.confidence_level);
  s.boolean("report_uncertainty", c.statistics.report_uncertainty);
  s.end();

  s.begin("output");
  s.str("directory", c.output.directory);
  s.str("format", to_string(c.output.format));
  s.boolean("save_surface_fields", c.output.save_surface_fields);
  s.end();
}

struct YamlSink {
  YAML::Emitter& e;
  void begin(std::string_view k) { e << YAML::Key << std::string(k) << YAML::Value << YAML::BeginMap; }
  void end() { e << YAML::EndMap; }
  void str(std::string_view k, std::string_view v) {
    e << YAML::Key << std::string(k) << YAML::Value << YAML::DoubleQuoted << std::string(v);
  }
  void dbl(std::string_view k, double v) { e << YAML::Key << std::string(k) << YAML::Value << io_detail::format_real(v); }
  void u64(std::string_view k, std::uint64_t v) { e << YAML::Key << std::string(k) << YAML::Value << v; }
  void boolean(std::string_view k, bool v) { e << YAML::Key << std::string(k) << YAML::Value << YAML::TrueFalseBool << YAML::LowerCase << v; }
  void vec(std::string_view k, const Vec3& v) {
    e << YAML::Key << std::string(k) << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int i = 0; i < 3; ++i) e << io_detail::format_real(v[i]);
    e << YAML::EndSeq;
  }
};

struct JsonSink {
  JsonWriter& w;
  void begin(std::string_view k) { w.key(k).begin_object(); }
  void end() { w.end_object(); }
  void str(std::string_view k, std::string_view v) { w.key(k).value(v); }
  void dbl(std::string_view k, double v) { w.key(k).value(v); }
  void u64(std::string_view k, std::uint64_t v) { w.key(k).value(v); }
  void boolean(std::string_view k, bool v) { w.key(k).value(v); }
  void vec(std::string_view k, const Vec3& v) { w.key(k).value(v); }
};

}  // namespace

std::string to_resolved_yaml(const SimulationConfig& cfg) {
  YAML::Emitter e;
  e << YAML::Comment("Resolved configuration written by vleo_solver: all defaults applied.");
  e << YAML::Newline;
  e << YAML::BeginMap;
  YamlSink sink{e};
  walk(cfg, sink);
  e << YAML::EndMap;
  return std::string(e.c_str()) + "\n";
}

void write_resolved_json(JsonWriter& w, const SimulationConfig& cfg) {
  w.begin_object();
  JsonSink sink{w};
  walk(cfg, sink);
  w.end_object();
}

void write_derived_json(JsonWriter& w, const SimulationConfig& cfg) {
  const DerivedQuantities& d = cfg.derived;
  w.begin_object();
  w.field("mole_fraction_sum", d.mole_fraction_sum);
  w.field("mean_molecular_mass_kg", d.mean_molecular_mass_kg);
  w.field("mass_density_kg_m3", d.mass_density_kg_m3);
  w.field("freestream_speed_mps", d.freestream_speed_mps);
  w.field("dynamic_pressure_Pa", d.dynamic_pressure_Pa);
  w.field("geometry_scale_to_m", d.geometry_scale_to_m);
  w.key("aero_axes_body").begin_object();
  w.field("drag_e_D", d.aero_axes.drag);
  w.field("side_e_Y", d.aero_axes.side);
  w.field("lift_e_L", d.aero_axes.lift);
  w.end_object();
  w.field("lift_axis_was_projected", d.lift_axis_was_projected);
  w.key("species").begin_array();
  for (const auto& s : cfg.gas.species) {
    w.begin_object();
    w.field("name", s.name);
    w.field("mole_fraction", s.mole_fraction);
    w.field("normalised_mole_fraction", s.mole_fraction / d.mole_fraction_sum);
    w.field("mass_kg", s.mass_kg);
    w.end_object();
  }
  w.end_array();
  if (!cfg.surface.material_file.empty()) w.field("surface_material_file", cfg.surface.material_file.string());
  w.end_object();
}

}  // namespace vleo
