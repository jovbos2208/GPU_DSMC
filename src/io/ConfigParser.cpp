#include "vleo/io/ConfigParser.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#include "YamlHelpers.hpp"
#include "vleo/core/Species.hpp"

namespace vleo {

namespace fs = std::filesystem;
using namespace io_detail;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr Range kPositive{0.0, kInf, false, true};
constexpr Range kUnitInterval{0.0, 1.0, true, true};

// Tolerance for the sum of species mole fractions.
constexpr double kFractionSumTol = 1.0e-6;

constexpr std::array<EnumEntry<SolverMode>, 2> kModes{{{"tpmc", SolverMode::TPMC}, {"dsmc", SolverMode::DSMC}}};
constexpr std::array<EnumEntry<LengthUnit>, 3> kUnits{
    {{"m", LengthUnit::Metre}, {"cm", LengthUnit::Centimetre}, {"mm", LengthUnit::Millimetre}}};
constexpr std::array<EnumEntry<GSIModel>, 2> kGsi{{{"maxwell", GSIModel::Maxwell}, {"cll", GSIModel::CLL}}};
constexpr std::array<EnumEntry<CollisionModel>, 1> kCollision{{{"VHS", CollisionModel::VHS}}};
constexpr std::array<EnumEntry<ParameterMode>, 2> kParamMode{
    {{"automatic", ParameterMode::Automatic}, {"fixed", ParameterMode::Fixed}}};
constexpr std::array<EnumEntry<ConvergenceQuantity>, 7> kConvQuantities{{{"C_D", ConvergenceQuantity::CD},
                                                                          {"C_L", ConvergenceQuantity::CL},
                                                                          {"C_Y", ConvergenceQuantity::CY},
                                                                          {"C_Mx", ConvergenceQuantity::CMx},
                                                                          {"C_My", ConvergenceQuantity::CMy},
                                                                          {"C_Mz", ConvergenceQuantity::CMz},
                                                                          {"heat_rate_W", ConvergenceQuantity::HeatRate}}};
constexpr std::array<EnumEntry<OutputFormat>, 3> kFormats{
    {{"json", OutputFormat::JSON}, {"csv", OutputFormat::CSV}, {"hdf5", OutputFormat::HDF5}}};

bool valid_name(const std::string& s) {
  for (char c : s) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                    c == '-' || c == '.';
    if (!ok) return false;
  }
  return !s.empty();
}

fs::path resolve(const fs::path& base_dir, const std::string& p) {
  fs::path path(p);
  if (path.is_relative()) path = base_dir / path;
  return path.lexically_normal();
}

// Fetch an optional section; returns an undefined node if absent.
YAML::Node section(const YAML::Node& root, std::string_view key, Issues& is, Need need) {
  const auto n = child(root, key, "", is, need);
  return n ? *n : YAML::Node(YAML::NodeType::Undefined);
}

void parse_simulation(const YAML::Node& n, SimulationSection& out, Issues& is) {
  const std::string p = "simulation";
  if (!check_map(n, p, {"name", "mode", "random_seed"}, is)) return;
  if (auto v = get_string(n, "name", p, is, Need::Required)) {
    if (!valid_name(*v))
      is.add(p + ".name", "may only contain letters, digits, '_', '-' and '.' (got '" + *v + "')", &n);
    else
      out.name = *v;
  }
  if (auto v = get_enum(n, "mode", p, is, Need::Required, kModes)) out.mode = *v;
  if (auto v = get_uint64(n, "random_seed", p, is, Need::Required)) out.random_seed = *v;
}

void parse_geometry(const YAML::Node& n, const fs::path& base_dir, GeometrySection& out, Issues& is) {
  const std::string p = "geometry";
  if (!check_map(n, p, {"file", "units", "require_closed"}, is)) return;
  if (auto v = get_string(n, "file", p, is, Need::Required)) {
    out.file = *v;
    out.resolved_file = resolve(base_dir, *v);
  }
  if (auto v = get_enum(n, "units", p, is, Need::Optional, kUnits, {{"in", "use m, cm or mm"}})) out.units = *v;
  if (auto v = get_bool(n, "require_closed", p, is, Need::Optional)) out.require_closed = *v;
}

void parse_domain(const YAML::Node& n, DomainSection& out, Issues& is) {
  const std::string p = "domain";
  if (!n.IsDefined()) return;
  if (!check_map(n, p, {"margin_m", "box_min_m", "box_max_m"}, is)) return;
  const bool has_margin = n["margin_m"].IsDefined();
  const bool has_min = n["box_min_m"].IsDefined();
  const bool has_max = n["box_max_m"].IsDefined();
  if (has_margin && (has_min || has_max)) {
    is.add(p, "give either margin_m or box_min_m/box_max_m, not both", &n);
    return;
  }
  if (has_margin) {
    if (auto v = get_double_in(n, "margin_m", p, is, Need::Required, kPositive)) out.margin_m = *v;
    return;
  }
  if (has_min || has_max) {
    const auto lo = get_vec3(n, "box_min_m", p, is, Need::Required);
    const auto hi = get_vec3(n, "box_max_m", p, is, Need::Required);
    if (lo && hi) {
      if (!(lo->x < hi->x && lo->y < hi->y && lo->z < hi->z))
        is.add(p, "box_min_m must be smaller than box_max_m in every component", &n);
      else {
        out.box_min_m = *lo;
        out.box_max_m = *hi;
      }
    }
  }
}

void parse_reference(const YAML::Node& n, ReferenceSection& out, Issues& is) {
  const std::string p = "reference";
  if (!check_map(n, p, {"area_m2", "length_m", "moment_center_m", "lift_axis_body"}, is)) return;
  if (auto v = get_double_in(n, "area_m2", p, is, Need::Required, kPositive)) out.area_m2 = *v;
  if (auto v = get_double_in(n, "length_m", p, is, Need::Required, kPositive)) out.length_m = *v;
  if (auto v = get_vec3(n, "moment_center_m", p, is, Need::Required)) out.moment_center_m = *v;
  if (auto v = get_vec3(n, "lift_axis_body", p, is, Need::Required)) {
    if (norm2(*v) == 0.0)
      is.add(p + ".lift_axis_body", "must be a non-zero vector");
    else
      out.lift_axis_body = *v;
  }
}

void parse_species(const YAML::Node& n, const std::string& p, std::vector<SpeciesFraction>& out, Issues& is) {
  if (!n.IsMap() || n.size() == 0) {
    is.add(p, "must be a non-empty mapping species: mole_fraction", &n);
    return;
  }
  std::vector<SpeciesFraction> list;
  bool ok = true;
  std::set<std::string> seen;
  for (auto it = n.begin(); it != n.end(); ++it) {
    const std::string name = it->first.IsScalar() ? it->first.Scalar() : std::string();
    const std::string sp = join_path(p, name);
    if (!seen.insert(name).second) {
      is.add(sp, "duplicate species", &it->first);
      ok = false;
      continue;
    }
    const auto info = find_species(name);
    if (!info) {
      std::string list_str;
      for (const auto& s : known_species())
        if (s.supported) list_str += (list_str.empty() ? "" : ", ") + std::string(s.name);
      is.add(sp, "unknown species (supported: " + list_str + ")", &it->first);
      ok = false;
      continue;
    }
    if (!info->supported) {
      is.add(sp,
             "molecular species are not supported yet: their rotational/vibrational energy modes "
             "(milestone M4) are required for a correct heat transfer",
             &it->first);
      ok = false;
      continue;
    }
    const auto x = scalar_to_double(it->second);
    if (!x) {
      is.add(sp, "mole fraction must be a finite real number", &it->second);
      ok = false;
      continue;
    }
    if (!check_range(*x, kUnitInterval, sp, is, &it->second)) {
      ok = false;
      continue;
    }
    list.push_back({name, *x, info->mass_kg, info->vhs});
  }
  if (!ok) return;

  double sum = 0.0;
  for (const auto& s : list) sum += s.mole_fraction;
  if (!(std::fabs(sum - 1.0) <= kFractionSumTol)) {
    is.add(p, "mole fractions must sum to 1 (within " + format_double(kFractionSumTol) + "), got " + format_double(sum),
           &n);
    return;
  }
  // Fractions are kept exactly as given (so the resolved configuration
  // round-trips bit-exactly); consumers normalise by their sum.
  out = std::move(list);
}

void parse_gas(const YAML::Node& n, GasSection& out, Issues& is) {
  const std::string p = "gas";
  if (!check_map(n, p, {"number_density_m3", "translational_temperature_K", "bulk_velocity_body_mps", "species"}, is))
    return;
  if (auto v = get_double_in(n, "number_density_m3", p, is, Need::Required, kPositive)) out.number_density_m3 = *v;
  if (auto v = get_double_in(n, "translational_temperature_K", p, is, Need::Required, kPositive)) out.translational_temperature_K = *v;
  if (auto v = get_vec3(n, "bulk_velocity_body_mps", p, is, Need::Required)) {
    if (norm2(*v) == 0.0)
      is.add(p + ".bulk_velocity_body_mps", "must be non-zero (it defines the drag direction)");
    else
      out.bulk_velocity_body_mps = *v;
  }
  if (auto s = child(n, "species", p, is, Need::Required)) parse_species(*s, p + ".species", out.species, is);
}

void parse_surface_block(const YAML::Node& n, const std::string& p, SurfaceSection& out, Issues& is) {
  if (!check_map(n, p, {"model", "temperature_K", "maxwell", "cll"}, is)) return;
  const auto model = get_enum(n, "model", p, is, Need::Required, kGsi);
  if (model) out.model = *model;
  if (auto v = get_double_in(n, "temperature_K", p, is, Need::Required, kPositive)) out.temperature_K = *v;

  if (auto m = child(n, "maxwell", p, is, Need::Optional)) {
    const std::string mp = p + ".maxwell";
    if (check_map(*m, mp, {"diffuse_fraction"}, is)) {
      if (auto v = get_double_in(*m, "diffuse_fraction", mp, is, Need::Required, kUnitInterval)) out.maxwell = MaxwellParameters{*v};
    }
  } else if (model == GSIModel::Maxwell) {
    is.add(p + ".maxwell", "required when model is 'maxwell'", &n);
  }

  if (auto c = child(n, "cll", p, is, Need::Optional)) {
    const std::string cp = p + ".cll";
    if (check_map(*c, cp, {"alpha_n", "sigma_t"}, is)) {
      const auto an = get_double_in(*c, "alpha_n", cp, is, Need::Required, kUnitInterval);
      const auto st = get_double_in(*c, "sigma_t", cp, is, Need::Required, Range{0.0, 2.0});
      if (an && st) out.cll = CLLParameters{*an, *st};
    }
  } else if (model == GSIModel::CLL) {
    is.add(p + ".cll", "required when model is 'cll'", &n);
  }
}

void parse_surface(const YAML::Node& n, const fs::path& base_dir, SurfaceSection& out, Issues& is) {
  const std::string p = "surface";
  if (n.IsMap() && n["material_file"].IsDefined()) {
    if (!check_map(n, p, {"material_file"}, is)) return;
    const auto file = get_string(n, "material_file", p, is, Need::Required);
    if (!file) return;
    const fs::path path = resolve(base_dir, *file);
    YAML::Node doc;
    try {
      doc = YAML::LoadFile(path.string());
    } catch (const YAML::BadFile&) {
      is.add(p + ".material_file", "cannot open '" + path.string() + "'", &n);
      return;
    } catch (const YAML::Exception& e) {
      is.add(p + ".material_file", "YAML error in '" + path.string() + "': " + e.what(), &n);
      return;
    }
    const std::string mp = "surface(" + path.filename().string() + ")";
    if (!check_map(doc, mp, {"surface"}, is)) return;
    const auto block = child(doc, "surface", mp, is, Need::Required);
    if (!block) return;
    if (block->IsMap() && (*block)["material_file"].IsDefined()) {
      is.add(mp + ".surface.material_file", "material files cannot reference other material files", &*block);
      return;
    }
    parse_surface_block(*block, mp + ".surface", out, is);
    out.material_file = path;
    return;
  }
  parse_surface_block(n, p, out, is);
}

void parse_tpmc(const YAML::Node& n, TPMCSection& out, Issues& is) {
  const std::string p = "tpmc";
  if (!n.IsDefined()) return;
  if (!check_map(n, p, {"particle_histories", "max_reflections", "batch_size", "convergence"}, is)) return;
  if (auto c = child(n, "convergence", p, is, Need::Optional)) {
    const std::string cp = p + ".convergence";
    if (check_map(*c, cp, {"quantity", "relative_error", "absolute_error", "min_histories"}, is)) {
      ConvergenceCriterion crit;
      bool ok = true;
      if (auto q = get_enum(*c, "quantity", cp, is, Need::Required, kConvQuantities))
        crit.quantity = *q;
      else
        ok = false;
      const bool has_rel = (*c)["relative_error"].IsDefined(), has_abs = (*c)["absolute_error"].IsDefined();
      if (has_rel == has_abs) {
        is.add(cp, "give exactly one of relative_error or absolute_error", &*c);
        ok = false;
      } else if (has_rel) {
        crit.relative_error = get_double_in(*c, "relative_error", cp, is, Need::Required, kPositive);
        ok = ok && crit.relative_error.has_value();
      } else {
        crit.absolute_error = get_double_in(*c, "absolute_error", cp, is, Need::Required, kPositive);
        ok = ok && crit.absolute_error.has_value();
      }
      if (auto m = get_uint64(*c, "min_histories", cp, is, Need::Optional)) crit.min_histories = *m;
      if (ok) out.convergence = crit;
    }
  }
  if (auto v = get_uint64(n, "batch_size", p, is, Need::Optional)) {
    if (*v == 0 || *v > (std::uint64_t{1} << 31))
      is.add(p + ".batch_size", "must be in [1, 2147483648]");
    else
      out.batch_size = *v;
  }
  if (auto v = get_uint64(n, "particle_histories", p, is, Need::Optional)) {
    if (*v == 0)
      is.add(p + ".particle_histories", "must be at least 1");
    else
      out.particle_histories = *v;
  }
  if (auto v = get_uint64(n, "max_reflections", p, is, Need::Optional)) {
    if (*v == 0 || *v > std::numeric_limits<std::uint32_t>::max())
      is.add(p + ".max_reflections", "must be in [1, 4294967295]");
    else
      out.max_reflections = static_cast<std::uint32_t>(*v);
  }
}

// "<what>_mode: automatic | fixed" together with "<value_key>" (required iff fixed).
void parse_mode_value(const YAML::Node& n, const std::string& p, std::string_view mode_key, std::string_view value_key,
                      ParameterMode& mode, std::optional<double>& value, Issues& is) {
  if (auto m = get_enum(n, mode_key, p, is, Need::Optional, kParamMode)) mode = *m;
  const std::string vp = join_path(p, value_key);
  const bool present = n[std::string(value_key)].IsDefined();
  if (mode == ParameterMode::Fixed) {
    if (auto v = get_double_in(n, value_key, p, is, Need::Required, kPositive)) value = *v;
  } else if (present) {
    is.add(vp, "only allowed when " + std::string(mode_key) + " is 'fixed'");
  }
}

void parse_dsmc(const YAML::Node& n, DSMCSection& out, Issues& is) {
  const std::string p = "dsmc";
  if (!n.IsDefined()) return;
  if (!check_map(n, p,
                 {"collision_model", "timestep_mode", "timestep_s", "cell_size_mode", "cell_size_m",
                  "particles_per_cell_target", "warmup_steps", "sample_steps", "blocks", "max_particles"},
                 is))
    return;
  if (auto v = get_enum(n, "collision_model", p, is, Need::Optional, kCollision,
                        {{"VSS", "planned after VHS (see implementation plan M3/M4)"}}))
    out.collision_model = *v;
  parse_mode_value(n, p, "timestep_mode", "timestep_s", out.timestep_mode, out.timestep_s, is);
  parse_mode_value(n, p, "cell_size_mode", "cell_size_m", out.cell_size_mode, out.cell_size_m, is);
  if (auto v = get_uint64(n, "particles_per_cell_target", p, is, Need::Optional)) {
    if (*v == 0 || *v > std::numeric_limits<std::uint32_t>::max())
      is.add(p + ".particles_per_cell_target", "must be in [1, 4294967295]");
    else
      out.particles_per_cell_target = static_cast<std::uint32_t>(*v);
  }
  if (auto v = get_uint64(n, "warmup_steps", p, is, Need::Optional)) out.warmup_steps = *v;
  if (auto v = get_uint64(n, "sample_steps", p, is, Need::Optional)) {
    if (*v < 2)
      is.add(p + ".sample_steps", "must be at least 2");
    else
      out.sample_steps = *v;
  }
  if (auto v = get_uint64(n, "blocks", p, is, Need::Optional)) {
    if (*v < 2 || *v > 100000)
      is.add(p + ".blocks", "must be in [2, 100000]");
    else
      out.blocks = static_cast<std::uint32_t>(*v);
  }
  if (auto v = get_uint64(n, "max_particles", p, is, Need::Optional)) {
    if (*v == 0)
      is.add(p + ".max_particles", "must be at least 1");
    else
      out.max_particles = *v;
  }
  if (out.blocks > out.sample_steps) is.add(p + ".blocks", "must not exceed dsmc.sample_steps");
}

void parse_statistics(const YAML::Node& n, StatisticsSection& out, Issues& is) {
  const std::string p = "statistics";
  if (!n.IsDefined()) return;
  if (!check_map(n, p, {"confidence_level", "report_uncertainty"}, is)) return;
  if (auto v = get_double_in(n, "confidence_level", p, is, Need::Optional, Range{0.0, 1.0, false, false})) out.confidence_level = *v;
  if (auto v = get_bool(n, "report_uncertainty", p, is, Need::Optional)) out.report_uncertainty = *v;
}

void parse_output(const YAML::Node& n, OutputSection& out, Issues& is) {
  const std::string p = "output";
  if (!n.IsDefined()) return;
  if (!check_map(n, p, {"directory", "format", "save_surface_fields"}, is)) return;
  if (auto v = get_string(n, "directory", p, is, Need::Optional)) out.directory = *v;
  if (auto v = get_enum(n, "format", p, is, Need::Optional, kFormats)) out.format = *v;
  if (auto v = get_bool(n, "save_surface_fields", p, is, Need::Optional)) out.save_surface_fields = *v;
}

void compute_derived(SimulationConfig& cfg, Issues& is) {
  DerivedQuantities& d = cfg.derived;
  double fraction_sum = 0.0;
  double mass_sum = 0.0;
  for (const auto& s : cfg.gas.species) {
    fraction_sum += s.mole_fraction;
    mass_sum += s.mole_fraction * s.mass_kg;
  }
  d.mole_fraction_sum = fraction_sum;
  d.mean_molecular_mass_kg = mass_sum / fraction_sum;
  d.mass_density_kg_m3 = cfg.gas.number_density_m3 * d.mean_molecular_mass_kg;
  d.freestream_speed_mps = norm(cfg.gas.bulk_velocity_body_mps);
  d.dynamic_pressure_Pa = 0.5 * d.mass_density_kg_m3 * d.freestream_speed_mps * d.freestream_speed_mps;
  d.geometry_scale_to_m = length_unit_to_metre(cfg.geometry.units);

  const AeroAxesStatus st = make_aero_axes(cfg.gas.bulk_velocity_body_mps, cfg.reference.lift_axis_body, d.aero_axes);
  if (st == AeroAxesStatus::LiftParallelToFlow) {
    is.add("reference.lift_axis_body", "is (numerically) parallel to gas.bulk_velocity_body_mps; the lift "
                                       "direction must have a component orthogonal to the free stream");
    return;
  }
  if (st != AeroAxesStatus::Ok) {
    is.add("reference.lift_axis_body", "cannot construct aerodynamic axes");
    return;
  }
  const double cos_angle =
      dot(normalized(cfg.reference.lift_axis_body), d.aero_axes.drag);  // 0 for an exactly orthogonal input
  d.lift_axis_was_projected = std::fabs(cos_angle) > 1.0e-12;
}

SimulationConfig parse_root(const YAML::Node& root, const fs::path& base_dir) {
  Issues is;
  SimulationConfig cfg;
  if (!root.IsDefined() || root.IsNull()) {
    is.add("", "configuration is empty");
    is.throw_if_any("simulation configuration");
  }
  if (!check_map(root, "",
                 {"simulation", "geometry", "domain", "reference", "gas", "surface", "tpmc", "dsmc", "statistics",
                  "output"},
                 is)) {
    is.throw_if_any("simulation configuration");
  }

  if (auto n = child(root, "simulation", "", is, Need::Required)) parse_simulation(*n, cfg.simulation, is);
  if (auto n = child(root, "geometry", "", is, Need::Required)) parse_geometry(*n, base_dir, cfg.geometry, is);
  if (auto n = child(root, "reference", "", is, Need::Required)) parse_reference(*n, cfg.reference, is);
  if (auto n = child(root, "gas", "", is, Need::Required)) parse_gas(*n, cfg.gas, is);
  if (auto n = child(root, "surface", "", is, Need::Required)) parse_surface(*n, base_dir, cfg.surface, is);
  parse_domain(section(root, "domain", is, Need::Optional), cfg.domain, is);
  parse_tpmc(section(root, "tpmc", is, Need::Optional), cfg.tpmc, is);
  parse_dsmc(section(root, "dsmc", is, Need::Optional), cfg.dsmc, is);
  parse_statistics(section(root, "statistics", is, Need::Optional), cfg.statistics, is);
  parse_output(section(root, "output", is, Need::Optional), cfg.output, is);

  // DSMC scope of the current release (milestone M3): one monatomic species
  // with VHS collision data. Gas mixtures (pair collisions) follow in M4.
  if (cfg.simulation.mode == SolverMode::DSMC && !cfg.gas.species.empty()) {
    if (cfg.gas.species.size() > 1)
      is.add("gas.species", "DSMC currently supports a single species; gas mixtures (species-pair collisions) "
                            "follow in milestone M4 - use mode: tpmc for mixtures");
    for (const auto& sp : cfg.gas.species)
      if (!sp.vhs.available)
        is.add("gas.species." + sp.name, "no validated VHS collision data for DSMC yet (milestone M4.1); "
                                         "DSMC species available now: He, Ar");
  }

  // Cross-field checks need valid individual fields.
  if (is.empty()) compute_derived(cfg, is);

  is.throw_if_any("simulation configuration");
  return cfg;
}

}  // namespace

SimulationConfig parse_simulation_config(std::string_view yaml_text, const fs::path& base_dir) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string(yaml_text));
  } catch (const YAML::Exception& e) {
    throw ConfigError({"YAML syntax error at line " + std::to_string(e.mark.line + 1) + ", column " +
                       std::to_string(e.mark.column + 1) + ": " + e.msg},
                      "simulation configuration");
  }
  return parse_root(root, fs::absolute(base_dir).lexically_normal());
}

SimulationConfig load_simulation_config(const fs::path& file) {
  const fs::path abs = fs::absolute(file).lexically_normal();
  std::ifstream in(abs, std::ios::binary);
  if (!in) throw ConfigError({"cannot open configuration file '" + abs.string() + "'"}, "simulation configuration");
  std::ostringstream text;
  text << in.rdbuf();
  SimulationConfig cfg = parse_simulation_config(text.str(), abs.parent_path());
  cfg.source_file = abs;
  return cfg;
}

}  // namespace vleo
