#include "vleo/tpmc/Results.hpp"

#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "vleo/core/BuildInfo.hpp"
#include "vleo/io/JsonWriter.hpp"

namespace vleo {

namespace {

std::string num(double d) {
  char buf[32];
  const auto [p, ec] = std::to_chars(buf, buf + sizeof(buf), d);
  return ec == std::errc() ? std::string(buf, p) : std::string("nan");
}

struct Row {
  const char* name;
  const char* unit;
  const Estimate* e;
};

std::vector<Row> rows(const TpmcEstimates& e) {
  return {{"force_x", "N", &e.force_N[0]},
          {"force_y", "N", &e.force_N[1]},
          {"force_z", "N", &e.force_N[2]},
          {"moment_x", "N m", &e.moment_Nm[0]},
          {"moment_y", "N m", &e.moment_Nm[1]},
          {"moment_z", "N m", &e.moment_Nm[2]},
          {"C_D", "-", &e.C_D},
          {"C_L", "-", &e.C_L},
          {"C_Y", "-", &e.C_Y},
          {"C_Mx", "-", &e.C_M[0]},
          {"C_My", "-", &e.C_M[1]},
          {"C_Mz", "-", &e.C_M[2]},
          {"heat_rate", "W", &e.heat_rate_W},
          {"heat_rate_positive", "W", &e.heat_rate_positive_W},
          {"heat_rate_negative", "W", &e.heat_rate_negative_W},
          {"drag_force", "N", &e.drag_N},
          {"lift_force", "N", &e.lift_N},
          {"side_force", "N", &e.side_N}};
}

void vec3_est(JsonWriter& w, const char* key, const std::array<Estimate, 3>& a, double Estimate::*field) {
  w.field(key, Vec3(a[0].*field, a[1].*field, a[2].*field));
}

// Writes one block (value, std_error, ci_low or ci_high) with the main quantities.
void block(JsonWriter& w, const TpmcEstimates& e, double Estimate::*f) {
  w.begin_object();
  vec3_est(w, "force_N", e.force_N, f);
  vec3_est(w, "moment_Nm", e.moment_Nm, f);
  w.field("C_D", e.C_D.*f);
  w.field("C_L", e.C_L.*f);
  w.field("C_Y", e.C_Y.*f);
  vec3_est(w, "C_M", e.C_M, f);
  w.field("heat_rate_W", e.heat_rate_W.*f);
  w.field("heat_rate_positive_W", e.heat_rate_positive_W.*f);
  w.field("heat_rate_negative_W", e.heat_rate_negative_W.*f);
  w.field("drag_force_N", e.drag_N.*f);
  w.field("lift_force_N", e.lift_N.*f);
  w.field("side_force_N", e.side_N.*f);
  w.end_object();
}

}  // namespace

std::string results_json(const TpmcResult& r, const SimulationConfig& cfg, const std::string& timestamp_utc) {
  const BuildInfo& b = build_info();
  const TpmcEstimates& e = r.estimates;
  JsonWriter w;
  w.begin_object();
  w.field("schema", kResultsSchema);

  w.key("simulation").begin_object();
  w.field("name", cfg.simulation.name);
  w.field("mode", to_string(cfg.simulation.mode));
  w.field("backend", b.configure.backend);
  w.field("gsi_model", to_string(cfg.surface.model));
  w.field("random_seed", cfg.simulation.random_seed);
  w.end_object();

  w.key("reference").begin_object();
  w.field("area_m2", cfg.reference.area_m2);
  w.field("length_m", cfg.reference.length_m);
  w.field("moment_center_m", cfg.reference.moment_center_m);
  w.field("dynamic_pressure_Pa", cfg.derived.dynamic_pressure_Pa);
  w.field("mass_density_kg_m3", cfg.derived.mass_density_kg_m3);
  w.field("freestream_speed_mps", cfg.derived.freestream_speed_mps);
  w.field("drag_axis_e_D", cfg.derived.aero_axes.drag);
  w.field("lift_axis_e_L", cfg.derived.aero_axes.lift);
  w.field("side_axis_e_Y", cfg.derived.aero_axes.side);
  w.end_object();

  w.key("results");
  block(w, e, &Estimate::value);
  w.key("std_error");
  block(w, e, &Estimate::std_error);
  w.key("confidence_interval").begin_object();
  w.key("low");
  block(w, e, &Estimate::ci_low);
  w.key("high");
  block(w, e, &Estimate::ci_high);
  w.end_object();

  w.key("statistics").begin_object();
  w.field("confidence_level", cfg.statistics.confidence_level);
  w.field("critical_value", r.critical_value);
  w.field("method", "independent histories: std_error = R * sqrt(var(x_h) / N)");
  w.field("histories", r.histories);
  w.field("batches", r.batches);
  w.field("converged", r.converged);
  w.key("convergence");
  if (cfg.tpmc.convergence) {
    const auto& c = *cfg.tpmc.convergence;
    w.begin_object();
    w.field("quantity", to_string(c.quantity));
    if (c.relative_error) w.field("relative_error_target", *c.relative_error);
    if (c.absolute_error) w.field("absolute_error_target", *c.absolute_error);
    const Estimate& q = select(e, c.quantity);
    w.field("achieved_std_error", q.std_error);
    w.field("achieved_relative_error", q.value != 0.0 ? q.std_error / std::fabs(q.value) : 0.0);
    w.field("min_histories", c.min_histories);
    w.end_object();
  } else {
    w.null();
  }
  w.end_object();

  w.key("diagnostics").begin_object();
  w.field("inflow_rate_per_s", r.inflow_rate_per_s);
  w.field("event_rate_factor_per_s", r.event_rate_factor_per_s);
  w.field("injected", r.counters.injected);
  w.field("exited", r.counters.exited);
  w.field("truncated", r.counters.truncated);
  w.field("lost_inside_body", r.counters.lost_inside_body);
  w.field("invalid", r.counters.invalid);
  w.field("surface_interactions", r.counters.surface_interactions);
  w.field("histories_with_hits", r.counters.histories_with_hits);
  w.field("max_interactions_per_history", r.counters.max_interactions);
  w.field("particle_balance_ok", r.counters.balanced());
  w.field("wall_time_s", r.wall_time_s);
  w.field("histories_per_s", r.wall_time_s > 0 ? static_cast<double>(r.histories) / r.wall_time_s : 0.0);
  w.end_object();

  w.key("provenance").begin_object();
  w.field("solver_version", b.configure.solver_version);
  w.field("git_commit", b.git.commit);
  w.field("git_dirty", b.git.dirty);
  w.field("build_type", b.configure.build_type);
  w.field("kokkos_version", b.kokkos_version);
  w.field("execution_space", b.default_execution_space);
  w.field("timestamp_utc", timestamp_utc);
  w.end_object();

  w.key("warnings").begin_array();
  for (const auto& s : r.warnings) w.value(s);
  w.end_array();
  w.end_object();
  return w.str();
}

std::string results_csv(const TpmcResult& r) {
  std::ostringstream os;
  os << "quantity,unit,value,std_error,ci_low,ci_high\n";
  for (const Row& row : rows(r.estimates))
    os << row.name << "," << row.unit << "," << num(row.e->value) << "," << num(row.e->std_error) << ","
       << num(row.e->ci_low) << "," << num(row.e->ci_high) << "\n";
  return os.str();
}

std::string surface_fields_csv(const TpmcResult& r, const Geometry& g) {
  if (!r.surface) throw std::logic_error("surface fields were not tallied");
  const SurfaceFields& f = *r.surface;
  std::ostringstream os;
  os << "triangle,source_index,material,centroid_x_m,centroid_y_m,centroid_z_m,normal_x,normal_y,normal_z,area_m2,"
        "force_x_N,force_y_N,force_z_N,pressure_Pa,shear_Pa,heat_rate_W,heat_flux_W_m2,heat_rate_positive_W,"
        "heat_rate_negative_W,impingement_rate_per_s\n";
  for (std::size_t t = 0; t < g.host.size(); ++t) {
    const Vec3 c = centroid(g.host.triangles[t]);
    const Vec3 n = g.host.normals[t];
    const double a = g.host.areas[t];
    // pressure: force component along -n (into the surface) per area; shear: tangential magnitude
    const double fn = -dot(f.force_N[t], n);
    const double ft = norm(f.force_N[t] + fn * n);
    os << t << "," << g.host.source_index[t] << "," << g.host.material[t] << "," << num(c.x) << "," << num(c.y)
       << "," << num(c.z) << "," << num(n.x) << "," << num(n.y) << "," << num(n.z) << "," << num(a) << ","
       << num(f.force_N[t].x) << "," << num(f.force_N[t].y) << "," << num(f.force_N[t].z) << "," << num(fn / a)
       << "," << num(ft / a) << "," << num(f.heat_rate_W[t]) << "," << num(f.heat_rate_W[t] / a) << ","
       << num(f.heat_rate_positive_W[t]) << "," << num(f.heat_rate_negative_W[t]) << ","
       << num(f.impingement_rate_per_s[t]) << "\n";
  }
  return os.str();
}

}  // namespace vleo
