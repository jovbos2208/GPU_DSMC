#include "vleo/dsmc/DsmcResults.hpp"

#include "vleo/io/JsonWriter.hpp"
#include "vleo/tpmc/Results.hpp"

namespace vleo {

std::string dsmc_results_json(const DsmcResult& r, const SimulationConfig& cfg, const std::string& timestamp_utc) {
  JsonWriter w;
  w.begin_object();
  w.field("schema", kResultsSchema);
  write_simulation_and_reference(w, cfg);
  write_estimate_blocks(w, r.estimates);

  const SeriesStatistics& drag = r.series[kQPDrag];
  const SeriesStatistics& heat = r.series[kQE];
  w.key("statistics").begin_object();
  w.field("confidence_level", cfg.statistics.confidence_level);
  w.field("critical_value", r.critical_value);
  w.field("method", "block averaging over the sampling phase: std_error = sd(block means) / sqrt(blocks)");
  w.field("warmup_steps", r.settings.warmup_steps);
  w.field("sample_steps", r.settings.sample_steps);
  w.field("blocks", static_cast<std::uint64_t>(drag.blocks));
  w.field("block_length_steps", static_cast<std::uint64_t>(drag.block_length));
  w.field("tau_int_steps_drag", drag.tau_int);
  w.field("tau_int_steps_heat", heat.tau_int);
  w.field("std_error_naive_C_D", r.estimates.C_D.std_error > 0 && drag.std_error_blocks > 0
                                     ? r.estimates.C_D.std_error * drag.std_error_naive / drag.std_error_blocks
                                     : 0.0);
  w.field("blocks_long_enough", drag.blocks_long_enough && heat.blocks_long_enough);
  w.field("converged", false);
  w.end_object();

  const DsmcSettings& s = r.settings;
  const DsmcDiagnostics& d = r.diagnostics;
  const DsmcCounters& c = r.counters;
  w.key("diagnostics").begin_object();
  w.field("timestep_s", s.timestep_s);
  w.field("timestep_automatic", s.timestep_automatic);
  w.field("cell_size_m", Vec3(d.cell_dx_m, d.cell_dy_m, d.cell_dz_m));
  w.field("cell_size_automatic", s.cell_size_automatic);
  w.field("particle_weight", s.particle_weight);
  w.field("mean_free_path_inf_m", d.lambda_inf_m);
  w.field("collision_time_inf_s", d.tau_inf_s);
  w.field("cell_over_lambda_inf", d.cell_over_lambda_inf);
  w.field("dt_over_tau_inf", d.dt_over_tau_inf);
  w.field("max_cfl", d.max_cfl);
  w.field("min_local_mean_free_path_m", d.min_local_lambda_m);
  w.field("fraction_cells_dx_gt_local_lambda", d.fraction_cells_dx_gt_lambda);
  w.field("mean_particles_per_gas_cell", d.mean_particles_per_gas_cell);
  w.field("fraction_cells_below_5_particles", d.fraction_cells_below_5);
  w.field("particles_mean", d.particles_mean);
  w.field("particles_min", d.particles_min);
  w.field("particles_max", d.particles_max);
  w.key("grid").begin_object();
  w.field("cells", r.grid.cells);
  w.field("gas_cells", r.grid.gas_cells);
  w.field("solid_cells", r.grid.solid_cells);
  w.field("cut_cells", r.grid.cut_cells);
  w.field("small_cut_cells", r.grid.small_cells);
  w.field("min_positive_gas_fraction", r.grid.min_positive_fraction);
  w.field("gas_volume_m3", r.grid.gas_volume_m3);
  w.field("domain_volume_m3", r.grid.domain_volume_m3);
  w.end_object();
  w.key("collisions").begin_object();
  w.field("candidates", c.collision_candidates);
  w.field("accepted", c.collisions);
  w.field("rejected", c.collision_candidates - c.collisions);
  w.field("sigma_g_max_updates", c.sigma_g_max_updates);
  w.field("acceptance_ratio_sampling", d.acceptance_ratio);
  w.field("collisions_per_particle_per_step", d.collisions_per_particle_per_step);
  w.field("free_stream_collisions_per_particle_per_step", d.free_stream_collisions_per_particle_per_step);
  w.end_object();
  w.field("steps", c.steps);
  w.field("injected", c.injected);
  w.field("exited", c.exited);
  w.field("truncated", c.truncated);
  w.field("lost_inside_body", c.lost_inside_body);
  w.field("invalid", c.invalid);
  w.field("removed_inside_at_fill", c.removed_inside_at_fill);
  w.field("surface_interactions", c.surface_interactions);
  w.field("particle_balance_ok", c.lost_inside_body + c.invalid == 0);
  w.field("wall_time_s", r.wall_time_s);
  w.end_object();

  write_provenance(w, timestamp_utc);
  w.key("warnings").begin_array();
  for (const auto& m : r.warnings) w.value(m);
  w.end_array();
  w.end_object();
  return w.str();
}

}  // namespace vleo
