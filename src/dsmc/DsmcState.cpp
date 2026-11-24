// Cell-state files of the DSMC solver (dsmc.dt_DSMCState_out): DsmcCellState
// -> VTKHDF unstructured grid of the gas cells (docs/dsmc.md section 10).

#include <cstdint>

#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/io/VtkHdf.hpp"

namespace vleo {

void write_dsmc_cell_state(const std::filesystem::path& path, const DsmcCellState& st) {
  VtkHexGrid g;
  g.origin = st.grid.origin;
  g.spacing = st.grid.cell_size;
  g.nx = st.grid.nx;
  g.ny = st.grid.ny;
  g.nz = st.grid.nz;
  for (std::size_t c = 0; c < st.gas_fraction.size(); ++c)
    if (st.gas_fraction[c] > 0.0) g.cells.push_back(static_cast<std::int64_t>(c));

  auto scalar = [&](const std::string& name, const std::vector<double>& v) {
    VtkCellField f{name, 1, {}};
    f.values.reserve(g.cells.size());
    for (const std::int64_t c : g.cells) f.values.push_back(v[static_cast<std::size_t>(c)]);
    g.cell_fields.push_back(std::move(f));
  };
  scalar("number_density_m3", st.number_density_m3);
  scalar("mass_density_kg_m3", st.mass_density_kg_m3);
  {
    VtkCellField f{"velocity_mps", 3, {}};
    f.values.reserve(3 * g.cells.size());
    for (const std::int64_t c : g.cells) {
      const Vec3& v = st.velocity_mps[static_cast<std::size_t>(c)];
      f.values.insert(f.values.end(), {v.x, v.y, v.z});
    }
    g.cell_fields.push_back(std::move(f));
  }
  scalar("temperature_K", st.temperature_K);
  scalar("temperature_trans_K", st.temperature_trans_K);
  scalar("temperature_rot_K", st.temperature_rot_K);
  scalar("temperature_vib_K", st.temperature_vib_K);
  scalar("pressure_Pa", st.pressure_Pa);
  scalar("mach", st.mach);
  scalar("mean_free_path_m", st.mean_free_path_m);
  scalar("cell_size_over_lambda", st.cell_size_over_lambda);
  scalar("collision_frequency_per_s", st.collision_frequency_per_s);
  scalar("particles_per_cell", st.particles);
  scalar("gas_fraction", st.gas_fraction);
  for (std::size_t q = 0; q < st.species.size() && st.species.size() > 1; ++q)
    scalar("number_density_" + st.species[q] + "_m3", st.species_number_density_m3[q]);
  {
    VtkCellField f{"cell_index", 1, {}};
    for (const std::int64_t c : g.cells) f.values.push_back(static_cast<double>(c));
    g.cell_fields.push_back(std::move(f));
  }

  g.field_data = {{"TimeValue", st.time_s},
                  {"AveragingWindow_s", st.window_s},
                  {"FirstStep", static_cast<double>(st.first_step)},
                  {"LastStep", static_cast<double>(st.last_step)},
                  {"SamplingPhase", st.sampling ? 1.0 : 0.0}};
  write_vtkhdf_hex_grid(path, g);
}

}  // namespace vleo
