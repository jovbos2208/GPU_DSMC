# Configuration files (M0.8, M0.9)

A run uses two YAML files:

- a **simulation configuration** (`configs/simulations/*.yaml`) with the physics
  and numerics - identical for every backend, and
- an optional **platform configuration** (`configs/platforms/*.yaml`) naming the
  backend and device. Its backend must match the binary.

```
vleo_solver --config configs/simulations/tpmc_cll.yaml \
            --platform configs/platforms/cpu.yaml [--output-dir DIR] [--check] [--print-config]
```

| Option | Meaning |
| ------ | ------- |
| `--check` | validate only (no Kokkos initialisation, no output); exit 0 if valid |
| `--print-config` | print the resolved configuration (YAML) to stdout |
| `--output-dir DIR` | overrides `output.directory` |
| `-n, --histories N` | overrides `tpmc.particle_histories` (recorded in `resolved_config.yaml`) |
| `--warmup-steps N`, `--sample-steps N` | override `dsmc.warmup_steps` / `dsmc.sample_steps` |
| `-q, --quiet` | no progress output |
| `--version` | build provenance (git commit, compiler, build type, Kokkos, backend) |
| `--batch FILE` | run several cases in one process (instead of `--config`): Kokkos, the backend self-test and each geometry (STL, BVH, upload) are set up once. One case per line, `<config.yaml> [<output dir>]`, `#` starts a comment, relative paths are relative to the batch file; without an output directory the case's `output.directory` is used. The overrides above apply to every case; `--output-dir` is not allowed. Results are identical to separate runs. A failing case does not stop the batch (exit code 1 at the end) |
| `--skip-existing` | skip cases whose output directory already contains `results.json` (written last, so it marks a finished case); resumes an interrupted batch |

Exit codes: 0 success, 1 runtime error, 2 invalid configuration, 64 usage error.

## Validation rules

Validation is strict and reports **all** problems at once, each with its key
path and, where available, the line number:

- Unknown keys are errors at every level (typos never pass silently); duplicate
  keys are errors.
- Required keys must be present; an explicit empty value (`key:`) is an error.
- Numbers must be unquoted and finite (`.nan`, `.inf` and `"1.0"` are rejected).
  Integers must be plain decimal digits (`2.0e6` is not an integer).
- Booleans are `true`/`false` only (not `yes`/`no`/`1`).
- Enumerations are case-insensitive and written back in canonical spelling.
- Relative paths are resolved against the directory of the file containing them.

## Simulation schema

R = required, O = optional (default in the last column).

### `simulation` (R)

| Key | Type | | Constraint / default |
| --- | ---- | - | -------------------- |
| `name` | string | R | letters, digits, `_ - .` |
| `mode` | enum | R | `tpmc` \| `dsmc` |
| `random_seed` | uint64 | R | 0 … 2⁶⁴−1; explicit for reproducibility |

### `geometry` (R)

| Key | Type | | Constraint / default |
| --- | ---- | - | -------------------- |
| `file` | path | R | binary or ASCII STL (see [geometry.md](geometry.md)) |
| `units` | enum | O | `m` \| `cm` \| `mm`; default `m` |
| `require_closed` | bool | O | default `true`: every component must be a closed, consistently oriented manifold; `false` accepts open components as two-sided surfaces |

### `domain` (O)

| Key | Type | Constraint / default |
| --- | ---- | -------------------- |
| `margin_m` | real | > 0; box = mesh bounding box grown by this margin. Default: 5 % of the largest mesh extent |
| `box_min_m`, `box_max_m` | [x, y, z] | explicit box (both required, min < max); must contain the body. Exclusive with `margin_m` |

### `reference` (R)

| Key | Type | | Constraint |
| --- | ---- | - | ---------- |
| `area_m2` | real | R | > 0 |
| `length_m` | real | R | > 0 |
| `moment_center_m` | [x, y, z] | R | finite |
| `lift_axis_body` | [x, y, z] | R | non-zero, not parallel to `gas.bulk_velocity_body_mps` |

### `gas` (R)

| Key | Type | | Constraint |
| --- | ---- | - | ---------- |
| `number_density_m3` | real | R | > 0 |
| `translational_temperature_K` | real | R | > 0 |
| `bulk_velocity_body_mps` | [x, y, z] | R | non-zero; gas velocity relative to the body |
| `species` | map name → fraction | R | non-empty; fractions ∈ [0, 1], sum = 1 ± 1e-6; species `O`, `N`, `He`, `Ar`, `N2`, `O2` ([species.md](species.md)); DSMC: up to 8 species |

### `surface` (R)

Either inline:

| Key | Type | | Constraint |
| --- | ---- | - | ---------- |
| `model` | enum | R | `maxwell` \| `cll` |
| `temperature_K` | real | R | > 0 |
| `maxwell.diffuse_fraction` | real | R if model = maxwell | [0, 1] |
| `cll.alpha_n` | real | R if model = cll | [0, 1], normal energy accommodation |
| `cll.sigma_t` | real | R if model = cll | [0, 2], tangential momentum accommodation |
| `cll.alpha_rot`, `cll.alpha_vib` | real | O | [0, 1], rotational / vibrational energy accommodation of molecules; default `alpha_n` |

The block of the non-selected model may be present; it is validated too.

or as a reference to a material file (then it must be the only key):

```yaml
surface:
  material_file: ../materials/example_surface.yaml   # contains a top-level `surface:` block
```

Material files cannot reference other material files.

### `tpmc` (O)

| Key | Type | Default | Constraint |
| --- | ---- | ------- | ---------- |
| `particle_histories` | uint64 | 1000000 | ≥ 1 |
| `max_reflections` | uint32 | 100 | ≥ 1; histories needing more surface interactions are truncated and counted |
| `batch_size` | uint64 | 1048576 | 1 … 2³¹; histories per kernel launch (memory vs. launch overhead) |
| `convergence.quantity` | enum | – | `C_D`, `C_L`, `C_Y`, `C_Mx`, `C_My`, `C_Mz`, `heat_rate_W` (block optional) |
| `convergence.relative_error` / `absolute_error` | real | – | > 0; exactly one; stop once SE ≤ target ([tpmc.md](tpmc.md) §5) |
| `convergence.min_histories` | uint64 | 0 | run at least this many histories |

### `dsmc` (O)

| Key | Type | Default | Constraint |
| --- | ---- | ------- | ---------- |
| `collision_model` | enum | `VHS` | `VHS` or `VSS` (VSS diameters and scattering exponents, [species.md](species.md)) |
| `timestep_mode` | enum | `automatic` | `automatic` \| `fixed` |
| `timestep_s` | real | – | > 0; required iff `timestep_mode: fixed`, forbidden otherwise |
| `cell_size_mode` | enum | `automatic` | `automatic` \| `fixed` |
| `cell_size_m` | real | – | > 0; required iff `cell_size_mode: fixed`, forbidden otherwise |
| `cell_size_pilot` | bool | `true` | only with `cell_size_mode: automatic`: refine the cell size from the local mean free path of short pilot runs ([dsmc.md](dsmc.md) §3.1); `false` keeps the free-stream rule |
| `particles_per_cell_target` | uint32 | 30 | ≥ 1; sets the particle weight W = n∞ V_cell / target |
| `warmup_steps` | uint64 | automatic | steps before sampling starts |
| `sample_steps` | uint64 | 1000 | ≥ 2; sampled steps |
| `blocks` | uint32 | 20 | 2 … 100000, ≤ `sample_steps`; blocks for the block-averaged SE |
| `max_particles` | uint64 | automatic | hard limit of the particle arrays |
| `dt_DSMCState_out` | real | – (off) | > 0 [s]; interval of simulated time between cell-state files: time-averaged density, velocity, temperatures, pressure, Mach number, … per grid cell as VTKHDF (HDF5) files for ParaView ([dsmc.md](dsmc.md) §10). Needs a build with HDF5 (`VLEO_HDF5`), otherwise the configuration is rejected |
| `state_sample_every` | integer | 1 | ≥ 1, only with `dt_DSMCState_out`: the cell-state moments are summed every n-th step of an output window (always including its first step). Consecutive steps are strongly correlated, so n of a few steps costs little statistics and saves most of the moments kernel (on GPUs up to ~30–45 % of the step time at n = 1) ([dsmc.md](dsmc.md) §10) |
| `reorder_interval` | integer | 20 | ≥ 0: every n steps the particle arrays are copied into cell order (0: never; W7700: 5–20 equally good, GOCE 244 km −34 % per step; RTX 4070 Ti SUPER: 20 is 2.4 % faster than 10), so the per-cell kernels (binning, collisions, cell moments) read contiguous memory. Cell lists are then sorted by particle id; the physics is bitwise unchanged, only the per-step force sums are added in a different order (round-off) ([dsmc.md](dsmc.md) §4) |
| `collision_kernel` | string | `automatic` | `automatic` (host backends: `per_cell`, GPUs: `hybrid`), `per_cell` (one thread per cell, candidates in sequence), `waves` (one team per cell, conflict-free waves of candidates) or `hybrid` (`per_cell` for cells with ≤ 256 candidates, `waves` for heavier cells); identical results, only the speed differs ([dsmc.md](dsmc.md) §4) |
| `collision_team_size` | integer | 32 | 8, 16, 32 or 64: threads per cell of the `waves` kernel (also the heavy cells of `hybrid`) on GPUs |
| `boundary` | string | `open` | `open`: free-stream inflow through all faces, outflow removes particles; `specular`: closed reservoir, specular walls, no inflow – the gas of the initial fill stays in the box (homogeneous gas, relaxation, profiling; [dsmc.md](dsmc.md) §11) |
| `faces` | map | – | only with `boundary: specular`: per-face override of the closed box, keys `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`. A face is `specular` (default), `periodic` (both faces of an axis) or a diffuse wall `{type: wall, temperature_K: T, velocity_mps: [u, v, w]}` (fully diffuse Maxwell at T, moving with the tangential velocity; the normal component must be 0). Force and heat rate of every wall face are reported in `results.json` → `diagnostics.walls` (Couette / Fourier verification, [dsmc.md](dsmc.md) §11.1, [verification.md](verification.md)) |

The automatic rules are in [dsmc.md](dsmc.md) §3; mixtures and internal
energy in §9; cell-state output in §10.

### `statistics` (O)

| Key | Type | Default | Constraint |
| --- | ---- | ------- | ---------- |
| `confidence_level` | real | 0.95 | (0, 1) |
| `report_uncertainty` | bool | true | |

### `output` (O)

| Key | Type | Default | Constraint |
| --- | ---- | ------- | ---------- |
| `directory` | string | `results/` | relative to the working directory |
| `format` | enum | `json` | `json` \| `csv` \| `hdf5`; `results.json` is always written, `csv` adds `results.csv`; `hdf5` is not available in this build and is rejected |
| `save_surface_fields` | bool | false | write per-triangle force/heat to `surface_fields.csv` |

## Platform schema

```yaml
platform:
  backend: cpu        # R: cpu | cuda | hip - must match the build
  device_id: 0        # O: GPU index, default 0 (ignored for cpu)
  num_threads: 0      # O: host threads (OpenMP), 0 = Kokkos default
  description: ...    # O: free text
```

## Outputs of a run

The TPMC result files (`results.json`, `results.csv`, `surface_fields.csv`)
are described in [tpmc.md](tpmc.md) §8 (DSMC: same schema, see
[dsmc.md](dsmc.md) §7).

### Common files

Geometry errors (unreadable STL, open or non-manifold mesh with
`require_closed: true`, domain box not containing the body) are reported like
configuration errors (exit code 2), also with `--check`.

`vleo_solver` writes into the output directory:

| File | Content |
| ---- | ------- |
| `run_metadata.json` | schema `vleo-run-metadata/1`: build provenance (git commit/dirty, compiler, build type, Kokkos version/backends/arch), runtime (UTC time, command line, execution space, concurrency, device), platform, resolved configuration, derived quantities (ρ∞, q∞, aerodynamic axes, species masses), `geometry` (mesh report: triangles, vertices, edges, defects, components with area/volume/orientation, materials, bounding box, BVH statistics, warnings), `inflow` (domain box, inflow rate per face and species, TPMC event-rate factor), notes incl. the backend self-test |
| `resolved_config.yaml` | the resolved configuration (all defaults filled in, absolute geometry path, material file inlined); valid input that reproduces the run |
| `input_config.yaml` | byte-exact copy of the input file |
| `kokkos_configuration.txt` | `Kokkos::print_configuration(verbose)` |
| `simulation.out` | human-readable run log, written while the run progresses (follow with `tail -f`): run and build, free-stream conditions (species, n, T, V, flight angles, speed ratio, q∞, λ∞ and Kn), gas-surface model, geometry (file, triangles, area, volume, bounding box), reference values and axes, domain and inflow per face, numerical parameters (DSMC: Δt, cell size, cells, **particle weight W = real molecules per simulator particle**, expected particle count, steps; TPMC: histories and the weight per history γ = R/N), progress (every 1 % for DSMC, at most every 1 % or 30 s for TPMC, with remaining time), results with standard errors and confidence intervals, diagnostics, warnings, status and output files. Rank 0 only. |
