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

The automatic rules are in [dsmc.md](dsmc.md) §3; mixtures and internal
energy in §9.

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
