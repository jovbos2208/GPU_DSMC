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
| `file` | path | R | STL file; existence is checked at start-up (warning in M0, error from M1) |
| `units` | enum | O | `m` \| `cm` \| `mm`; default `m` |

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
| `species` | map name → fraction | R | non-empty; fractions ∈ [0, 1], sum = 1 ± 1e-6; species `O`, `N`, `He`. `N2`/`O2` are recognised but rejected until M4 (internal energy). |

### `surface` (R)

Either inline:

| Key | Type | | Constraint |
| --- | ---- | - | ---------- |
| `model` | enum | R | `maxwell` \| `cll` |
| `temperature_K` | real | R | > 0 |
| `maxwell.diffuse_fraction` | real | R if model = maxwell | [0, 1] |
| `cll.alpha_n` | real | R if model = cll | [0, 1], normal energy accommodation |
| `cll.sigma_t` | real | R if model = cll | [0, 2], tangential momentum accommodation |

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
| `max_reflections` | uint32 | 100 | ≥ 1 |

### `dsmc` (O)

| Key | Type | Default | Constraint |
| --- | ---- | ------- | ---------- |
| `collision_model` | enum | `VHS` | `VHS` (`VSS` planned) |
| `timestep_mode` | enum | `automatic` | `automatic` \| `fixed` |
| `timestep_s` | real | – | > 0; required iff `timestep_mode: fixed`, forbidden otherwise |
| `cell_size_mode` | enum | `automatic` | `automatic` \| `fixed` |
| `cell_size_m` | real | – | > 0; required iff `cell_size_mode: fixed`, forbidden otherwise |
| `particles_per_cell_target` | uint32 | 30 | ≥ 1 |

`automatic` is a planned solver function (M3); M0 only validates the input.

### `statistics` (O)

| Key | Type | Default | Constraint |
| --- | ---- | ------- | ---------- |
| `confidence_level` | real | 0.95 | (0, 1) |
| `report_uncertainty` | bool | true | |

### `output` (O)

| Key | Type | Default | Constraint |
| --- | ---- | ------- | ---------- |
| `directory` | string | `results/` | relative to the working directory |
| `format` | enum | `json` | `json` \| `csv` \| `hdf5` |
| `save_surface_fields` | bool | false | |

## Platform schema

```yaml
platform:
  backend: cpu        # R: cpu | cuda | hip - must match the build
  device_id: 0        # O: GPU index, default 0 (ignored for cpu)
  num_threads: 0      # O: host threads (OpenMP), 0 = Kokkos default
  description: ...    # O: free text
```

## Outputs of a run (M0)

`vleo_solver` writes into the output directory:

| File | Content |
| ---- | ------- |
| `run_metadata.json` | schema `vleo-run-metadata/1`: build provenance (git commit/dirty, compiler, build type, Kokkos version/backends/arch), runtime (UTC time, command line, execution space, concurrency, device), platform, resolved configuration, derived quantities (ρ∞, q∞, aerodynamic axes, species masses), notes incl. the backend self-test |
| `resolved_config.yaml` | the resolved configuration (all defaults filled in, absolute geometry path, material file inlined); valid input that reproduces the run |
| `input_config.yaml` | byte-exact copy of the input file |
| `kokkos_configuration.txt` | `Kokkos::print_configuration(verbose)` |
