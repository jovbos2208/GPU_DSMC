# User guide

Workflow for computing aerodynamic coefficients and heat loads of a body in
rarefied flow. Reference documentation: configuration schema
[configuration.md](configuration.md), methods [tpmc.md](tpmc.md) and
[dsmc.md](dsmc.md), conventions [conventions.md](conventions.md).

## 1. Build

```bash
cmake --preset cpu-release && cmake --build --preset cpu-release -j      # CPU (OpenMP)
cmake --preset hip-gfx1101 && cmake --build --preset hip-gfx1101 -j      # AMD GPU (choose your arch)
ctest --preset <preset>                                                  # all tests must pass
```

Presets, GPU architectures and verified toolchains: [building.md](building.md).

## 2. Prepare the geometry

- Triangulated STL (ASCII or binary), in metres unless `geometry.units`
  says otherwise, in the **body frame** (axes and sign conventions:
  [conventions.md](conventions.md)).
- Closed, consistently oriented solids (outward normals); several solids
  become separate components/materials. Open surfaces (plates) need
  `geometry.require_closed: false` and are treated as two-sided.
- `vleo_solver --check -c case.yaml` reads and checks the mesh (defects,
  orientation, component volumes) without running; see
  [geometry.md](geometry.md).

## 3. Choose the solver

| Regime | Solver | Typical VLEO altitude |
| ------ | ------ | --------------------- |
| Kn ≫ 1 (λ∞ much larger than the body) | `tpmc` – no gas–gas collisions, fast, exact free-molecular limit | above ≈ 150 km |
| Kn ≈ 0.01 … 10 | `dsmc` – collisions (VHS/VSS, Larsen–Borgnakke) | below ≈ 150 km, dense parts of the orbit |

λ∞ is printed by the DSMC run and written to `results.json`; if DSMC with
`n → 0` and TPMC agree (dsmc.md §8), TPMC is sufficient.

## 4. Write the configuration

Start from `configs/simulations/` (`tpmc_maxwell`, `tpmc_cll`, `tpmc_air`,
`dsmc_maxwell`, `dsmc_cll`). Required blocks: `simulation` (mode, seed),
`geometry`, `reference` (area, length, moment centre, lift axis), `gas`
(number density, temperature, velocity in the body frame, mole fractions),
`surface` (Maxwell or CLL, wall temperature, per-material overrides).

`vleo_solver --print-config -c case.yaml` shows every default that will be
used; the same file is written as `resolved_config.yaml` and reproduces the
run.

## 5. Run

```bash
vleo_solver -c case.yaml -p configs/platforms/amd.yaml -o results/case
vleo_solver -c case.yaml --histories 2e7 -o results/case_long   # TPMC: more histories
```

The platform file selects device and threads and is checked against the
build backend. Results: `results.json` (F, M, C_D, C_L, C_Y, C_M, Q̇ with
standard errors and confidence intervals), `results.csv`, optional
`surface_fields.csv`, `run_metadata.json` (provenance).

## 6. Make the result trustworthy

**TPMC:** the standard error falls as 1/√N; run until the reported
confidence interval is small enough (`tpmc.convergence` can stop
automatically). Nothing else to converge.

**DSMC** has numerical parameters; the automatic choices (dsmc.md §3) are
starting points:

1. Keep `cell_size_mode: automatic` with the default pilot runs
   (`cell_size_pilot: true`): the cell size then follows the smallest local
   mean free path (dsmc.md §3.1). Check `diagnostics.cell_size_pilot` and
   `fraction_cells_dx_gt_local_lambda` (should be a few % at most).
2. Check `warnings` in `results.json` and on stderr: CFL > 1, few particles
   per cell, too short blocks for the standard error, small cut cells.
3. For final numbers: repeat with 0.7× the cell size and half the time step;
   the change of C_D and Q̇ should be below the accuracy you need. The
   recorded study (dsmc.md §8) shows that the cell size dominates, the time
   step and particle number hardly matter once the cells resolve λ_loc.
4. The DSMC standard error comes from block averaging of correlated steps;
   `blocks_long_enough: false` means: more `sample_steps`.

## 7. Performance

- Use a dedicated GPU for large cases. Measured so far (benchmarks/README.md):
  a laptop iGPU (Radeon 740M) is *slower* than the 12-thread laptop CPU
  (shared memory bandwidth); dedicated-GPU numbers (W7700, NVIDIA) are still
  to be recorded.
- `--profile` adds per-phase timings to `diagnostics.profile`;
  `scripts/benchmark.py` measures the standard cases
  ([benchmarks/README.md](../benchmarks/README.md)).
- DSMC cost scales with the number of cells × particles per cell; the pilot
  rule can increase the cell count substantially for dense flows around thin
  parts – the pilot report in the log shows the chosen cell size before the
  main run starts.

## 8. Validation evidence

| What | Where |
| ---- | ----- |
| Supported combinations and the tests that validate them | [capabilities.md](capabilities.md) |
| Free-molecular theory, GSI kernels, statistics | [tpmc.md](tpmc.md) §6–7 |
| DSMC collision rates, relaxation, conservation, DSMC ≡ TPMC limit, convergence studies | [dsmc.md](dsmc.md) §6–8 |
| Mixtures and internal energy | [dsmc.md](dsmc.md) §9 |
| Comparison with SPARTA (S1–S4) | [sparta.md](sparta.md), `validation/sparta/` |
| Cross-backend agreement | [building.md](building.md), [tpmc.md](tpmc.md) §9 |
| Reproducibility of the shipped examples | `tests/golden/`, [development.md](development.md) §3 |
