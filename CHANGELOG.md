# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the
scheme in [docs/development.md](docs/development.md#versioning).

## [Unreleased]

## [0.9.0] – M5: performance, automation, release preparation

Release candidate for v1.0. Gate M5 is open only for CUDA (no NVIDIA hardware
available, see [docs/building.md](docs/building.md)).

### Added
- Phase profiler (`--profile`, `VLEO_PROFILE=1`): fenced timings per solver
  phase in `results.json → diagnostics.profile`.
- Benchmark suite `scripts/benchmark.py` (B1–B4, TPMC and DSMC) with
  efficiency metric SE² · wall time; baseline and A/B results in
  `benchmarks/` ([benchmarks/README.md](benchmarks/README.md)).
- Automatic DSMC cell size from the local mean free path of short pilot runs
  (`dsmc.cell_size_pilot`, default on for `cell_size_mode: automatic`).
- DSMC diagnostics: `particle_steps`, `particle_steps_per_s`,
  `min_local_mean_free_path_resolved_cells_m`, `cell_size_pilot`.
- CI (GitHub Actions): format check, CPU Release/Debug builds with all tests,
  AddressSanitizer + UBSan (`cpu-asan` preset), clang-tidy (informational);
  GPU workflow for self-hosted AMD/NVIDIA runners.
- Example runs with fixed seeds checked against stored references
  (`scripts/check_examples.py`, `tests/golden/`, CTest label `examples`).
- `NOTICE`, this changelog, release checklist, `.clang-format`, `.clang-tidy`.

### Changed
- TPMC on host backends: injection and transport fused into one kernel.
- BVH node boxes in single precision on GPU backends (outward-rounded and
  padded; hits are still computed in double precision and are identical).
- DSMC: lazy compaction of ended particles (only when > 5 % are free) and
  cell sort by storage index; fewer reductions in non-sampling steps.
- Results are bitwise unchanged by all optimisations (same seed and build).

## [0.4.0] – M4: gas mixtures, internal energy, validation

### Added
- Multi-species DSMC (O, N, He, Ar, N2, O2), VHS and VSS collisions.
- Larsen–Borgnakke rotational and quantum vibrational energy exchange;
  internal energy accommodation at the wall (CLL/Maxwell) and in the heat balance.
- SPARTA comparison (cases S1–S4, all within tolerance; `validation/sparta`).
- Capability matrix `docs/capabilities.md`.

## [0.3.0] – M3: DSMC (monatomic, elastic VHS)

### Added
- DSMC time loop with NTC collisions, cut cells, block-averaged statistics,
  resolution diagnostics; convergence studies (cell size, time step, particles).

## [0.2.0] – M2: TPMC solver

### Added
- Maxwell and CLL gas–surface interaction, forces, moments, coefficients and
  heat transfer with standard errors, JSON/CSV output, surface fields.

## [0.1.0] – M0/M1: project base, GPU geometry and transport

### Added
- CMake/Kokkos project with CPU, CUDA and HIP presets, YAML configuration,
  counter-based RNG, provenance metadata.
- STL import and mesh checks, BVH, watertight ray–triangle tests, inflow
  sampling and particle transport on CPU and GPU.
