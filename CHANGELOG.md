# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the
scheme in [docs/development.md](docs/development.md#versioning).

## [Unreleased]

### Added
- `dsmc.boundary: open | specular`: closed DSMC reservoir (specular walls, no
  inflow) as a run configuration (docs/dsmc.md §11).
- Profiling cases `benchmarks/profiling/` (closed air reservoir, GOCE 244 km,
  a collision-heavy dense GOCE variant) with `profile.sh` (built-in phases,
  `kp_trace` Kokkos timeline tool, rocprofv3, rocprof-sys, perf, plus a
  GPU/CPU load monitor) and `analyze_profile.py` (kernel and
  phase tables, GPU idle gaps, host wait time, timeline plots). The
  `Profiler::Scope` phases and the run stages are now Kokkos profiling regions
  when a Kokkos tool is loaded (no fences, no cost otherwise).
- DSMC cell-state output `dsmc.dt_DSMCState_out` (seconds of simulated time):
  per-cell averages over each output window – number and mass density,
  velocity, translational / rotational / vibrational / overall temperature,
  pressure, Mach number, local mean free path, cell size / λ, collision
  frequency, particles per cell, species densities – written as VTKHDF
  (HDF5, unstructured hexahedral grid of the gas cells) to
  `dsmc_state/dsmc_state_NNNNNN.vtkhdf` with a ParaView file series
  (`dsmc_state.vtkhdf.series`). `scripts/vtkhdf_to_vtu.py` converts to
  `.vtu` + `.pvd`. Optional HDF5 dependency (`VLEO_HDF5=AUTO|ON|OFF`);
  docs/dsmc.md §10. DSMC results are unchanged (sampling only reads the
  particles).
- `--batch <file>`: several cases in one process (Kokkos, self-test and geometries
  set up once; results identical to separate runs) and `--skip-existing` to
  resume. `results.json` is now written last and marks a finished case.
- `simulation.out` in every output directory: human-readable run log written
  while the run progresses (conditions, geometry, reference values, domain and
  inflow, numerical parameters incl. the DSMC particle weight W and the TPMC
  weight per history, progress with remaining time, results with SE/CI,
  diagnostics, status). DSMC progress is reported every 1 % (terminal: 5 %).
- MPI parallelisation (M6.1–M6.6, [docs/parallel.md](docs/parallel.md)):
  `VLEO_ENABLE_MPI`, presets `cpu-mpi`, `hip-gfx1101-mpi`, `hip-gfx1103-mpi`,
  one rank per GPU. TPMC splits the histories by id. DSMC decomposes the domain
  into z planes, migrates particles after the move, sorts cell lists by
  global id and rebalances the planes at the end of the warm-up. 1–4 ranks
  give identical counters and results equal to rounding (`mpi` tests,
  `scripts/check_ranks.py`). Only rank 0 writes output; `run_metadata.json`
  records ranks and devices; `results.json` records the decomposition.
- Cut-cell classification of the DSMC grid is shared among the ranks and
  runs one team per cut cell (set-up of B5 on one W7700: 1.12 s → 0.19 s;
  gas fractions bitwise unchanged).
- `--attitude <alpha> <beta>` (flight direction in the body frame, speed kept)
  and `scripts/attitude_sweep.py` (M6.8).
- GOCE and CHAMP geometries of March et al. 2019 (`examples/geometry/*_march2019.stl`,
  mm) with reference quantities, and `validation/satellites/run_sat.py`
  (comparison with SPARTA, free-molecular and DSMC) ([docs/satellites.md](docs/satellites.md)).
- `benchmarks/sparta_perf.py`: performance comparison with SPARTA (M6.12).
- Benchmark B5 (large DSMC case) and `benchmark.py run --ranks/--mpiexec/--scale`;
  scaling on 2× W7700 in benchmarks/README.md (M6.7).
- Profile: per-rank phases and loads, migration sub-phases, `run.setup`,
  `run.fill`, `run.steps`, `run.result`.

### Fixed
- DSMC species diagnostic `mean_particles` counted the slots of ended
  particles that were not yet compacted; it now counts active particles only.

### Changed
- DSMC cell sort on GPU backends: parallel rank sort (one thread per particle)
  instead of a serial insertion sort per cell, which dominated the time step
  on the Radeon PRO W7700 (59 % in B3). Results bitwise unchanged.
- Scripts are Python 3.6 compatible (server toolchain).
- DSMC collisions: one random stream per candidate pair and a fixed
  (σg)_max per step; GPU backends process the candidates of a cell in
  parallel waves of independent pairs (bitwise identical to the per-cell
  kernel on host backends). **DSMC results change bitwise** versus 0.9.0
  (statistically equal); fixed-seed example references regenerated.
- Test of the rotational relaxation rate: 600 particles per cell and 7 %
  tolerance (offset of −3.5 % ± 1.4 % found for the old and new kernel alike).
- Equipartition test: the standard-error (power) check scales with
  √VLEO_TEST_SCALE for the shortened Debug runs.

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
