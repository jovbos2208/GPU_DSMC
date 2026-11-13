# VLEO Particle Solver

GPU-accelerated 3D TPMC/DSMC solver for rarefied gas flows around triangulated
bodies (satellites in very low Earth orbit), with Maxwell and
Cercignani–Lampis–Lord gas-surface interaction. C++20 + Kokkos: one source tree
for CPU, NVIDIA (CUDA) and AMD (HIP).

The roadmap and checklist are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
**Status: version 0.9.0 – milestone M5 (release candidate for v1.0).** TPMC
(free-molecular) and DSMC (transitional regime) for gas mixtures (O, N, He, Ar,
N2, O2) with VHS/VSS collisions, Larsen–Borgnakke rotational and vibrational
energy, Maxwell and CLL gas-surface interaction including internal energy:
forces, moments, C_D, C_L, C_Y, C_M and heat transfer with standard errors and
confidence intervals, JSON/CSV output and per-triangle surface fields.
Verified against analytic theory, equilibrium collision theory, TPMC and
SPARTA (cases S1–S4). Automatic DSMC cell size from the local mean free path,
phase profiler, benchmarks and CI. Open for v1.0: CUDA builds on NVIDIA
hardware. Supported combinations: [docs/capabilities.md](docs/capabilities.md);
changes: [CHANGELOG.md](CHANGELOG.md).

## Quick start (CPU)

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug -j
ctest --preset cpu-debug

build/cpu-debug/vleo_solver --config configs/simulations/tpmc_cll.yaml \
                            --platform configs/platforms/cpu.yaml \
                            --output-dir results/tpmc_cll
# -> results/tpmc_cll/results.json (F, M, C_D, C_L, C_M, Q with uncertainties)
```

## Documentation

| Document | Content |
| -------- | ------- |
| [docs/user_guide.md](docs/user_guide.md) | workflow: geometry, solver choice, configuration, running, convergence checks, validation evidence |
| [docs/building.md](docs/building.md) | presets (CPU/CUDA/HIP), test command, verified toolchains |
| [docs/configuration.md](docs/configuration.md) | YAML schema, CLI, output files |
| [docs/conventions.md](docs/conventions.md) | SI units, axes, normals, sign conventions, RNG |
| [docs/geometry.md](docs/geometry.md) | STL import, mesh checks, epsilon strategy, BVH, inflow, transport |
| [docs/tpmc.md](docs/tpmc.md) | Maxwell/CLL kernels, event weight, tallies, statistics, output, physics verification |
| [docs/dsmc.md](docs/dsmc.md) | DSMC time loop, grid and cut cells, NTC/VHS, block averaging, diagnostics, convergence studies, mixtures and internal energy |
| [docs/species.md](docs/species.md) | species data with sources |
| [docs/capabilities.md](docs/capabilities.md) | capability matrix: supported combinations and their validation |
| [docs/dependencies.md](docs/dependencies.md) | pinned Kokkos / yaml-cpp / Catch2 versions |
| [docs/sparta.md](docs/sparta.md) | SPARTA comparison cases S1–S4 and results |
| [docs/development.md](docs/development.md) | checks, sanitizers, CI and GPU runners, fixed-seed examples, versioning, release checklist |
| [benchmarks/README.md](benchmarks/README.md) | benchmark cases, profiles, optimisations, before/after results |

## Layout

```
include/vleo/   public headers (core, geometry, gsi, io)
src/            libraries vleo_core, _geometry, _particles, _gsi, _tpmc, _dsmc, _tally, _io; main.cpp
configs/        platforms/, simulations/, materials/
tests/          unit/, regression/ (smoke), physics/, performance/, data/
third_party/    vendored kokkos, yaml-cpp, catch2 (see third_party/VERSIONS.md)
```
