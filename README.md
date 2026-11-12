# VLEO Particle Solver

GPU-accelerated 3D TPMC/DSMC solver for rarefied gas flows around triangulated
bodies (satellites in very low Earth orbit), with Maxwell and
Cercignani–Lampis–Lord gas-surface interaction. C++20 + Kokkos: one source tree
for CPU, NVIDIA (CUDA) and AMD (HIP).

The roadmap and checklist are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
**Status: milestone M3 – release v0.2-dsmc.** TPMC (free-molecular) and DSMC
(transitional, elastic VHS collisions, NTC, cut cells) with Maxwell and CLL
gas-surface interaction: forces, moments, C_D, C_L, C_Y, C_M and heat transfer
with standard errors and confidence intervals, JSON/CSV output and
per-triangle surface fields. Verified against analytic free-molecular theory,
equilibrium collision theory and TPMC. DSMC currently supports one monatomic
species (He, Ar); mixtures and internal energy follow in M4.

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
| [docs/building.md](docs/building.md) | presets (CPU/CUDA/HIP), test command, verified toolchains |
| [docs/configuration.md](docs/configuration.md) | YAML schema, CLI, output files |
| [docs/conventions.md](docs/conventions.md) | SI units, axes, normals, sign conventions, RNG |
| [docs/geometry.md](docs/geometry.md) | STL import, mesh checks, epsilon strategy, BVH, inflow, transport |
| [docs/tpmc.md](docs/tpmc.md) | Maxwell/CLL kernels, event weight, tallies, statistics, output, physics verification |
| [docs/dsmc.md](docs/dsmc.md) | DSMC time loop, grid and cut cells, NTC/VHS, block averaging, diagnostics, convergence studies |
| [docs/dependencies.md](docs/dependencies.md) | pinned Kokkos / yaml-cpp / Catch2 versions |

## Layout

```
include/vleo/   public headers (core, geometry, gsi, io)
src/            libraries vleo_core, _geometry, _particles, _gsi, _tpmc, _dsmc, _tally, _io; main.cpp
configs/        platforms/, simulations/, materials/
tests/          unit/, regression/ (smoke), physics/, performance/, data/
third_party/    vendored kokkos, yaml-cpp, catch2 (see third_party/VERSIONS.md)
```
