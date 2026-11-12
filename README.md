# VLEO Particle Solver

GPU-accelerated 3D TPMC/DSMC solver for rarefied gas flows around triangulated
bodies (satellites in very low Earth orbit), with Maxwell and
Cercignani–Lampis–Lord gas-surface interaction. C++20 + Kokkos: one source tree
for CPU, NVIDIA (CUDA) and AMD (HIP).

The roadmap and checklist are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
**Status: milestone M0** (project base and CPU reference) – configuration,
provenance, core math, RNG and tests. No particle simulation yet.

## Quick start (CPU)

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug -j
ctest --preset cpu-debug

build/cpu-debug/vleo_solver --config configs/simulations/tpmc_cll.yaml \
                            --platform configs/platforms/cpu.yaml \
                            --output-dir results/tpmc_cll
```

## Documentation

| Document | Content |
| -------- | ------- |
| [docs/building.md](docs/building.md) | presets (CPU/CUDA/HIP), test command, verified toolchains |
| [docs/configuration.md](docs/configuration.md) | YAML schema, CLI, output files |
| [docs/conventions.md](docs/conventions.md) | SI units, axes, normals, sign conventions, RNG |
| [docs/dependencies.md](docs/dependencies.md) | pinned Kokkos / yaml-cpp / Catch2 versions |

## Layout

```
include/vleo/   public headers (core, geometry, gsi, io)
src/            libraries vleo_core, _geometry, _particles, _gsi, _tpmc, _dsmc, _tally, _io; main.cpp
configs/        platforms/, simulations/, materials/
tests/          unit/, regression/ (smoke), physics/, performance/, data/
third_party/    vendored kokkos, yaml-cpp, catch2 (see third_party/VERSIONS.md)
```
