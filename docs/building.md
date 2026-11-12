# Building and testing

One source tree, three kinds of binaries. The backend is fixed at compile time
by Kokkos; a configuration file cannot switch it.

| Preset        | Backend | Kokkos spaces    | Build type | Purpose |
| ------------- | ------- | ---------------- | ---------- | ------- |
| `cpu-debug`   | cpu     | Serial           | Debug, Kokkos debug + bounds checks, `-Werror` | development, numerical reference |
| `cpu-release` | cpu     | OpenMP + Serial  | Release    | CPU production / reference runs |
| `cuda`        | cuda    | Cuda + Serial    | Release    | NVIDIA GPUs (default `Kokkos_ARCH_AMPERE80`, A100) |
| `hip`         | hip     | HIP + Serial     | Release    | AMD GPUs (default `Kokkos_ARCH_AMD_GFX942`, MI300X) |
| `hip-gfx1100` | hip     | HIP + Serial     | Release    | AMD RDNA3 (`Kokkos_ARCH_AMD_GFX1100`): RX 7900, Radeon 7xxM iGPUs with `HSA_OVERRIDE_GFX_VERSION=11.0.0` |

Requirements: CMake ≥ 3.27, a C++20 compiler (GCC ≥ 10.4, Clang ≥ 14), for CUDA
NVCC ≥ 12.2, for HIP ROCm ≥ 6.2 (minimums of Kokkos 5.2.2, see
[dependencies.md](dependencies.md)). All libraries are vendored; no network
access is needed.

## Commands

```bash
# CPU (reference)
cmake --preset cpu-debug
cmake --build --preset cpu-debug -j
ctest --preset cpu-debug              # runs ALL tests (unit + smoke)

# NVIDIA
cmake --preset cuda                   # other GPU: -DKokkos_ARCH_AMPERE80=OFF -DKokkos_ARCH_HOPPER90=ON
cmake --build --preset cuda -j
ctest --preset cuda

# AMD (hip = MI300X; hip-gfx1100 = RDNA3)
cmake --preset hip                    # other GPU: -DKokkos_ARCH_AMD_GFX942=OFF -DKokkos_ARCH_AMD_GFX90A=ON
cmake --build --preset hip -j
ctest --preset hip
```

Build directories are `build/<preset>/`. Any CMake generator works; add
`-G Ninja` to the configure command if Ninja is installed. Without presets,
`ctest --test-dir build/<preset> --output-on-failure` is equivalent.

Useful subsets: `ctest --preset cpu-debug -L unit`, `-L smoke`, `-L rng`, `-L config`,
`-L geometry`, `-L particles`, `-L gate`.

## What the tests cover (M0, M1)

| Test | Content |
| ---- | ------- |
| `test_vec3` | Vec3/Mat3 algebra, rotations; host vs. device results |
| `test_frames` | local surface basis (t1, t2, n), aerodynamic axes; on host and device |
| `test_geometry` | triangle area/normal/centroid, degeneracy, bounding boxes; host vs. device |
| `test_rng` | Philox known-answer vectors, input mapping, determinism, host/device bitwise equality, execution-order independence, uniform/normal statistics, edge cases, exhaustion |
| `test_config` | shipped configs, defaults, derived quantities, material files, YAML round trip, ~100 negative cases |
| `test_json` | JSON writer |
| `test_build_info` | provenance fields, backend self-test |
| `test_stl`, `test_mesh` | STL import, mesh validation and orientation, device arrays |
| `test_ray` | ray-triangle analytic cases, watertightness, BVH ≡ brute force (host and device) |
| `test_inflow` | flux-weighted inflow distribution (KS, χ², moments), domain |
| `test_transport` | Gate M1: cube particle balance and analytic hit rate, multiple reflections, truncation, failure states |
| `smoke_*` | the `vleo_solver` executable: full start-up with the platform file of the build, metadata completeness (incl. geometry and inflow), re-parse of the resolved config, all shipped simulation configs, failure paths (foreign platform, invalid config, missing or open geometry, usage), `--version` |

Device tests run on the default execution space of the build (Serial/OpenMP on
CPU builds, the GPU on CUDA/HIP builds).

## Verified toolchains

| Date | Preset | Toolchain | Result |
| ---- | ------ | --------- | ------ |
| 2026-11-12 | `cpu-debug` | GCC 15.2.0, CMake 4.2.3, Linux x86_64 | build without warnings (`-Werror`), M0: 20/20, M1: 27/27 tests pass |
| 2026-11-12 | `cpu-release` | GCC 15.2.0, CMake 4.2.3, OpenMP, 12 threads | M0: 20/20, M1: 27/27 tests pass |
| – | `cuda` | not yet available | **open** – requires a machine with CUDA ≥ 12.2 |
| 2026-11-12 | `hip-gfx1100` | hipcc (HIP 7.1.52801, Clang 21.1.8), CMake 4.2.3, Radeon 740M iGPU (gfx1103 run as gfx1100 via `HSA_OVERRIDE_GFX_VERSION=11.0.0`) | build without warnings, M0: 20/20, M1: 27/27 tests pass; RNG self-test checksum bitwise identical to the CPU builds |
| – | `hip` (GFX942) | not yet available | **open** – compiles the same sources for MI300X; requires that hardware to run |

Add a row whenever a preset is built on a new toolchain. Gate M0 still
requires a `cuda` row.

## Notes for GPU builds

- The `cuda` preset compiles all code (including yaml-cpp and Catch2) through
  Kokkos' `nvcc_wrapper`; `CMAKE_CXX_COMPILER` is set by the preset. Set
  `NVCC_WRAPPER_DEFAULT_COMPILER` to choose the host compiler.
- Device code in tests is written as functor structs, not as `KOKKOS_LAMBDA`
  inside `TEST_CASE` bodies, to satisfy the CUDA extended-lambda rules.
- The HIP presets set `Kokkos_ENABLE_ROCTHRUST=OFF`: rocThrust is an optional
  Kokkos TPL (faster `Kokkos::sort`) that is often not installed with ROCm
  and is not needed by the solver. Re-enable it with `-DKokkos_ENABLE_ROCTHRUST=ON`.
- The binary only runs on the GPU architecture it was compiled for: choose the
  preset (or `Kokkos_ARCH_AMD_*`) that matches `rocminfo`.
- Run the GPU binary with the matching platform file (`configs/platforms/nvidia.yaml`
  or `amd.yaml`); a mismatch is rejected at start-up.
