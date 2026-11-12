# Dependencies (M0.3)

All dependencies are pinned and vendored in `third_party/` so every build of a
given solver commit uses exactly the same library code. Nothing is downloaded
at configure time.

| Dependency | Version          | Purpose                                   | Linked into        |
| ---------- | ---------------- | ----------------------------------------- | ------------------ |
| Kokkos     | 5.2.2            | Performance portability: Serial/OpenMP, CUDA, HIP | all targets |
| yaml-cpp   | 0.9.0            | Parsing configuration files               | `vleo_io` (private) |
| Catch2     | 3.8.1            | Unit/physics test framework               | tests only         |

Exact upstream commits and licenses: [`third_party/VERSIONS.md`](../third_party/VERSIONS.md).

## Why these versions

- **Kokkos 5.2.2**: Kokkos 5 requires C++20, matching the project standard.
  It supports CUDA (NVCC >= 12.2) and HIP (ROCm/hipcc >= 6.2) from the same
  source. 5.2.2 is the newest patch release of the 5.2 series; moving to a
  newer minor release is a deliberate dependency update (see below).
- **yaml-cpp 0.9.0**: current release; its CMake files work with CMake 4.x
  (`cmake_minimum_required(VERSION 3.5...3.30)`); used only as a private
  dependency of `vleo_io`, so no yaml-cpp type appears in public headers.
- **Catch2 3.8.1**: mature v3 release; the tests use a custom `main` that
  initialises Kokkos before running the session.

## Toolchain requirements imposed by the dependencies

| Component | Minimum (Kokkos 5.2.2) | Verified in this repository |
| --------- | ---------------------- | --------------------------- |
| CMake     | 3.27 (project), Kokkos needs 3.22 | 4.2.3 |
| GCC       | 10.4                   | 15.2.0 (Ubuntu), cpu-debug and cpu-release |
| Clang (CPU) | 14.0                 | not yet verified |
| NVCC (CUDA) | 12.2                 | **not yet verified** - no NVIDIA GPU/toolchain available |
| hipcc (ROCm) | 6.2                 | HIP 7.1.52801 / Clang 21.1.8, `hip-gfx1100` preset |

See [`building.md`](building.md) for the build commands and the status of the
GPU builds.

## Updating a dependency

1. Download the new release tag into an empty directory, e.g.
   `git clone --depth 1 --branch <tag> <url> /tmp/dep && rm -rf /tmp/dep/.git`.
2. Replace `third_party/<dep>/` with it (no local modifications).
3. Update `third_party/VERSIONS.md` and this file (version, commit, date, rationale).
4. Rebuild and run the complete test suite on CPU, CUDA and HIP (`ctest --preset <preset>`).
5. Commit the dependency update separately from any solver change.
