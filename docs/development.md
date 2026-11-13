# Development, CI and releases

## 1. Checks before a commit

```bash
python3 scripts/check_format.py                  # formatting rules (also run by CI)
cmake --preset cpu-debug && cmake --build --preset cpu-debug -j && ctest --preset cpu-debug
cmake --preset cpu-release && cmake --build --preset cpu-release -j && ctest --preset cpu-release
```

Physics changes additionally need a documented reference case
(IMPLEMENTATION_PLAN.md, section 11) and, for performance work, a before/after
benchmark ([benchmarks/README.md](../benchmarks/README.md)).

### Formatting and static analysis

- `scripts/check_format.py` enforces: UTF-8, LF, final newline, no tabs, no
  trailing whitespace, C++ lines ≤ 150 characters. The intended layout
  (Google style, 120 columns) is in `.clang-format` for editors; the code base
  is not mass-reformatted by it.
- `.clang-tidy` holds the static-analysis checks (bugprone, clang-analyzer,
  performance). In CI the clang-tidy job is informational
  (`continue-on-error`) until the findings have been triaged; compiler
  warnings (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion`) are errors in
  the `cpu-debug` and `cpu-asan` presets.

### Sanitizers (M5.8)

```bash
cmake --preset cpu-asan && cmake --build --preset cpu-asan -j && ctest --preset cpu-asan
```

`cpu-asan` is a Debug build (Kokkos Serial) with AddressSanitizer, leak
detection and UndefinedBehaviorSanitizer (`-fno-sanitize-recover`: any UB
report fails the test). The whole build, including Kokkos and yaml-cpp, is
instrumented. ThreadSanitizer is not used: Kokkos OpenMP produces false
positives without an instrumented OpenMP runtime.

## 2. Continuous integration

| Workflow | Trigger | Jobs |
| -------- | ------- | ---- |
| `.github/workflows/ci.yml` | push to `main`, pull requests | `format`; `cpu-release` and `cpu-debug` (build, all CTest tests incl. examples with fixed seeds); `sanitizers` (`cpu-asan`); `static-analysis` (clang-tidy, informational) |
| `.github/workflows/gpu.yml` | weekly (Mon 02:00 UTC), manual | `amd`: HIP build + all tests + examples + benchmark on a self-hosted runner; `nvidia`: CUDA (manual, once a runner exists) |

### GPU runners (M5.7)

GPU jobs need self-hosted GitHub Actions runners on machines with the GPU
toolchain:

1. On the GPU machine: *Repository → Settings → Actions → Runners → New
   self-hosted runner*, follow the instructions, and add the label `gpu-amd`
   (or `gpu-nvidia`) when `config.sh` asks for labels. Run it as a service
   (`./svc.sh install && ./svc.sh start`) under an unprivileged account that
   can access `/dev/kfd` and `/dev/dri` (group `render`/`video`).
2. Environment modules: put the commands that make the toolchain available
   into `~/.vleo-gpu-env` of the runner account; the workflow sources it,
   e.g. for the W7700 server:

   ```bash
   module use /software/stack/gpu/amd/2025_07/share/spack/lmod/linux-opensuse15-x86_64/Core
   module load llvm-amdgpu/6.3.1 hip rocminfo cmake/3.31.5
   ```

3. Set the repository variable `VLEO_HIP_PRESET` (*Settings → Secrets and
   variables → Actions → Variables*) if the runner's GPU is not gfx1101.

Self-hosted runners must not run workflows from forks of a public repository
(the repository is private; keep it that way or restrict the runner group).

## 3. Examples with fixed seeds (M5.10)

`scripts/check_examples.py <vleo_solver>` runs every configuration in
`configs/simulations/` with its seed and a shortened run length and compares
all main quantities with `tests/golden/<name>.json`:

- same build (compiler, flags, Kokkos version/arch, execution space, thread
  count) as the reference: agreement to 1e-9 relative (bitwise reproducible);
- otherwise statistical agreement, z ≤ 5.

It is registered as the CTest test `examples_fixed_seeds` (label `examples`,
Release builds only). After an intended physics change, regenerate the
references with `--update`, review the diff and mention it in the changelog.

## 4. Versioning

Semantic versioning `MAJOR.MINOR.PATCH`, defined once in the top-level
`CMakeLists.txt` (`project(... VERSION ...)`) and reported by
`vleo_solver --version` and in every `run_metadata.json`:

- **MAJOR**: incompatible change of the configuration schema or of the result
  files (`schema` fields `vleo-results/N`, `vleo-run-metadata/N` change too).
- **MINOR**: new physics or features, compatible input/output.
- **PATCH**: bug fixes without intended change of results beyond the
  statistical error.

Pre-1.0 minor versions follow the milestones (0.1 = M0/M1 … 0.9 = M5 release
candidate). 1.0.0 is tagged when Gate M5 is fully satisfied (including CUDA).
Tags are `vMAJOR.MINOR.PATCH`, annotated, on `main`.

## 5. Release checklist

1. All Gate items of the milestone are ticked in `IMPLEMENTATION_PLAN.md`
   (or explicitly marked open with a reason).
2. `CHANGELOG.md`: move *Unreleased* to the new version with the date;
   bump the version in `CMakeLists.txt`.
3. From a **clean checkout** of the release commit
   (`git clone … && git checkout <commit>`; `git status` empty):
   - `python3 scripts/check_format.py`
   - `cpu-debug`, `cpu-release`, `cpu-asan`: configure, build, `ctest` – all pass
   - each available GPU preset on real hardware: `ctest` – all pass; record
     the platform in `docs/building.md` ("Verified toolchains")
   - `scripts/check_examples.py` on every tested build
4. Validation: re-run `validation/sparta/run_cases.py` if physics changed;
   archived reports in `validation/` must match the release.
5. Performance: `scripts/benchmark.py run` on the reference GPU; commit the
   JSON under `benchmarks/results/` and compare with the previous release.
6. `vleo_solver --version` shows the new version and a clean (`dirty: false`)
   git state.
7. Tag: `git tag -a vX.Y.Z -m "…" && git push origin vX.Y.Z`; create the
   GitHub release with the changelog section. Source archives contain the
   vendored dependencies (`third_party/`, licenses in `NOTICE`).
