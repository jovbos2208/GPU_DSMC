# Profiling cases (GPU and CPU timeline)

Two DSMC cases and a driver that show **which kernel / phase runs when** on the
GPU and on the host CPU, how long the host waits for the GPU, and where the
GPU is idle.

| Case | Content | Steps | Run time (W7700) |
| ---- | ------- | ----- | ---------------- |
| `p3_reservoir_air.yaml` (default) | closed reservoir (`dsmc.boundary: specular`), 1 m³ of N2/O2 at rest, 1000 K, λ = 2 cells, 0.1 collisions per particle and step: uniform, collision-dominated load, no inflow/outflow, no body (2 mm marker cube) | 200 + 800, cell state every 200 steps | ~1.5 min (est.) |
| `p1_goce_244km.yaml` | GOCE full scale, 244 km (NRLMSIS 2.1), CLL; production case, nearly free-molecular: move/BVH dominates, collisions idle | 400 + 600, cell state every 200 steps | ~1.5 min |
| `p2_goce_dense.yaml` | same geometry and numerics at 10⁴ × the density (λ∞ = 0.13 m): collision kernels carry load | 800 + 400, cell state every 200 steps | ~2 min |

P3: 67³ = 0.30 M cells × 50 particles = 15 M particles; physics checks in
`results.json`: constant particle count, collisions per particle and step =
Δt/τ∞, and in the cell states T_tr = T_rot = 1000 K, gas at rest (CPU test
with 8000 cells: 0.1004 vs 0.1005, particle count exact).
P1/P2 use 0.08 m cells (0.78 M cells), 20 particles per cell (≈ 15–17 M
particles). All write 5–6 VTKHDF cell-state files, so the timeline contains
every kind of work: injection, move (BVH ray tracing + CLL), the step reduction
(the host waits here), binning (histogram, scan, fill, sort), collisions, cell
moments, the cell-state copy to the host plus the HDF5 write, and the progress
report.

## Running (hagen)

```bash
module load llvm-amdgpu/6.3.1 hip rocminfo hdf5/1.14.4-3   # + rocprofiler-sdk / rocprof-sys if separate modules
cd ~/workspace/GPU_DSMC/benchmarks/profiling
DEVICE=1 ./profile.sh phases                     # fenced phase times only (no tool needed)
DEVICE=1 ./profile.sh kokkos                     # timeline with kp_trace (no vendor tool needed)
DEVICE=1 ./profile.sh rocprof                    # rocprofv3 timeline, if installed
DEVICE=1 ./profile.sh all p1_goce_244km.yaml     # every available tool, GOCE case
```

`DEVICE` = HIP index of the free GPU, `BIN` = solver binary (default
`build/hip-gfx1101-hdf5/vleo_solver`). Results go to `results/<case>/<mode>/`;
`analyze_profile.py` runs at the end and writes `results/<case>/summary.md`
and the figures.

| Mode | Tool | Shows |
| ---- | ---- | ----- |
| `phases` | built-in `VLEO_PROFILE=1` | fenced wall time per phase (`dsmc.move`, `dsmc.bin.*`, `dsmc.collide`, …); exact per phase, but the fences serialise host and GPU |
| `kokkos` | `kp_trace.cpp` (Kokkos Tools library, compiled by the script with g++) | timeline of regions, every kernel by label, deep copies and fences → `kokkos/trace.csv` + `trace.json` (Perfetto); with `KP_TRACE_FENCE=1` (default) each kernel is fenced, so its interval is its device time and the gaps are host-only time; `KP_TRACE_FENCE=0` shows the asynchronous launches instead |
| `rocprof` | `rocprofv3` | unfenced timeline: GPU kernels and copies, HIP API calls of the host thread, Kokkos regions and kernel labels (`--kokkos-trace`); `.pftrace` for Perfetto + CSV |
| `sys` | `rocprof-sys-sample` (omnitrace) | host call-stack sampling (where the CPU spends its time, incl. HIP runtime threads) together with GPU activity and Kokkos regions in one Perfetto trace |
| `perf` | Linux `perf` | host hot spots and per-thread time (needs `kernel.perf_event_paranoid ≤ 2`) |

Every mode also writes `monitor.csv` (every 0.25 s: GPU busy %, VRAM and
power of all AMD GPUs from sysfs; CPU % = cores in use, RSS and thread count of
the solver), plotted as `monitor_<mode>.png`.

## Collision kernel comparison

```bash
DEVICE=1 ./compare_collision_kernels.sh                                   # P3 reservoir
DEVICE=1 ./compare_collision_kernels.sh p3_reservoir_air.yaml p2_goce_dense.yaml
```

Runs each case shortened (100 + 200 steps, no cell-state output, fenced phase
profiler) with `per_cell` and `waves` at 8/16/32/64 threads per cell and prints
`dsmc.collide` per step; the collision counters must agree for all variants.
Table in `results/collide/summary.txt`.

## Option variants

```bash
DEVICE=1 VARIANTS="reorder_interval=0 reorder_interval=5 reorder_interval=5,state_sample_every=5" \
    ./compare_variants.sh p3_reservoir_air.yaml p1_goce_244km.yaml
```

Any `dsmc` options as comma-separated `key=value` lists; shortened runs
(100 + 200 steps) with the fenced phase profiler; table of the phases per step
and a check that collisions and surface interactions agree for all variants.

## What is marked in the code

Every phase of the internal profiler (`Profiler::Scope`) is also a Kokkos
profiling region, and `run_dsmc` / `main` add the stages. Regions are only
emitted when a Kokkos tool is loaded (otherwise one branch, no fences):

```
vleo::self_test
vleo::dsmc_setup                     geometry/grid upload, cut cells
vleo::dsmc_fill                      free-stream initial fill
vleo::dsmc_step(warmup|sampling)     one time step
  dsmc.inject  dsmc.move  dsmc.step_reduce  dsmc.migrate(.*)  dsmc.compact
  dsmc.bin.histogram  dsmc.bin.scan  dsmc.bin.fill  dsmc.bin.sort
  dsmc.collide  dsmc.cell_state  dsmc.sample
  vleo::dsmc_state_output           cell-state moments to the host + VTKHDF write
  vleo::dsmc_progress
vleo::dsmc_result
vleo::write_outputs
```

Kernel labels (`vleo::dsmc_move`, `vleo::dsmc_collide_waves`, …) come from
the Kokkos launches themselves; `analyze_profile.py` attaches them to the HIP
kernels through the correlation id of the launching HIP call.

## Reading the results

- `summary.md`: kernel table (calls, total, mean, share), share per group,
  GPU busy fraction of the step loop and the idle gaps between kernels, host
  time in HIP calls (synchronising calls = host waiting for the GPU), copies.
- `timeline_steps.png`: three time steps; rows from bottom: step, phase,
  kernel launches on the host, HIP calls (red = waits for the GPU), GPU
  kernels (coloured by group), copies.
- `timeline_state_output.png`: the step with a cell-state output (GPU idle
  while the host writes HDF5).
- Perfetto (`rocprof/trace/**/*.pftrace`, https://ui.perfetto.dev): zoomable
  version with every call; the Kokkos regions appear as host ranges.

Without a tool, the regions cost nothing measurable; with `phases`, every
phase is fenced, so its numbers add up to the wall time but hide the overlap
of host and GPU that `rocprof` shows.
