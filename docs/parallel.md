# Several GPUs: MPI parallelisation (M6.1–M6.6)

The solver runs on several GPUs (or CPU cores) with MPI, **one rank per GPU**.
A multi-rank run gives the same result as a single-rank run: integer counters
are identical, floating-point quantities differ only by the rounding of the
sums over ranks (checked by the `mpi` tests, section 5).

## 1. Building and starting

```bash
cmake --preset hip-gfx1101-mpi && cmake --build --preset hip-gfx1101-mpi   # 2x W7700 server
cmake --preset cpu-mpi && cmake --build --preset cpu-mpi                   # CPU, one core per rank
mpirun -n 2 build/hip-gfx1101-mpi/vleo_solver --config sim.yaml
```

- `VLEO_ENABLE_MPI=ON` (presets `cpu-mpi`, `hip-gfx1101-mpi`, `hip-gfx1103-mpi`)
  requires an MPI-3 library (`find_package(MPI)`). Without it every MPI call
  is a single-rank no-op (`include/vleo/core/Comm.hpp`), and the build is
  unchanged.
- Data is exchanged through host memory. A GPU-aware MPI is not needed.
- **GPU assignment:** with more than one rank, Kokkos maps rank → GPU as
  *local rank mod visible devices*. It takes the local rank from the
  launcher's environment (`OMPI_COMM_WORLD_LOCAL_RANK`, `SLURM_LOCALID`, …).
  `device_id` in the platform file is then ignored. The actual device of
  every rank is recorded in `run_metadata.json` (`runtime.mpi.rank_device_ids`).
  If several ranks share one GPU, the solver warns.
- **Output:** only rank 0 prints progress and writes files. The other ranks
  print runtime errors only. A runtime error on any rank aborts the whole job
  (`MPI_Abort`), so no rank waits forever in a collective call.
- `vleo_solver --version` shows the MPI library. `results.json` records the
  number of ranks under `diagnostics.parallel`.

## 2. TPMC (M6.2)

Each global batch of histories `[done, done + n)` is split into contiguous
id ranges, one per rank. A history keeps its id and therefore its random
streams. The per-rank moments (mean, M2) are gathered and merged in rank
order with the Chan–Golub–LeVeque update, so every rank holds the same
merged moments and takes the same convergence decision. Counters are summed
exactly, and surface fields are summed at the end.

## 3. DSMC domain decomposition (M6.3, M6.4)

| Step | Single rank | Several ranks |
| ---- | ----------- | ------------- |
| ownership | all cells | contiguous range of z planes = cell range `[c_lo, c_hi)` |
| grid set-up | all cells classified | every P-th cell per rank (interleaved), summed exactly |
| fill | ids `[0, N)` | contiguous id share per rank, then migration to the cell owners |
| injection | count per channel from `(step, channel)` | same counts on all ranks; contiguous share of the injection indices; migration after the move |
| move | all particles | own particles |
| migration | – | after the move: every active particle outside the owned planes is sent to the owner (`MPI_Alltoallv`, 96-byte records with id, interaction count, internal energies) |
| binning | cell lists sorted by storage index | sorted by **global particle id** |
| collisions | all cells | owned cells; streams `(cell, candidate, step)` as before |
| reductions | – | step tallies and status counts (one `MPI_Allreduce` per step); collision counts (one per step); cell populations, surface fields, pair collision counts and the maximum speed once, in `result()` |

**Why the results are identical.** On one rank the storage order equals the
id order (fill, injection and compaction preserve it). Sorting the cell lists
by id on several ranks therefore gives every cell the same particles in the
same order. Pair selection, acceptance and scattering use streams bound to
(cell, candidate, step) and (particle, event), so they reproduce exactly.
Each cell has exactly one owner, which also owns the cell state (σg_max,
NTC remainder). Per-cell sums over ranks have a single non-zero contribution,
so they are exact. Only sums over particles of several ranks (forces, heat,
surface fields) are rounded differently.

**Collective functions.** `DsmcSimulation` (constructor, `fill_freestream`,
`set_particles`, `step`, `rebalance`, `result`), `run_dsmc`,
`refine_cell_size_with_pilot`, `make_collision_grid` and `run_tpmc` must be
called by all ranks. Pilot runs (M5) are ordinary `run_dsmc` calls, so their
diagnostics are global.

`set_particles` (tests) expects the same full particle list on every rank.
Each rank keeps the particles in its own planes. The host accessors
(`positions()`, `velocities()`, …) return the particles of the calling rank;
`num_particles()` is global and `num_local_particles()` is local.

More ranks than z planes is allowed: the extra ranks own no cells and only
move particles before migration.

## 4. Load balancing (M6.5)

The initial split balances the gas volume per plane. At the end of the
warm-up (`DsmcSettings::load_balance`, default on), the planes are re-split
by the particle count of the last binning. Rank 0 computes the split and
broadcasts it. The cell state of every plane that changes owner (σg_max, NTC
remainder, sampled populations) is moved exactly. The particles then migrate
to the new owners. The run continues exactly as before: `test_comm` compares
runs with and without balancing and requires identical counters.
`results.json` reports `initial_plane_bounds`, `plane_bounds` and the
imbalance (max/mean particles per rank) before and after.

## 5. Tests (M6.6)

| Test | Content |
| ---- | ------- |
| `test_comm`, `test_comm_np2`, `test_comm_np3` | reductions, allgather, broadcast, alltoallv (incl. empty blocks); DSMC load balancing on/off identical (CLL, N2/Ar mixture, off-centre body) |
| `mpi_ranks_tpmc` | TPMC O/N2, CLL, 5 batches, surface fields: 1, 2, 3, 4 ranks |
| `mpi_ranks_dsmc_air` | DSMC N2/O (VSS, Larsen–Borgnakke), CLL, satellite, surface fields |
| `mpi_ranks_dsmc_pilot` | DSMC He/Ar, open plate, automatic cell size with pilot runs |

`scripts/check_ranks.py` runs the cases and compares every multi-rank run
with the single-rank run. Integer fields must be identical. Floating-point
fields must agree to 1e-9 relative, or to 10⁻⁶ SE for main quantities that
are close to zero. `surface_fields.csv` is compared the same way. OpenMPI
needs `--oversubscribe` for more ranks than cores (`VLEO_MPIEXEC_FLAGS`,
set automatically).

Fixed bug: since M4, the per-species `mean_particles` diagnostic counted the
slots of ended particles that were not yet compacted. It now counts active
particles only.

## 6. Limits and performance notes

- The decomposition axis is fixed to z (contiguous cell index ranges). For
  domains with few z planes, the balancing granularity is one plane.
- The cell arrays are allocated in full on every rank (memory ∝ total cells
  per GPU). Only the owned range is processed per step.
- The profile in `results.json` is that of rank 0 (`dsmc.migrate`,
  `dsmc.mpi` and `dsmc.rebalance` are the parallel overheads).
- Scaling measurements on 2× W7700: see M6.7 (`benchmarks/README.md`).
