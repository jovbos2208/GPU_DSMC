# DSMC with elastic VHS collisions (M3)

DSMC extends the collisionless transport of TPMC with intermolecular
collisions. Geometry, inflow, the Maxwell/CLL kernels, the event balance and
the result schema are shared with TPMC ([geometry.md](geometry.md),
[tpmc.md](tpmc.md)).

**Scope of this release:** one monatomic species with VHS data (He, Ar from
Bird 1994, Table A1). Gas mixtures (species-pair collisions), atomic O/N
collision data and internal energy follow in M4; the configuration rejects
them for `mode: dsmc` with an explicit message. The collision kernel itself is
already written for pairs of different masses.

## 1. Time step (M3.1)

| # | Phase | Details |
| - | ----- | ------- |
| 1 | inject | per (face, species) channel: k = ⌊R_c Δt / W + U⌋ new particles (U from the stream (step, channel)); each enters at a random time within the step |
| 2 | move | ballistic motion for Δt with surface interactions (Maxwell/CLL), multiple reflections and the event balance of tpmc.md §2; leaving the domain removes the particle (open boundary) |
| 3 | sample | sampling phase only: per-step sums of Δp, Δl, ΔE, ΔE± (deterministic reduction), surface fields (atomic), cell populations |
| 4 | compact | order-preserving removal of exited/failed particles (prefix sum) |
| 5 | bin | cell index → histogram (atomic counts) → exclusive prefix sum → fill → per-cell sort by particle id |
| 6 | collide | NTC per cell (section 4) |

All random numbers are bound to (cell, step), (particle id, event) or
(step, channel). The particle order is deterministic (prefix-sum compaction,
id-sorted cells) and the global tallies are reductions, so a run is bitwise
reproducible for a given backend and configuration (tested with 12 OpenMP
threads and on the GPU).

The initial state is free-stream gas: particles are placed uniformly in the
domain box with the drifting Maxwellian, those inside the body are discarded.

## 2. Collision grid and cut cells (M3.2, M3.3, M3.9, M3.10)

- Cartesian grid over the domain box; n_k = ⌈L_k / Δx⌉ cells per axis with
  edge L_k / n_k ≤ Δx (the box is divided exactly). c = ix + nx (iy + ny iz).
  Positions on or rounded slightly beyond the box are clamped to the boundary cell.
- **Binning** is verified in `test_grid`: every particle appears exactly once,
  in the cell of its position, sorted by id.
- **Gas fraction f of a cell:** cells not touched by a closed surface use a
  point-in-body test of their centre (f = 0 or 1). Cells overlapping the
  bounding box of a closed-surface triangle (plus one neighbour layer) are
  sampled with n³ stratified points (default n = 16); f is the share outside
  every closed component. Point-in-body uses the crossing parity along three
  generic directions with a majority vote. Open (two-sided) surfaces have no
  volume.
- **Accuracy:** exact when the surface lies on the sample sub-grid (tested
  with an axis-aligned box at 1/16 offsets); otherwise the error is confined to
  a shell of half a sample spacing, |ΔV| ≤ A · Δx / (2n) (tested for a sphere
  with n = 8, 16, 32). Thin parts below the sample spacing (e.g. 2 cm panels in
  22 cm cells) are resolved only approximately; the grid report states the
  total gas volume for checking.
- **Small cells:** the collision volume is V_eff = max(f, f_min) V_cell with
  f_min = 0.05, so a tiny gas fraction cannot inflate the candidate number;
  the number of such cells is reported as a warning.

## 3. Numerical parameters and automatic choices

| Parameter | Fixed | Automatic |
| --------- | ----- | --------- |
| cell size | `dsmc.cell_size_m` | min(λ∞ / 2, L_max / 16) |
| time step | `dsmc.timestep_s` | min(τ∞ / 10, Δx_min / (4 (V∞ + 3 c_mp))) |
| particle weight | – | W = n∞ V_cell / `particles_per_cell_target` |
| warm-up | `dsmc.warmup_steps` | 3 · (domain diagonal / max(V∞, c_mp)) / Δt |
| sampling | `dsmc.sample_steps` (default 1000), `dsmc.blocks` (default 20) | – |
| capacity | `dsmc.max_particles` | max(10⁶, 6 × free-stream population) |

λ∞ = c̄ / (n∞ ⟨σg⟩) and τ∞ = 1 / (n∞ ⟨σg⟩) with the equilibrium VHS average
⟨σ_T g⟩ = 2√π d_ref² √(2kT_ref/m_r) (T/T_ref)^{1−ω}. The automatic values are
starting points, not accuracy guarantees: section 6 shows the separate
convergence studies, and the run diagnostics report the achieved resolution.

## 4. NTC collisions (M3.4 – M3.8)

Per cell with N particles and collision volume V_eff:

- candidates: N_cand = ½ N (N − 1) W (σ_T g)_max Δt / V_eff + r, the integer
  part is used, the fraction r is carried to the next step of that cell;
- pair selection: i uniform in [0, N), j uniform in [0, N − 1) shifted past i,
  so a particle never collides with itself;
- acceptance: σ_T(g) g / (σ_T g)_max with the VHS cross-section
  σ_T(g) = π d² (2kT_ref/(m_r g²))^{ω−1/2} / Γ(5/2 − ω);
- the bound (σ_T g)_max is per cell, initialised with σ_T g at 3 c_mp and
  raised when a candidate exceeds it. Each cell is processed by one thread, so
  the update is race-free. The candidate number of the current step uses the
  bound from its start (Bird's practice); the counted updates show that the
  bounds settle during the warm-up;
- post-collision velocities: v_cm is kept, g' = |g| · (isotropic unit vector),
  v_p' = v_cm + m_q/(m_p+m_q) g', v_q' = v_cm − m_p/(m_p+m_q) g'.
- diagnostics: candidates, accepted, rejected, bound updates, acceptance ratio
  and collisions per particle per step (compare with the free-stream Δt/τ∞).

## 5. Sampling and statistics (M3.11 – M3.13)

- **Warm-up and sampling are separate phases:** only steps after
  `warmup_steps` enter the statistics and the surface fields.
- Per sampled step the rates are (W/Δt) · Σ (event values of that step); the
  result is the mean over the sampling phase.
- **Standard error by block averaging:** the sampling phase is split into
  `blocks` equal blocks, SE = sd(block means)/√B. The integrated
  autocorrelation time τ_int (Geyer's initial positive sequence) is reported for
  the drag and the heat rate; blocks shorter than 5 τ_int trigger a warning.
  Verified on an AR(1) process with known τ_int (`test_dsmc`).
- **Resolution diagnostics** (results.json → `diagnostics`): Δx, Δt, λ∞, τ∞,
  Δx/λ∞, Δt/τ∞, maximum CFL number, minimum local mean free path (from the
  sampled density, λ∞ n∞ / n_cell), share of gas cells with Δx > local λ,
  mean particles per gas cell, share of cells with < 5 particles, particle
  numbers, grid report (cut cells, small cells, gas volume). Warnings are
  issued for Δx > local λ, Δt > 0.2 τ∞, CFL > 1, > 10 % cells with < 5
  particles, small cut cells, short blocks, truncated or failed particles.

## 6. Verification and convergence studies (M3.14 – M3.17)

Tolerances were fixed before the tests were run:

| Check | Tolerance | Test |
| ----- | --------- | ---- |
| binary collisions conserve momentum and energy, \|g\| preserved | 1e-13 relative | `test_vhs` (equal and unequal masses, on the device) |
| isotropic scattering | KS √n D < 1.95, ⟨g'·g⟩ within 5 SE | `test_vhs` |
| ⟨σg⟩ formula vs. numerical Maxwell average | 1e-8 | `test_vhs` |
| closed specular argon box: energy conserved over the run | 1e-10 relative | `test_dsmc` |
| collision count vs. equilibrium theory ½ N(N−1)/M · W⟨σg⟩Δt/V_c | 5 √K + 0.5 % | `test_dsmc` (3.2·10⁵ collisions: −0.34 %, 1.9 σ) |
| equilibrium velocity distribution stays Maxwellian | KS | `test_dsmc` |
| relaxation T_x = 2T, T_y = T_z = T/2 → isotropic after 30 τ | 5 SE | `test_dsmc` (1.006 / 1.005 / 0.989, SE 0.0125) |
| cut-cell volumes | exact / shell bound | `test_grid` |
| free-molecular DSMC vs. TPMC (satellite, CLL) | 5 √(SE_a² + SE_b²) | `test_dsmc`; long run with 6·10⁴ sampled steps: C_D, C_L, C_Y, Q̇ within 1.5 σ (ΔC_D 3·10⁻⁴) |
| reproducibility | bitwise | `test_dsmc`, smoke `repro_dsmc_*` |

The studies M3.16/M3.17 are produced by `scripts/dsmc_studies.py`
(0.5 m cube, helium, 900 K, 7.5 km/s, diffuse wall at 300 K; A_ref = 0.25 m²).
Each parameter is varied with the other two fixed; the recorded tables are in
section 8.

## 7. Output

`results.json` uses the TPMC schema (tpmc.md §8) with a DSMC-specific
`statistics` block (warm-up and sampled steps, blocks, block length, τ_int,
naive SE for comparison) and `diagnostics` (section 5, collision statistics,
grid report, particle balance). CLI overrides: `--warmup-steps N`,
`--sample-steps N`.

## 8. Recorded study results

Generated with `scripts/dsmc_studies.py build/cpu-release/vleo_solver <dir>`
(2026-11-12, cpu-release, 12 OpenMP threads; 400 warm-up + 4000 sampled steps,
automatic cell size and time step, 30 particles per cell).

### Dilution (M3.16): DSMC → TPMC as n → 0

TPMC reference (free-molecular): C_D = 2.9125 ± 0.0008.

| n∞ [m⁻³] | λ∞ | collisions / particle / step | C_D (DSMC) | ΔC_D vs. TPMC |
| -------- | -- | ---------------------------- | ---------- | ------------- |
| 1e14 | 50 km | 0 | 2.9114 ± 0.0015 | −0.04 % (0.6 σ) |
| 1e17 | 50 m | 1.8e-4 | 2.9096 ± 0.0015 | −0.10 % |
| 1e18 | 5 m | 1.9e-3 | 2.8954 ± 0.0015 | −0.59 % |
| 1e19 | 0.5 m | 0.025 | 2.8510 ± 0.0014 | −2.1 % |
| 1e20 | 5 cm | 0.14 | 2.4673 ± 0.0009 | −15.3 % |

In the free-molecular limit DSMC reproduces TPMC within the statistical
error; with increasing density the drag decreases monotonically (shielding
by the compressed gas in front of the body), as expected in the transitional
regime.

### Cell size, time step, particles per cell (M3.17)

Run on the `hip-gfx1101` build (2× Radeon PRO W7700, HIP 6.3.1), 2026-11-13;
same cube case at n∞ = 1e19 (λ∞ = 0.50 m, τ∞ = 2.3e-4 s, Kn ≈ 1 based on the
cube edge). Each parameter is varied with the other two fixed; ΔC_D is relative
to the most refined case of the study.

**Cell size** (Δt = 2 µs, 30 particles per cell target):

| Δx | Δx / λ∞ | C_D | Q̇ [W] | ΔC_D | particles / gas cell | collisions / particle / step |
| -- | ------- | --- | ----- | ---- | -------------------- | ---------------------------- |
| 0.2 m | 0.40 | 3.0752 ± 0.0071 | 4888 ± 11 | +8.06 % | 53.3 | 0.0459 |
| 0.1 m | 0.20 | 2.9471 ± 0.0020 | 4566 ± 3.9 | +3.56 % | 51.9 | 0.0516 |
| 0.05 m | 0.10 | 2.8528 ± 0.0011 | 4393 ± 1.8 | +0.24 % | 59.2 | 0.0532 |
| 0.025 m | 0.05 | 2.8458 ± 0.0004 | 4355 ± 0.61 | 0 | 58.7 | 0.0542 |

**Time step** (Δx = 0.05 m, 30 particles per cell target):

| Δt | Δt / τ∞ | C_D | Q̇ [W] | ΔC_D | collisions / particle / step |
| -- | ------- | --- | ----- | ---- | ---------------------------- |
| 8 µs | 0.035 | 2.8551 ± 0.0004 | 4396 ± 0.65 | +0.15 % | 0.213 |
| 4 µs | 0.017 | 2.8537 ± 0.0008 | 4394 ± 1.1 | +0.10 % | 0.107 |
| 2 µs | 0.0087 | 2.8528 ± 0.0011 | 4393 ± 1.8 | +0.07 % | 0.053 |
| 1 µs | 0.0043 | 2.8508 ± 0.0016 | 4393 ± 2.5 | 0 | 0.027 |

**Particles per cell** (Δx = 0.05 m, Δt = 2 µs; "target" refers to the
free-stream density, the actual mean is about twice as high because of the
dense, slow re-emitted gas near the body):

| target | particles / gas cell | C_D | Q̇ [W] | ΔC_D |
| ------ | -------------------- | --- | ----- | ---- |
| 5 | 9.9 | 2.8524 ± 0.0023 | 4394 ± 3.7 | +0.02 % |
| 15 | 29.6 | 2.8529 ± 0.0011 | 4394 ± 1.6 | +0.03 % |
| 45 | 88.8 | 2.8532 ± 0.0008 | 4394 ± 1.3 | +0.05 % |
| 135 | 266.2 | 2.8519 ± 0.0005 | 4392 ± 0.73 | 0 |

**Conclusions**

- **Cell size is the dominant numerical parameter.** C_D and Q̇ converge from
  above; the error is +8 % at Δx = 0.4 λ∞ and +0.24 % at 0.1 λ∞. The reason is
  the compressed gas in front of the body, where the local mean free path is
  far below λ∞ (the run diagnostics report it). A cell size of λ∞/2 is
  therefore **not** sufficient for blunt bodies at Kn ≈ 1; the automatic rule
  min(λ∞/2, L_max/16) happened to give the converged 0.05 m here only through
  the L_max/16 cap. Use the reported `fraction_cells_dx_gt_local_lambda` and a
  cell-size study for production runs (see the open item below).
- **Time step:** weak, monotonic dependence, ≤ 0.15 % up to Δt = 0.035 τ∞
  (CFL ≈ 2 for the fastest particles at 8 µs). The automatic τ∞/10 rule is
  limited by the cell-crossing criterion in such cases.
- **Particles per cell:** no dependence above the statistical noise
  (≤ 0.05 %) from 10 to 266 particles per gas cell; more particles only reduce
  the statistical error.
- Q̇ shows the same behaviour as C_D (cell size +11 % at 0.4 λ∞, otherwise
  ≤ 0.1 %).

**Open item (for M5):** make the automatic cell size depend on the local mean
free path (adaptive refinement or a λ-based rule from a short pilot run)
instead of λ∞.
