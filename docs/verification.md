# Verification suite

Standardised checks that the DSMC solver reproduces known physics, independent
of the satellite application. The suite grows in stages; every case has a
reference solution, a tolerance fixed before the first run, and a test that
runs in `ctest`.

| Stage | Content | Status |
| ----- | ------- | ------ |
| 0 | Particle level: binary collisions, scattering, ⟨σg⟩, equilibrium, relaxation, wall models, mixtures, internal energy | done ([dsmc.md](dsmc.md) §6, §9; [tpmc.md](tpmc.md)) |
| 1 | Transport coefficients: Couette (viscosity) and Fourier (thermal conductivity), free-molecular and near-continuum | **done** (this page) |
| 1 | Diffusion in a binary mixture, normal shock in argon (Bird, Alsmeyer) | planned |
| 2 | Discretisation: error vs. Δt, Δx and particles per cell on the Couette / Fourier cases | planned |
| 3 | Benchmarks with published data: cylinder in argon at Ma 10 (Lofthouse), sphere drag over Kn, 70° blunted cone | planned |
| 4 | VLEO: C_D(h) DSMC vs. TPMC with upstream-length convergence (GOCE altitude series) | runs exist, not yet a test |

`ctest -L verification` runs the long cases; `ctest -LE verification` skips them.

## 1. Couette and Fourier flow

### Set-up

The gas fills a box between two diffuse walls at y = ±H/2 (`dsmc.faces`,
[dsmc.md](dsmc.md) §11.1). The other faces are periodic (Couette) or specular
(Fourier), so the flow is one-dimensional. Argon, VHS and VSS parameters of
Bird (1994), T_ref = 273 K, ω = 0.81.

| Case | Walls | Measured | Reference |
| ---- | ----- | -------- | --------- |
| Couette | T = 273 K, u = ∓U in x | wall shear τ = −F_x / A, core gradient du/dy | μ = τ / (du/dy) |
| Fourier | T₁ (y_min), T₂ (y_max), at rest | heat flux q into the cold wall, core profile T(y) | κ from q = −κ dT/dy |

### Free-molecular limit (Kn ≈ 1300, `test_dsmc_walls`)

Every molecule flies from wall to wall. With n the mean number density in the
gap and c̄ = √(8kT/πm):

- Couette: one-way flux n c̄/4, each hit carries 2mU of x-momentum and 2mU² of
  energy in the wall frame, so τ = ½ n m c̄ U and q = ½ n m c̄ U² at both walls.
- Fourier: half-Maxwellians from both walls with equal fluxes Γ,
  n = 2Γ (1/c̄₁ + 1/c̄₂), q = 2kΓ (T₁ − T₂).

Tolerance 5 SE + 0.5 % (residual collisions, finite number of wall hits).
Result: shear and heat of both walls within the tolerance; the statistical
error is about 0.05 %.

### Near-continuum limit (Kn = 0.05, `test_dsmc_transport`)

H = 20 λ, Δy = λ/3, Δt = 0.2 τ, 100 particles per cell (24 000 particles),
n = 10²¹ m⁻³. Couette: U = ±100 m/s. Fourier: T₁ = 323 K, T₂ = 223 K. The runs
start from the linear profiles (Fourier: density ∝ 1/T), warm up for 2000 steps
(about 5 diffusion times) and sample 2 × 10⁴ steps in 10 blocks.

- **Gradients** come from a least-squares fit in the core |y| < 0.3 H, outside
  the Knudsen layers. For the Fourier case κ ∝ T^ω is accounted for exactly by
  fitting Θ = T^(1+ω): q = −κ_ref T_ref^(−ω) / (1 + ω) · dΘ/dy.
- **Fluxes** come from the wall tallies (`diagnostics.walls`). As a
  conservation check they are compared with the fluxes through 21 planes in the
  core, counted from the particles that cross a plane during a step.
- **Reference:** first-order Chapman–Enskog for VHS/VSS (Bird 1994,
  eqs. 4.62–4.64),
  μ = 5(α+1)(α+2) √(π m k T_ref) / (4α (5−2ω)(7−2ω) π d²) · (T/T_ref)^ω,
  κ = 15k/(4m) μ. The exact Chapman–Enskog values for ω = 0.81 are higher by
  0.2 % (μ) and 0.3 % (κ) (Sonine corrections for the equivalent inverse-power
  potential).
- **Tolerance:** 5 SE + 2 % for the coefficients, 5 combined SE for wall vs.
  plane fluxes.

Results (cpu-release, 16 OpenMP threads, 2026-11-26):

| Model | μ / μ_CE | κ / κ_CE | wall τ vs. plane flux | wall q vs. plane flux |
| ----- | -------- | -------- | --------------------- | --------------------- |
| VHS | 1.000 ± 0.008 | 1.006 ± 0.010 | 0.1459 / 0.1449 Pa (0.7 σ) | 49.65 / 49.71 W/m² (0.2 σ) |
| VSS | 1.019 ± 0.008 | 0.985 ± 0.012 | 0.1429 / 0.1421 Pa (0.9 σ) | 48.64 / 48.49 W/m² (0.3 σ) |

HIP on the laptop GPU (gfx1103, same seeds; the trajectories differ from the
CPU run because the wall and collision sampling use the device math library):
VHS μ/μ_CE = 0.994 ± 0.008, κ/κ_CE = 1.015 ± 0.013; VSS 1.005 ± 0.008 and
0.992 ± 0.012; wall and plane fluxes within 1.3 σ. Run time of
`test_dsmc_transport`: 5 min on the CPU, 8 min on the laptop GPU (24 000
particles are far too few to fill a GPU; the host-side plane counting
dominates). The W7700 (gfx1101, hagen) gives bit-for-bit the same numbers as
the laptop GPU.

**Finding: post-collision stress is biased.** The box average Σ m v_x v_y / V
of the particle velocities after the collisions is 6.8 % (VHS) below the wall
shear at Δt = 0.2 τ and 2.6 % at Δt = 0.05 τ, independent of the cell size
(λ/3 and λ/6 give the same). The collisions at the end of each step relax the
stress by about Δt/τ_μ and the free flight rebuilds it, so the post-collision
value is low by about Δt/(2τ_μ). Fluxes must therefore be taken from the walls
or from plane crossings, not from post-collision moments (see [dsmc.md](dsmc.md)
§11.1).

### Running a case with the solver

Results on the W7700 (hagen, 2026-11-26; 4000 warm-up + 20 000 sampled steps,
10 cell-state files of the sampling phase, H = 24 mm, Kn = 0.054, VHS):

| Case | Wall flux | Core gradient | Result |
| ---- | --------- | ------------- | ------ |
| `couette_argon.yaml` | τ = 0.15541 ± 0.00067 Pa | du/dy = 7161 ± 9 s⁻¹, core T 280.5 K | μ = 2.1704·10⁻⁵ Pa s, +0.34 % vs. Chapman–Enskog (0.8 σ) |
| `fourier_argon.yaml` | q = 52.99 ± 0.30 W/m² | T^(1+ω) fit | κ/κ_CE = 1.0045 ± 0.0065 (0.7 σ) |

Both agree with the first-order Chapman–Enskog values within one standard
error and with the expected higher-order correction (+0.2–0.3 %).


`verification/couette_argon.yaml` and `verification/fourier_argon.yaml` set up
the near-continuum cases for the command-line solver (uniform initial gas,
longer warm-up, cell-state output for the profiles);
`verification/evaluate_transport.py <run-dir>` fits the profiles from the
cell-state files of the sampling phase and prints μ or κ against Chapman–Enskog
(needs numpy and h5py or VTK).
