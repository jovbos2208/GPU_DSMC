# Verification suite

Standardised checks that the DSMC solver reproduces known physics, independent
of the satellite application. The suite grows in stages; every case has a
reference solution, a tolerance fixed before the first run, and a test that
runs in `ctest`.

| Stage | Content | Status |
| ----- | ------- | ------ |
| 0 | Particle level: binary collisions, scattering, ⟨σg⟩, equilibrium, relaxation, wall models, mixtures, internal energy | done ([dsmc.md](dsmc.md) §6, §9; [tpmc.md](tpmc.md)) |
| 1 | Transport coefficients: Couette (viscosity) and Fourier (thermal conductivity), free-molecular and near-continuum | **done** (this page) |
| 1 | Normal shock in argon: Rankine–Hugoniot, thickness vs. Navier–Stokes, Mach series | **done** (ctest at Ma 1.4; solver cases Ma 1.4–8) |
| 1 | Diffusion in a binary mixture; shock thickness vs. experiment (Alsmeyer) and a second code (SPARTA) | planned |
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

## 2. Normal shock in argon

### Set-up

A tube x ∈ [−30, 30] λ₁ between two inflow faces (`dsmc.faces` type
`inflow`, [dsmc.md](dsmc.md) §11.1): x_min injects the upstream state
(n₁ = 10²¹ m⁻³, T₁ = 273 K, u₁ = Ma a₁), x_max the Rankine–Hugoniot downstream
state (monatomic, γ = 5/3). Both halves are filled with their state at the
start (`fill_depth_m`), so the shock forms at x = 0 within a few collision
times. The side faces are specular. VHS argon (Bird 1994), cells λ₁/4.
λ₁ is the VHS mean free path of the upstream gas (`mean_free_path_inf_m`).

The shock position is neutrally stable and performs a slow random walk
(up to 2–3 λ₁ over a run). Profiles are therefore aligned on the shock centre
(n = (n₁ + n₂)/2) per averaging window before they are added.

**Thickness.** δ = 2w of the least-squares fit
n = (n₁+n₂)/2 + (n₂−n₁)/2 · tanh((x − x₀)/w) over |x| < 20 λ₁, the same estimator
for DSMC and Navier–Stokes. (The textbook definition (n₂−n₁)/max dn/dx taken
from the largest local slope of a noisy profile is biased: the maximum of a
noisy slope is too large, δ too small – 7.3 ± 1.5 λ₁ instead of 11.1 ± 0.2 λ₁
at Ma 1.4.)

**Navier–Stokes reference.** In the shock frame with the mass flux m₀ = ρ₁u₁
and the momentum and energy fluxes P, E of the upstream state:
4/3 μ u′ = m₀u + ρRT − P,  κ T′ = m₀c_vT − m₀u²/2 + Pu − E, with the same μ(T)
(first-order Chapman–Enskog VHS) and κ = 15kμ/(4m). The profile is the
trajectory from the upstream node into the downstream saddle, integrated
backwards from the saddle along its stable eigenvector (RK4, 0.2 µm steps;
C++ in `test_dsmc_shock`, Python in `verification/shock_profile.py`, identical
to 4 digits).

### Results (cpu-release, `test_dsmc_shock_ma14` and runs at other Mach numbers, 2026-11-26)

| Ma | n₂/n₁ (RH) | T₂/T₁ (RH) | δ DSMC [λ₁] | δ NS [λ₁] | δ_DSMC / δ_NS |
| -- | ---------- | ---------- | ----------- | --------- | ------------- |
| 1.2 | 1.278 (1.297) | 1.192 (1.195) | 20.4 ± 0.5 | 19.35 | 1.05 ± 0.03 |
| 1.4 | 1.577 (1.581) | 1.392 (1.392) | 11.12 ± 0.16 | 10.12 | 1.10 ± 0.02 |
| 2.0 | 2.286 (2.286) | 2.081 (2.078) | 6.18 ± 0.03 | 4.87 | 1.27 ± 0.01 |

(At Ma 1.2 the shock, 20 λ₁ thick, does not reach its plateaus within ±30 λ₁.)
The plateaus reproduce Rankine–Hugoniot to 0.1–0.3 %. Navier–Stokes
underestimates the thickness increasingly with the Mach number, as expected
for a shock that is a few mean free paths thick; the ratio tends to 1 for weak
shocks. The ctest therefore requires 1 ≤ δ/δ_NS ≤ 1.2 at Ma 1.4 (5 SE margin)
and 1–1.5 % on the plateaus and the mass flux.

### Mach series with the solver

```bash
verification/make_shock_configs.py --mach 1.4 2 3 5 8   # shock_ma*.yaml + batch_shock.txt
cd verification && vleo_solver --batch batch_shock.txt --platform <platform.yaml>
verification/shock_profile.py results/shock_ma* --png shock.png
```

4 × 4 cells across, 50 particles per cell (0.3–0.58 M particles), Δt from
min(0.1 τ₁, 0.2 τ₂, Δx/(u + 2c_mp)), 4 + 20 downstream flow-through times,
40 cell-state files. The script aligns the files, prints the Rankine–Hugoniot
checks and the thickness for DSMC and Navier–Stokes, and plots the profiles and
the reciprocal thickness over Ma.

Results on the W7700 (hagen, 2026-11-26; 33–79 s per case). Plateaus 15–25 λ₁
behind the centre against the prescribed upstream state; δ_tanh = 2w of the
tanh fit with the VHS λ₁; λ_μ/δ_max: Alsmeyer's normalisation,
λ_μ = 16μ₁/(5ρ₁√(2πRT₁)) = 1.3198 λ₁ and δ = (n₂ − n₁)/max dn/dx:

| Ma | n₂/n₁ (RH) | T₂/T₁ (RH) | n₂u₂/(n₁u₁) | δ_tanh DSMC [λ₁] | δ_tanh NS [λ₁] | DSMC/NS | λ_μ/δ_max DSMC | λ_μ/δ_max NS |
| -- | ---------- | ---------- | ----------- | ---------------- | -------------- | ------- | -------------- | ------------ |
| 1.4 | 1.5798 (1.5806) | 1.3919 (1.3918) | 1.0000 | 10.78 ± 0.05 | 10.12 | 1.07 | 0.126 | 0.130 |
| 2 | 2.2857 (2.2857) | 2.0783 (2.0781) | 0.9999 | 6.119 ± 0.006 | 4.869 | 1.26 | 0.218 | 0.269 |
| 3 | 3.0004 (3.0000) | 3.6661 (3.6667) | 1.0000 | 5.232 ± 0.003 | 3.536 | 1.48 | 0.256 | 0.370 |
| 5 | 3.5723 (3.5714) | 8.6780 (8.6800) | 0.9999 | 5.764 ± 0.006 | 3.454 | 1.67 | 0.237 | 0.381 |
| 8 | 3.8210 (3.8209) | 20.869 (20.872) | 1.0003 | 7.158 ± 0.006 | 4.065 | 1.76 | 0.191 | 0.326 |

(Standard errors of δ from 5 groups of 8 files; the max-slope value is taken from
the fully averaged profile.)

- Rankine–Hugoniot is reproduced to 0.05 % or better at all Mach numbers, the
  mass flux to 0.03 %.
- Navier–Stokes is adequate for weak shocks (3 % in λ_μ/δ at Ma 1.4) and
  predicts shocks that are 1.5–1.8 times too thin from Ma 3 on.
- The reciprocal thickness has its maximum near Ma 3 and falls at higher Mach
  numbers, the shape of the argon measurements (Alsmeyer 1976, Ma 1.55–9,
  [JFM 74, 497](https://www.doi.org/10.1017/S0022112076001912)). Alsmeyer found the best agreement with a
  viscosity exponent ω = 0.72; the argon data of this solver use ω = 0.81
  (Bird 1994), which gives somewhat thicker strong shocks. A quantitative
  comparison needs the tabulated measurements.
- The temperature rises ahead of the density; the upstream temperature
  precursor grows with Ma (at Ma 8, 23 % of the jump at 5–10 λ₁ ahead of the
  centre, 1 % at 15–20 λ₁). At the inflow face (27–30 λ₁) it is below 0.05 %
  of the jump, so the tube is long enough. A plateau ratio taken against a
  *measured* T₁ 15–25 λ₁ upstream is biased by this precursor (T₂/T₁ = 18.5
  instead of 20.9 at Ma 8); the table uses the prescribed upstream state.

Next: SPARTA on the same five cases (independent code), and the measured
λ/δ of Alsmeyer once the table is available.
