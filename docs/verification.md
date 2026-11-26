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
| 1 | Binary diffusion (O/N, He/Ar; VHS, VSS) vs. Chapman–Enskog | **done** |
| 1 | Shock thickness vs. experiment (Alsmeyer 1976) | qualitative only (shape of λ/δ(Ma)); the tabulated measurements were not accessible |
| 2 | Discretisation: error vs. Δt, Δx and particles per cell on the Couette / Fourier cases | **done** |
| 3 | Hypersonic cylinder in argon (Lofthouse 2008), Ma 10 and 25, Kn 0.25 and 0.05: drag, peak heat flux | **done** (solver runs, section 5) |
| 3 | Sphere drag over Kn in argon (Ma 4, Kn 10 … 0.03, free-molecular limit) vs. exact theory and correlations | **done** (solver runs, section 6) |
| 3 | 70° blunted cone | planned |
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

## 3. Binary diffusion

### Set-up (`test_dsmc_diffusion`, label `verification`)

An equimolar mixture A/B at n = 10²¹ m⁻³, T = 273 K in a closed specular box
x ∈ [0, L] (2 × 2 cells across, cells λ/3, 100 particles per cell, Δt = 0.2 τ;
λ, τ of the mixture). The particles start with uniform density and
temperature and the composition x_A(x) = ½ + 0.4 cos(πx/L). With specular
walls the cosine is an eigenmode of Fick's equation and the molar-average
velocity stays zero, so the amplitude decays as exp(−D₁₂π²t/L²). The amplitude
a(t) = (2/N) Σ_A cos(πx/L) is sampled every 5 steps and ln a is fitted from 20 τ
until a ≈ 0.15 a₀; independent runs (seeds) give the standard error.

Reference: first-order Chapman–Enskog for the VHS / VSS pair of the solver
(diameter, ω, T_ref and α averaged):
D₁₂ = 3(α+1)/(8(5−2ω)) · √(2πkT/m_r)/(nπd²) · (T/T_ref)^(ω−½)
(VSS momentum-transfer cross-section 2σ_T/(α+1); hard spheres:
3/(8nd²)·√(kT/(2πm_r))).

### Results (W7700, 16 runs per point, 2026-11-26)

D₁₂ / D₁₂(Chapman–Enskog, 1st order) against the box length:

| L / λ | kλ | O/N VHS | He/Ar VHS | He/Ar VSS |
| ----- | -- | ------- | --------- | --------- |
| 15 | 0.21 | 0.998 ± 0.020 | 0.958 ± 0.028 | 0.687 ± 0.025 |
| 30 | 0.10 | 0.999 ± 0.007 | 1.072 ± 0.016 | 1.038 ± 0.011 |
| 60 | 0.05 | 1.001 ± 0.006 | 1.080 ± 0.008 | 1.088 ± 0.007 |

- **O/N** (masses 16/14, identical collision data; the higher Chapman–Enskog
  orders are below 1 %): the first approximation is reproduced to 0.1–0.2 % at
  every box length.
- **He/Ar** converges to 1.08–1.10 times the first approximation as kλ → 0. The
  higher Chapman–Enskog orders are positive for very different masses (up to
  +13 % for hard spheres in the Lorentz limit); the value has not yet been
  compared with Kihara's second approximation. The ratio of the VSS to the VHS
  result, 1.207, matches the ratio of the first approximations, 1.198.
- **Finite wavelength.** A short box slows the decay strongly when one species
  is light: helium has a mean free path several times the mixture λ, so its
  effective kλ is larger (He/Ar VSS: −31 % at L = 15 λ). For mixtures with
  light species the domain must be long compared with the mean free path of
  the light species, not only with the mixture λ.

The ctest runs L = 30 λ with 6 runs (40 s on the CPU) and requires O/N within
5 SE + 2 % and He/Ar between 0.97 and 1.12 times the first approximation.
`DIFF_L=60 DIFF_RUNS=16 tests/test_dsmc_diffusion -s` repeats the converged study.

## 4. Discretisation errors (stage 2)

### Set-up

The near-continuum Couette and Fourier cases of section 1 (argon, VHS,
Kn = 0.05, μ from the wall shear, κ from the wall heat flux) with one parameter
varied at a time around the reference Δx = λ/3, Δt = 0.2 τ, 100 particles per
cell, 24 000 particles. The particle number and the simulated time stay the
same in every sweep (larger cells: more particles per cell; fewer particles per
cell: wider channel; smaller Δt: more steps), so all points have the same
statistics. Study mode of `test_dsmc_walls` (`VLEO_STUDY_*`), driven by

```bash
verification/run_transport_study.py run build/<preset>/tests/test_dsmc_walls   # 30 runs, resumable
verification/run_transport_study.py plot                                       # fit and figure
```

### Results (W7700, 2026-11-26)

Ratio to first-order Chapman–Enskog (± 1 SE, ≈ 0.01):

| Δx/λ (Δt = 0.2 τ) | 1/3 | 2/3 | 1 | 1.5 | 2 |
| ----------------- | --- | --- | - | --- | - |
| μ | 0.995 | 1.017 | 1.064 | 1.106 | 1.213 |
| κ | 1.015 | 1.011 | 1.029 | 1.058 | 1.073 |

| Δt/τ (Δx = λ/3) | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
| --------------- | ---- | --- | --- | --- | --- |
| μ | 1.006 | 0.994 | 0.995 | 1.007 | 1.024 |
| κ | 0.985 | 1.009 | 1.015 | 0.997 | 1.000 |

| particles per cell | 5 | 10 | 20 | 50 | 100 |
| ------------------ | - | -- | -- | -- | --- |
| μ | 0.972 | 0.996 | 1.005 | 1.000 | 1.005 |
| κ | 0.972 | 0.982 | 1.000 | 1.000 | 1.003 |

Weighted fits r₀ + b x² (λ, τ: VHS mean free path and collision time of the
solver). The hard-sphere theory (Alexander, Garcia & Alder 1998; Garcia &
Wagner 2000; Hadjiconstantinou 2000) is written in the hard-sphere mean free
path, which for VHS argon corresponds to λ_μ = 16μ/(5ρ√(2πRT)) = 1.32 λ; the
coefficients below are converted to λ_μ, τ_μ = λ_μ/c̄ (factor 1.32² = 1.74):

| Error term | b (λ, τ) | b (λ_μ, τ_μ) | hard spheres |
| ---------- | -------- | ------------ | ------------ |
| μ, cell size | 0.055 ± 0.003 | 0.096 ± 0.005 | 16/(45π) = 0.113 |
| κ, cell size | 0.016 ± 0.004 | 0.029 ± 0.007 | 32/(225π) = 0.045 |
| μ, time step | 0.043 ± 0.014 | 0.075 ± 0.025 | 16/(75π) = 0.068 |
| κ, time step | 0.004 ± 0.021 | 0.007 ± 0.037 | 64/(675π) = 0.030 |

- The cell size is the dominant discretisation error. It is quadratic, as the
  theory predicts, and somewhat smaller than for hard spheres (VHS argon,
  ω = 0.81, is a softer potential): +1.4 % in μ at Δx = λ/2, +5.5 % at Δx = λ.
- The time step matters little below 0.4 τ; μ rises by 2.5 % at 0.8 τ, κ
  shows no measurable effect.
- From 20 particles per cell on there is no effect within 1 %. With 10 the
  conductivity is 2 % low, with 5 both coefficients are 3 % low (2–3 σ): few
  particles per cell underestimate the transport coefficients.
- Recommended for transport-dominated regions: Δx ≤ λ/2 (local λ), Δt ≤ 0.4 τ
  (local τ), ≥ 20 particles per cell.
- GOCE at 120 km (0.1 m cells, 8–20 particles per cell): next to the body
  Δx/λ_local ≈ 0.8, i.e. about +3.5 % in μ and +1 % in κ there, falling off
  quickly away from the body; 8–12 particles per cell in the long-domain runs
  lower the coefficients by up to 1–2 %; Δt/τ ≪ 0.1. These errors are small
  compared with the 15–17 % collision effect on C_D at 120 km.

## 5. Hypersonic cylinder in argon (stage 3)

### Set-up

Reference: Lofthouse, *Nonequilibrium hypersonic aerothermodynamics using the
direct simulation Monte Carlo and Navier–Stokes models*, PhD thesis, University
of Michigan 2008, chapter 4 (DSMC code MONACO), tables 4.1–4.4. Cylinder of
0.3048 m diameter in argon, T∞ = 200 K, U∞ = 2624 m/s (Ma 10, wall 500 K) or
6585 m/s (Ma 25, wall 1500 K), fully diffuse wall, VHS argon ω = 0.734,
T_ref = 1000 K, d_ref = 3.595·10⁻¹⁰ m (`gas.collision_data`), n∞ from table 4.1
(Kn∞ = 0.25: 1.699·10¹⁹ m⁻³, Kn∞ = 0.05: 8.494·10¹⁹ m⁻³).

The 3D solver runs a two-dimensional half model: symmetry plane y = 0 and the z
faces specular, slab one cell thick, the 360-facet cylinder cut by the box
(`domain.clip_geometry`), the faces x_min, x_max and y_max are inflow faces with
the free-stream state. Uniform cells: 1 mm (Kn 0.25, 2.2 M cells, domain
−6R…6R × 8R) and 0.5 mm (Kn 0.05, 3.3 M cells, −3R…4R × 5R); time step
min(0.2 τ_wall, Δx/(U + 3c_mp)); 5 flow-throughs warm-up and 5 sampling; about
10 particles per gas cell on average. The drag per unit length is 2F_x/Δz,
the peak heat flux the mean over the facets within 2° of the stagnation line.

```bash
cd verification
python3 make_cylinder_configs.py --cases ma10_kn0.25 ma10_kn0.05 ma25_kn0.25 ma25_kn0.05
vleo_solver --batch batch_cylinder.txt --platform <platform.yaml> --skip-existing
python3 cylinder_eval.py results/cylinder_ma10_kn0p25 results/cylinder_ma10_kn0p05 \
    results/cylinder_ma25_kn0p25 results/cylinder_ma25_kn0p05 --png results/cylinder.png
```

### Results (W7700, commit 0eea90f, 2026-11-26)

| Case | Drag [N/m] | Lofthouse | Δ | Peak heat flux [kW/m²] | Lofthouse | Δ |
| ---- | ---------- | --------- | - | ---------------------- | --------- | - |
| Ma 10, Kn 0.25 | 2.064 ± 0.001 | 2.092 | −1.3 % | 5.78 | 5.926 | −2.4 % |
| Ma 10, Kn 0.05 | 8.792 ± 0.003 | 8.866 | −0.8 % | 15.57 | 15.85 | −1.8 % |
| Ma 25, Kn 0.25 | 13.35 ± 0.004 | 13.19 | +1.2 % | 104.4 | 108.2 | −3.5 % |
| Ma 25, Kn 0.05 | 56.84 ± 0.02 | 56.92 | −0.1 % | 302.2 | 304.5 | −0.8 % |

- Drag agrees within ±1.3 %, the peak heat flux is 0.8–3.5 % low. The
  statistical errors (drag 0.03 %) are negligible; the remaining difference
  is of the size expected between two DSMC codes with different grids, surface
  resolution and averaging of the peak (MONACO gives no uncertainty).
- Stagnation pressure p_s/(ρ∞U∞²): 0.98 / 0.92 (Ma 10, Kn 0.25 / 0.05),
  0.99 / 0.93 (Ma 25), between the continuum Pitot value (≈ 0.88) and the
  free-molecular value with diffuse re-emission (1.16 at Ma 10, 1.11 at Ma 25),
  approaching the continuum value as Kn falls. The shear stress peaks at
  44–48° from the stagnation line.
- Resolution: next to the stagnation point Δx/λ_local is 0.7 (Ma 10,
  Kn 0.25) up to 2.4 (Ma 25, Kn 0.05, min λ_local = 0.20 mm); with about 10
  particles per cell the runs are below the stage-2 recommendation
  (Δx ≤ λ/2, ≥ 20 per cell) in the shock layer of the Kn 0.05 cases. A
  4 mm grid at Ma 10, Kn 0.25 gave 5.90 kW/m², i.e. the coarser grid raises
  the heat flux, consistent with section 4. The good agreement at Ma 25,
  Kn 0.05 is therefore not a converged result; a refined or adapted grid is
  needed before the Kn 0.05 heat flux counts as verified to better than a few
  percent.

Performance (HIP, one W7700, wall time / (mean particles × steps), including
the cell-state output every 5 steps):

| Case | Particles | Steps | Wall time | ns per particle and step | Collisions per particle and step |
| ---- | --------- | ----- | --------- | ------------------------ | -------------------------------- |
| Ma 10, Kn 0.25 | 21.5 M | 26 000 | 1200 s | 2.15 | 0.0035 |
| Ma 10, Kn 0.05 | 34.7 M | 36 000 | 2644 s | 2.11 | 0.010 |
| Ma 25, Kn 0.25 | 21.7 M | 22 000 | 1037 s | 2.17 | 0.0026 |
| Ma 25, Kn 0.05 | 35.2 M | 42 000 | 3022 s | 2.04 | 0.006 |

About 470 M particle steps per second, independent of density and collision
rate: the cost is set by moving and sorting, not by collisions. The sampling
phase costs about 15 % more than the warm-up (cell sampling and state output).
The mean particle count is that of the sampling phase; with the lower count
of the warm-up the index is about 2.2 ns.

## 6. Sphere drag over the Knudsen number (stage 3)

### Set-up

Argon sphere, D = 0.1 m, Ma 4 at low-density wind-tunnel conditions
(stagnation temperature T₀ = 300 K: T∞ = 47.4 K, U∞ = 513 m/s; wall at T₀,
fully diffuse, T_w/T∞ = 6.33), VHS argon from the species database (Bird 1994).
Kn = λ_VHS,∞/D = 10, 3, 1, 0.3, 0.1, 0.03 (Re = ρ∞U∞D/μ∞ = 0.5 … 166, μ the
first-order Chapman–Enskog viscosity of the VHS gas). Quarter model (symmetry
planes y = 0, z = 0 specular, `domain.clip_geometry`), the other four faces are
inflow faces with the free stream; box min(4, 1 + 6λ∞/D)·D upstream and
sideways, 1.5 D downstream; cells D/20, refined to λ_wall/2 (Kn 0.1, D/43) and
0.75 λ_wall (Kn 0.03, D/95, estimated from the pitot pressure); 10 particles per
cell in the free stream. The free-molecular limit is a TPMC run of the full
sphere (UV sphere, 32 040 facets).

```bash
cd verification
python3 make_sphere_configs.py                  # configs, STL (not in the repo), batch_sphere.txt
vleo_solver --batch batch_sphere.txt --platform <platform.yaml> --skip-existing
python3 sphere_eval.py results/sphere_ma4_fm results/sphere_ma4_kn* --png results/sphere.png
```

The Kn 0.03 case (32 M particles) does not fit on one 16 GB GPU because of the
per-particle step tally; it ran with `mpirun -n 2` on two W7700. A 1-rank and a
2-rank run of the same case give bitwise identical results (checked on the
CPU), so the inflow faces and the clipped geometry are decomposition-safe.

References: the exact free-molecular drag of a sphere with diffuse
re-emission (Patterson 1971), and the correlations of Henderson (AIAA J. 14,
1976; supersonic branch) and Loth (AIAA J. 46, 2008; rarefaction branch,
Re ≤ 45) as quoted by Singh et al. (arXiv 2012.04813). The correlations are
fits to ballistic-range and low-density-tunnel measurements with errors of
order 10–15 %. Note: Singh et al. print the term 2/s in Henderson's formula;
the original 2/s² is used here, which gives Henderson's free-molecular limit
2.87 (exact 2.96), whereas 2/s would give 3.27.

### Results (W7700, commit 0eea90f, 2026-11-26)

| Kn | Re | C_D (± SE) | C_D / C_fm | Henderson | Δ | Loth | Δ |
| -- | -- | ---------- | ---------- | --------- | - | ---- | - |
| ∞ (TPMC) | 0 | 2.9608 ± 0.0007 | 0.9997 | 2.874 | +3.0 % | 2.962 (exact) | −0.03 % |
| 10 | 0.50 | 2.920 ± 0.004 | 0.986 | 2.562 | +14.0 % | 2.859 | +2.1 % |
| 3 | 1.66 | 2.817 ± 0.003 | 0.951 | 2.371 | +18.8 % | 2.785 | +1.1 % |
| 1 | 5.0 | 2.583 ± 0.003 | 0.872 | 2.141 | +20.6 % | 2.674 | −3.4 % |
| 0.3 | 16.6 | 2.125 ± 0.002 | 0.717 | 1.852 | +14.7 % | 2.478 | −14.3 % |
| 0.1 | 50 | 1.651 ± 0.001 | 0.557 | 1.595 | +3.5 % | – | – |
| 0.03 | 166 | 1.346 ± 0.0003 | 0.455 | 1.358 | −0.9 % | – | – |

- Free-molecular limit: TPMC agrees with the exact value to 0.03 %.
- Near-free-molecular: the drag reduction is proportional to 1/Kn,
  1 − C_D/C_fm = 0.142/Kn (Kn 10), 0.146/Kn (Kn 3), 0.128/Kn (Kn 1), as the
  first-collision theory predicts for the leading correction.
- Rarefied regime (Kn ≥ 1): within 3.4 % of Loth's correlation, which is built
  on the exact free-molecular limit; Henderson is 14–21 % lower there (his
  free-molecular limit alone is 3 % low).
- Near-continuum (Kn 0.1, 0.03): within 3.5 % and 0.9 % of Henderson.
- Kn 0.3 (Re 17) lies between the two correlations, which differ by 34 %
  there; the transition region is not constrained better than ±15 % by them.
  Measured argon data (Aroesty 1963; AEDC sphere-drag reports) would make this
  point a sharp test.
- Resolution of Kn 0.03: the smallest local λ in resolved cells is 0.78 mm
  (Δx = 1.05 mm, Δx/λ ≈ 1.35 next to the stagnation point; the 0.12 mm in
  the warnings is in small cut cells); 16 % of the cells, mostly in the wake,
  hold fewer than 5 particles. From section 4, a drag error of about 1 % is
  expected from this; the case is not grid-converged.

Performance: 1.7–2.0 ns per particle and step on one W7700 (Kn 10 … 0.1),
1.09 ns on two W7700 for Kn 0.03 (32 M particles), i.e. a parallel efficiency
of about 0.9 for two GPUs.
