# TPMC solver: surface models, tallies, statistics and output (M2)

Conventions (units, normals, signs, axes) are in [conventions.md](conventions.md);
geometry, inflow and transport in [geometry.md](geometry.md).

## 1. Gas-surface interaction (`include/vleo/gsi/Scatter.hpp`)

One header-only implementation is used by TPMC (and later DSMC) on every
backend. The interface follows section 4 of the implementation plan:

```cpp
ScatteringResult scatter_particle(velocity_in, wall_velocity, outward_normal,
                                  particle_mass, internal_energy_in, surface, rng);
```

All scattering happens in the frame of the wall, c = v − v_w, in the local
basis (t1, t2, n), where n points into the gas on the side that was hit
(two-sided surfaces pass the normal facing the particle). With
s = √(k T_w / m):

| Model | Re-emission |
| ----- | ----------- |
| Maxwell, R ≥ f_d (R ~ U[0,1)) | specular: c_r = c_i − 2 (c_i·n) n |
| Maxwell, R < f_d | diffuse: c_r,t1 = s Z1, c_r,t2 = s Z2, c_r,n = s √(−2 ln R1), R1 ~ U(0,1) open |
| CLL | c_r,t = (1 − σ_t) c_i,t + s √(σ_t(2 − σ_t)) (Z1, Z2); c_r,n = √((√(1−α_n) \|c_i,n\| + s√α_n Z3)² + s² α_n Z4²) |

- **CLL parameter convention:** α_n ∈ [0, 1] is the *normal energy*
  accommodation, σ_t ∈ [0, 2] the *tangential momentum* accommodation. The
  tangential energy accommodation is α_t = σ_t (2 − σ_t). Codes that take α_t
  (e.g. SPARTA's `acc_t`) must be given α_t, not σ_t.
- Exact consequences (tested): ⟨c_r,t⟩ = (1 − σ_t) c_i,t,
  Var(c_r,t) = s² σ_t(2 − σ_t) per component, ⟨c_r,n²⟩ = (1 − α_n) c_i,n² + 2 α_n s².
- Limits: CLL(0, 0) is specular, CLL(1, 1) is fully diffuse (same distribution
  as Maxwell f_d = 1).
- **Outward guarantee (M2.4):** the diffuse normal component uses an open
  uniform variate (U(0,1), never 0 or 1), the CLL normal component is redrawn
  in the probability-zero case c_n = 0, so c_r·n > 0 always. A non-positive
  normal component would end the history as `Invalid` (counted; never observed).
- **Wall velocity:** the result is v_r = c_r + v_w. The current configuration
  has rigid bodies at rest in the body frame (v_w = 0); the kernel and the
  energy balance are written for a general tangential v_w (tested).
- Random numbers: interaction k of history h uses the stream
  (seed, h, k + 1, SurfaceInteraction). Maxwell draws R first, then (diffuse)
  Z1, Z2 and R1; CLL draws Z1 … Z4.
- **Internal energy (M4.8):** molecules (N2, O2) carry rotational and
  vibrational energy. Maxwell specular keeps both, Maxwell diffuse resamples
  both at T_w. CLL: rotation with x' = (√((1−α_rot) x) + √(α_rot/2) Z5)² +
  (α_rot/2) Z6², x = e_rot/(kT_w), so ⟨e_rot'⟩ = (1−α_rot) e_rot + α_rot kT_w
  (α_rot = 1 is diffuse); vibration resampled at T_w with probability α_vib.
  `cll.alpha_rot` and `cll.alpha_vib` default to `alpha_n`. The energy balance
  is ΔE = ΔE_trans + (e_rot,i + e_vib,i) − (e_rot,r + e_vib,r).
  Verified: N2 flat plate (diffuse) vs. the analytic flux including ⟨e_rot⟩ =
  kT and the equilibrium vibrational energy (0.6624 ± 0.0003 W vs 0.6623 W);
  N2/O2 equilibrium with Maxwell and CLL walls has zero net heat flux.

Surface properties are stored per material id (all materials currently share
the `surface` block of the configuration).

## 2. Event balance (M2.5)

For every processed interaction e of a particle with mass m_e at r_e:

| Quantity | Definition |
| -------- | ---------- |
| momentum to the body | Δp_e = m_e (v_i − v_r) |
| angular momentum about r_0 | Δl_e = (r_e − r_0) × Δp_e, r_0 = `reference.moment_center_m` |
| energy to the wall | ΔE_e = ½ m_e (\|v_i − v_w\|² − \|v_r − v_w\|²) |

ΔE_e ≥ 0 and ΔE_e < 0 are summed separately (`heat_rate_positive_W`,
`heat_rate_negative_W`). Positive Q̇ = heat from the gas into the surface.
Truncated histories (see geometry.md §6) contribute the interactions that were
processed; their count is reported as a warning.

## 3. TPMC event weight (M2.6)

The six faces of the domain emit free-stream molecules at the total rate
R = Σ_faces Σ_species Γ A [1/s] (geometry.md §5). Every history is an
independent sample of one such molecule, so N histories represent the rate
γ = R / N each, and

  F = Σ_e γ Δp_e = R · mean_h(ΔP_h),  M = R · mean_h(ΔL_h),  Q̇ = R · mean_h(ΔE_h),

where ΔP_h = Σ_{e∈h} Δp_e is the total momentum transferred by history h.
N is the number of histories **actually run**, which keeps the estimate
unbiased when the run stops early on a convergence criterion. Histories that
never hit the body contribute zero; they are part of the sample.

## 4. Tallies (M2.7, M2.9)

- Per history, ΔP, ΔL, ΔE, ΔE⁺ and ΔE⁻ are accumulated in registers along the
  trajectory and written once to the history tally array (no atomics).
- Per batch, two deterministic device reductions compute the mean and the sum
  of squared deviations of these quantities and of the projections
  ΔP·e_D, ΔP·e_L, ΔP·e_Y. Batches are merged with the pairwise update of
  Chan, Golub and LeVeque (numerically stable when most histories are zero).
- With `output.save_surface_fields: true` the **same event values** are added
  atomically to per-triangle arrays (force, heat, heat⁺, heat⁻, hits) and
  scaled by γ at the end. The sum over all triangles equals the global result
  to rounding (tested to 1e-10). Per-triangle values may differ in the last
  bits between runs on parallel backends (atomic summation order); the
  global results do not.

## 5. Coefficients and statistics (M2.8, M2.10, M2.11)

- Aerodynamic axes e_D, e_L, e_Y come from the validated configuration
  (conventions.md §4). With q∞ = ½ ρ∞ V∞²:
  C_D = F·e_D / (q∞ A_ref), C_L = F·e_L / (q∞ A_ref), C_Y = F·e_Y / (q∞ A_ref),
  C_M = M / (q∞ A_ref L_ref).
- **Standard error:** the histories are i.i.d., so SE(Y) = |c| R √(s²/N)
  with the sample variance s² of the per-history quantity and the scale c
  (1, 1/(q A) or 1/(q A L)). Coefficients get their SE from the variance of the
  projected per-history momentum (no error propagation approximations).
- **Confidence interval:** value ± z SE with z = Φ⁻¹((1 + level)/2)
  (`statistics.confidence_level`, normal approximation, N is large).
- **Batches:** `tpmc.batch_size` histories per kernel launch; the result does
  not depend on the batch size except for rounding (histories are identified
  by a global index).
- **Convergence stop (optional):**

  ```yaml
  tpmc:
    particle_histories: 100000000   # upper limit
    convergence:
      quantity: C_D                 # C_D, C_L, C_Y, C_Mx, C_My, C_Mz, heat_rate_W
      relative_error: 0.002         # or absolute_error (exactly one)
      min_histories: 1000000
  ```

  After each batch the run stops once SE ≤ target (relative: SE ≤ ε |value|)
  and at least `min_histories` were run. `statistics.converged` reports the
  outcome; an unreached target is a warning.

## 6. Free-molecular flat plate (reference for M2.18)

For one side of a plate with gas-side normal n, species with number density
n_s and mass m, c = √(2kT/m), speed ratio toward the wall s = u·(−n)/c,
tangential bulk velocity u_t, A = (√π/2)(1 + erf s), B = ½ e^{−s²}:

| Flux (per area) | Expression |
| --------------- | ---------- |
| number | Γ = n_s c / (2√π) [e^{−s²} + √π s (1 + erf s)] |
| normal momentum | P_i = n_s m c² / √π · J₂, J₂ = s B + ½ A (1 + 2s²) |
| tangential momentum | τ_i = m Γ u_t |
| energy | E_i = ½ m [n_s c³ / √π · J₃ + Γ (\|u_t\|² + c²)], J₃ = B (s² + 1) + A s (s² + 3/2) |

Force on the side (momentum to the body) and heat:

- diffuse at T_w: F = −(P_i + m Γ s_w √(π/2)) n + τ_i, Q = E_i − 2 k T_w Γ (s_w = √(kT_w/m))
- specular: F = −2 P_i n, Q = 0; Maxwell f_d mixes both linearly
- CLL: tangential force σ_t τ_i;
  Q = E_i − ½ m [(1−σ_t)² Γ (\|u_t\|² + c²) + 2 s_w² σ_t(2−σ_t) Γ + (1−α_n) n_s c³ J₃/√π + 2 α_n s_w² Γ]
  (the mean normal re-emission speed of CLL has no closed form, so the normal
  force is not compared).

Both sides are summed. The formulas hold exactly for every point of a single
plate (no shadowing), so TPMC must reproduce them within statistics.

## 7. Verification and tolerances

The tolerances were fixed before the tests were run:

| Kind of check | Tolerance |
| ------------- | --------- |
| deterministic identities (specular limits, CLL(0,0) ≡ Maxwell(0), Σ surface fields = total) | 1e-12 … 1e-14 relative (rounding) |
| simulation vs. analytic value | \|x − x_ref\| ≤ 5 SE |
| two independent simulations | \|a − b\| ≤ 5 √(SE_a² + SE_b²) |
| distribution of samples | Kolmogorov–Smirnov √n D < 1.95 (p ≈ 0.1 %), moments within 5 SE |

| Plan item | Test (`tests/unit/`) |
| --------- | -------------------- |
| M2.13 Maxwell specular limit, Q = 0 | `test_gsi`: exact reflection, no energy change; `test_tpmc`: satellite with f_d = 0 has Q̇ = 0 to 1e-12, CLL(0,0) gives identical forces |
| M2.14 diffuse Maxwell | `test_gsi`: KS tests of t1, t2 (normal) and n (Rayleigh), mean energy 2kT_w, independence of the incident velocity, mixture fraction |
| M2.15 CLL limits | `test_gsi`: CLL(0,0) specular, CLL(1,1) diffuse (KS), general moments for 4 parameter pairs; `test_tpmc`: CLL(1,1) vs Maxwell(1) on the satellite |
| M2.16 thermal equilibrium | `test_tpmc`: gas and wall at 400 K, concave body, Maxwell f_d ∈ {1, 0.4} and CLL(0.5, 0.8): \|Q̇\| ≤ 5 SE with SE < 2 % of the absorbed energy |
| M2.17 symmetric body | `test_tpmc`: satellite in axial flow: F_y, F_z, C_L, C_Y, C_M vanish within 5 SE |
| M2.18 flat plate | `test_tpmc`: Maxwell f_d ∈ {1, 0, 0.6} at 30°, 60°, 90° (all force components and Q̇), CLL (tangential force, Q̇) for three parameter pairs; O/He mixture |

With 4·10⁸ histories (Maxwell f_d = 1, 30°) the plate force and heat agree
with the analytic values to 0.7–2.7·10⁻⁵ relative (standard error 5.7·10⁻⁵),
i.e. there is no detectable bias above the 10⁻⁵ level.

Debug builds run a quarter of the histories (`VLEO_TEST_SCALE`); the 5-SE
criteria are valid for any sample size.

## 8. Output (M2.12)

| File | When | Content |
| ---- | ---- | ------- |
| `results.json` | always | schema `vleo-results/1`: `simulation`, `reference` (A_ref, L_ref, r_0, q∞, ρ∞, V∞, axes), `results` (force_N, moment_Nm, C_D, C_L, C_Y, C_M, heat_rate_W, heat_rate_positive/negative_W, drag/lift/side_force_N), `std_error` and `confidence_interval.low/high` with the same keys, `statistics` (confidence level, z, histories, batches, converged, convergence target and achieved error), `diagnostics` (inflow rate, γ, particle balance, interactions, wall time, histories/s), `provenance` (version, git commit, build, Kokkos, execution space, time), `warnings` |
| `results.csv` | `output.format: csv` | quantity, unit, value, std_error, ci_low, ci_high |
| `surface_fields.csv` | `output.save_surface_fields: true` | per triangle (BVH order with `source_index` into the STL): material, centroid, normal, area, force, pressure, shear, heat rate and flux, impingement rate |
| `run_metadata.json`, `resolved_config.yaml`, `input_config.yaml`, `kokkos_configuration.txt` | always | see configuration.md |

HDF5 is not part of this build; `output.format: hdf5` is rejected at start-up.

Command line: `vleo_solver -c <config> [-p <platform>] [-o <dir>] [-n <histories>] [-q]`.
`-n` overrides `tpmc.particle_histories` and is recorded in `resolved_config.yaml`.

## 9. Cross-backend agreement

`scripts/compare_results.py A.json B.json` compares all main quantities of two
runs with identical configuration and prints, per quantity, the relative
difference and the z-score; it accepts numerical equality (≤ 1e-9 relative)
or statistical equality (z ≤ 5). Because every history is a pure function of
(seed, history id), different backends follow the same histories; results
differ only through floating-point details of the device math libraries and
reduction order. See building.md for the recorded comparisons.
