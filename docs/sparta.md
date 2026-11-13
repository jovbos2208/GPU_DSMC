# Comparison with SPARTA (M4.10 – M4.14)

SPARTA (https://sparta.github.io, release 27Aug2026) is the independent
reference solver. It is not a dependency; it is built separately and only used
by the validation cases in `validation/sparta/`.

## Parameter translation (M4.11)

Checked against `src/surf_collide_cll.cpp` of the SPARTA release:

| This solver | SPARTA `surf_collide cll Tsurf acc_n acc_t acc_rot acc_vib` | Note |
| ----------- | ------------------------------------------------------------ | ---- |
| `alpha_n` | `acc_n = alpha_n` | identical Rician normal kernel |
| `sigma_t` (tangential **momentum** accommodation, [0, 2]) | `acc_t = sigma_t (2 − sigma_t)` (tangential **energy** accommodation, [0, 1]) | SPARTA's mean tangential velocity is √(1−acc_t) c_t = (1−σ_t) c_t, which holds for σ_t ≤ 1 only; σ_t > 1 (back-scattering) has no SPARTA equivalent |
| `alpha_rot` | `acc_rot = alpha_rot` | identical Rician form for 2 rotational DOF |
| `alpha_vib` | `acc_vib = alpha_vib` | **equal only for 0 and 1**: SPARTA uses Lord's continuous form followed by quantisation, this solver resamples the level at T_w with probability α_vib |
| Maxwell `diffuse_fraction` f_d | `surf_collide diffuse Tsurf f_d` | identical |

Example: σ_t = 0.65 ⇒ `acc_t = 0.8775`, never `acc_t = 0.65`.

Collision data: the cases write the species masses and VHS/VSS parameters of
this solver ([species.md](species.md)) into the SPARTA `.species`/`.vss`
files, so both codes use identical data. Rotational relaxation: SPARTA's
`RotRel` is the per-collision relaxation probability, the same quantity as
1/Z_rot here.

## Cases and acceptance limits (M4.10, M4.13)

Fixed before any comparison was run (2026-11-13). Each case compares the
force components F_x, F_y, F_z and the heat rate Q̇ on the body.

| Case | Flow | Gas | Wall | Solvers compared | Acceptance |
| ---- | ---- | --- | ---- | ---------------- | ---------- |
| S1 | free-molecular, 0.5 m cube, V = 7.5 km/s, T = 900 K | O | Maxwell diffuse, T_w = 300 K | TPMC vs SPARTA (`collide none`) | \|Δ\| ≤ 3 σ_comb + 0.5 % |
| S2 | as S1 | O | CLL α_n = 0.85, σ_t = 0.65 (acc_t = 0.8775) | TPMC vs SPARTA | \|Δ\| ≤ 3 σ_comb + 0.5 % |
| S3 | transitional, 0.5 m cube, n = 1e19, Kn ≈ 1 | Ar, VHS | Maxwell diffuse | DSMC vs SPARTA (`collide vss` with α = 1) | \|Δ\| ≤ 3 σ_comb + 2 % |
| S4 | as S3 | N2, VHS, rotation (Z_rot = 5), vibration off | Maxwell diffuse | DSMC vs SPARTA | \|Δ\| ≤ 3 σ_comb + 3 % |

σ_comb = √(SE_a² + SE_b²). The systematic allowances cover the different
numerical schemes of the two codes (grid, time stepping, sampling): 0.5 % for
the deterministic-geometry free-molecular cases, 2 % (3 % with rotation) for
DSMC at Kn ≈ 1, where each code uses its own converged cell size and time
step (Δx ≤ λ∞/10, Δt ≤ τ∞/10). A case fails if any of the four quantities
violates its limit; failures are reported, not tuned away.

## Results (M4.12, M4.14)

Run 2026-11-13 with `validation/sparta/run_cases.py` (cpu-release, SPARTA
27Aug2026 `spa_serial`); inputs, logs, results and per-case reports are
archived in `validation/sparta/S1` … `S4`. All four cases **pass**:

| Case | F_x | F_y | F_z | Q̇ | Result |
| ---- | --- | --- | --- | -- | ------ |
| S1 free-molecular, O, Maxwell diffuse | +0.01 % (−0.2 σ) | +0.12 % (1.5 σ) | −0.03 % (0.5 σ) | −0.00 % | pass |
| S2 free-molecular, O, CLL (acc_t = 0.8775) | +0.01 % | +0.12 % (1.3 σ) | −0.02 % | −0.00 % | pass |
| S3 transitional, Ar, VHS | +0.11 % (−2.9 σ) | +0.12 % | +0.10 % | +0.07 % (1.9 σ) | pass |
| S4 transitional, N2 + rotation | +0.14 % (−4.2 σ) | −0.21 % | +0.27 % (−2.5 σ) | +0.09 % (2.6 σ) | pass |

(Differences vleo_solver − SPARTA relative to SPARTA; σ = combined standard error.)

Interpretation: in the free-molecular cases both codes agree within their
statistics (≤ 0.12 %). In the DSMC cases the differences are 0.1–0.3 %,
i.e. well inside the 2 % / 3 % allowance but statistically resolved
(up to 4 σ): the two codes differ in implementation details (initial
(σg)_max and its updates, cell-based vs. trajectory-based handling of the
moving particles, Larsen-Borgnakke variant for rotation), which produce
systematic differences of this size at Kn ≈ 1. They are an order of magnitude
below the discretisation error found in the cell-size study (dsmc.md §8).
