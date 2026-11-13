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

## Status

The cases are defined; running them requires a SPARTA build (`make serial`
in SPARTA's `src/`), see `validation/sparta/README.md`.
