# GOCE and CHAMP (M6.8 – M6.11)

## 1. Geometries (M6.9)

| File | Source | Units | Triangles | Extent x × y × z [m] | Wetted area | Frontal area A_x | Closed |
| ---- | ------ | ----- | --------- | -------------------- | ----------- | ---------------- | ------ |
| `examples/geometry/goce_march2019.stl` | March et al. 2019, CATIA export (`GOCE_sag50.stl`) | mm | 2480 | 5.31 × 2.10 × 2.44 | 34.60 m² | 1.027 m² | yes, 1 component |
| `examples/geometry/champ_march2019.stl` | March et al. 2019, CATIA export (`champ500step850.stl`) | mm | 2224 | 8.33 × 1.68 × 1.39 | 18.11 m² | 0.770 m² | yes, 3 components (body + two 19 cm² parts) |

Checks with the solver's mesh report: no degenerate triangles, no
boundary, non-manifold or inconsistently oriented edges, and stored normals
agree with the vertex order. Only the line endings were converted
(CRLF → LF); the vertices are identical to the supplied files.
A Blender export (`GOCE.stl`, unknown unit, scale ≈ 1/36.9 of the CATIA
model) was not used: its TPMC run had histories truncated at 100
reflections, which points to internal cavities.

**Body axes:** +x is the flight direction (CHAMP: boom at +x), z is the lift
axis (`reference.lift_axis_body: [0, 0, 1]`). The frontal areas are
projections onto the y-z plane (2 mm raster) and agree with the usual
cross-sections (GOCE ≈ 1.1 m², CHAMP ≈ 0.75 m²).

**Reference quantities:** A_ref = 1 m², L_ref = 1 m, moment centre = CAD
origin (0, 0, 0). Coefficients relative to the frontal area follow from
C · A_ref / A_x.

## 2. Attitude sweep (M6.8)

`vleo_solver --attitude <alpha> <beta>` sets the flight direction in the
body frame to

    d = (cos α cos β, sin β, sin α cos β),   gas.bulk_velocity_body_mps = −|V| d,

keeping the speed of the configuration (α = β = 0: flight along +x). The
aerodynamic axes are recomputed; a lift axis parallel to the flow is
rejected. `scripts/attitude_sweep.py` runs a list of attitudes and
writes `sweep.json`, `sweep.csv` and a Markdown table:

```bash
python3 scripts/attitude_sweep.py --solver build/<preset>/vleo_solver --config sim.yaml \
    --out results/sweep --alpha -10 0 10 20 --beta 0 10 [-- --histories 5000000]
```

## 3. Comparison with SPARTA (M6.10, M6.11)

No published coefficient tables are available. The reference is therefore
SPARTA (27Aug2026, git 95b9aba) with identical inputs:
`validation/satellites/run_sat.py`. Conditions: V = 7700 m/s, T = 900 K,
T_wall = 300 K. Domain: body bounding box + 0.3 m (free-molecular) or
+ 1.0 m (DSMC), the same in both codes.

| Case | Satellite | Flow | Gas | Wall | Attitudes (α, β) [deg] |
| ---- | --------- | ---- | --- | ---- | ---------------------- |
| G-FM-D | GOCE | free-molecular (TPMC vs. SPARTA), n = 10¹² m⁻³ (λ∞ ≈ 750 km) | O 0.7, N2 0.3 | diffuse | (0,0) (10,0) (30,0) (0,15) |
| G-FM-C | GOCE | free-molecular | O | CLL α_n = 0.85, σ_t = 0.65 | (0,0) (10,0) |
| C-FM-D | CHAMP | free-molecular | O 0.7, N2 0.3 | diffuse | (0,0) (10,0) (30,0) (0,15) |
| C-FM-C | CHAMP | free-molecular | O | CLL | (0,0) (10,0) |
| G-DSMC | GOCE | DSMC, n = 3·10¹⁷ m⁻³ (λ∞ ≈ 8 m, Kn ≈ 1.5) | O | diffuse | (0,0) |
| C-DSMC | CHAMP | DSMC, n = 3·10¹⁷ m⁻³ | O | diffuse | (0,0) |

The DSMC density was first 3·10¹⁸ m⁻³. There the pilot runs resolved the
dense layer at the wall (local λ of a few cm), and the run needed ~200 M
particles: more than the 16 GB of one W7700 (allocation failure). With
3·10¹⁷ m⁻³ the cells are ~10× larger.

**Free-molecular SPARTA runs need a `collide` command.** Without one,
SPARTA gives molecules no internal energy (emission and re-emission). The
first G-FM-D run (n = 10¹⁵ m⁻³, no `collide`) therefore had a heat rate
0.19–0.29 % below vleo_solver (z = 4–6, within the 1 % limit). That matches
the missing rotational energy of N2 (Γ_N2 k (900 − 300 K)). With O alone
(CLL) Q agreed to 0.01–0.03 %. All free-molecular runs therefore use
`collide vss` at n = 10¹² m⁻³, where collisions are negligible.

**Numerics.** Free-molecular SPARTA runs use 0.2 m cells (irrelevant
without collisions), a time step dx / (4 (V + 3 c_mp)), about 10⁶
particles, 2 blocks of warm-up and 20 blocks of at least 2 flow-through
times each. vleo_solver uses 2·10⁷ TPMC histories. DSMC: vleo_solver chooses
the cell size automatically with pilot runs (M5). SPARTA then uses the
same grid, time step and particle weight; its warm-up is that of vleo_solver
rounded up to whole blocks. CLL in SPARTA: acc_t = σ_t (2 − σ_t), the
internal-energy coefficients equal α_n (docs/sparta.md).

**Compared quantities.** F_x, F_y, F_z and the heat rate Q, with the
standard error of 20 block means (SPARTA: `fix ave/surf` per block). Moments
about the CAD origin are compared as Σ c_t × F_t over the triangle
centroids c_t in both codes. The centroid approximation itself deviates
from the exact moment by up to ~7 % on GOCE (large triangles with a
non-uniform load), so the exact moment of vleo_solver is reported only for
information. Its standard error is used as σ for both centroid moments.

**Acceptance (fixed before the runs):**

    free-molecular:  |a − b| ≤ 3 σ_comb + 1 % · max(|a|, |b|)
    DSMC:            |a − b| ≤ 3 σ_comb + 3 % · max(|a|, |b|)

with σ_comb = √(σ_a² + σ_b²). The systematic part covers differences in
the time-step-based (SPARTA) versus event-based (TPMC) treatment of
multiple reflections, and for DSMC the collision and grid details already
quantified in M4 (S3/S4 within 0.27 %).

**Cell-size rule on a real geometry (M6.11).** For every DSMC case
vleo_solver also runs with the free-stream cell-size rule (no pilot) and
with half the pilot cell size. The pilot rule is adequate if C_D and Q at
half the cell size agree with the pilot grid within 3 σ + 1 %.

## 4. Results (2026-11-16, server `hagen`, `m6-mpi` at `de78517`)

vleo_solver on one Radeon PRO W7700 (`hip-gfx1101-mpi`). SPARTA on 24 CPU
ranks (`spa_mpi`; the other cores were in use by other users, which does not
affect the results). All 14 comparisons pass. Reports, inputs and logs:
`validation/satellites/<case>/` on the server.

**Free-molecular (M6.10), 12 attitudes.** Deviation vleo_solver − SPARTA of
the quantities that are not zero by symmetry:

| Case | (α, β) | F_x | F_z (lift) / F_y (β = 15°) | Q | M (largest component) | max \|z\| |
| ---- | ------ | --- | -------------------------- | - | --------------------- | ---------- |
| G-FM-D | (0,0) (10,0) (30,0) (0,15) | +0.03, +0.06, −0.07, −0.01 % | F_z +0.12, −0.06 %; F_y 0.00 % | +0.02, +0.06, −0.06, −0.02 % | M_y +0.33, −0.15 %; M_z −0.15 % | 2.07 |
| G-FM-C | (0,0) (10,0) | −0.01, +0.03 % | F_z +0.15 % | +0.03, +0.06 % | M_y +0.40 % | 2.33 |
| C-FM-D | (0,0) (10,0) (30,0) (0,15) | +0.03, +0.02, +0.05, −0.02 % | F_z +0.20, +0.03, +0.05, −0.39 %; F_y −0.01 % | +0.02, +0.02, +0.05, −0.02 % | M_y +0.53, +0.43, −0.01, +0.73 % | 1.54 |
| C-FM-C | (0,0) (10,0) | +0.01, +0.06 % | F_z −0.04, +0.06 % | +0.01, +0.05 % | M_y −0.07, +0.62 % | 1.80 |

Large relative differences of components that vanish by symmetry (e.g.
F_y at β = 0) are noise around zero, all with |z| < 2.4. Coefficients
(A_ref = 1 m²) at α = β = 0: GOCE C_D = 3.750 (diffuse), 3.909 (CLL);
CHAMP C_D = 2.338 (diffuse), 2.061 (CLL), C_L = 0.054 / 0.223.

**DSMC (M6.11), n = 3·10¹⁷ m⁻³, α = β = 0:**

| Case | C_D | F_x | F_z | Q | M_y (centroids) | max \|z\| |
| ---- | --- | --- | --- | - | --------------- | ---------- |
| G-DSMC | 4.027 ± 0.015 | +0.41 % | (≈ 0) | +0.30 % | (≈ 0) | 1.80 |
| C-DSMC | 2.477 ± 0.015 | +0.17 % | −0.16 % | +0.19 % | −7.5 % (z = 1.13) | 1.21 |

**Cell-size rule on the real geometries (M6.11).** The pilot runs kept the
free-stream rule (L_max / 16): its cells were already within 1.5 λ_min.

| Case | grid | cell [m] | C_D | Q [W] |
| ---- | ---- | -------- | --- | ----- |
| G-DSMC | free-stream rule = pilot | 0.457 × 0.456 × 0.444 | 4.0271 ± 0.015 | 3610 ± 14 |
| | half cell | 0.229 × 0.228 × 0.222 | 3.9755 ± 0.0075 (−1.3 %, z = 3.1) | 3565 ± 6.6 (−1.25 %, z = 2.9) |
| C-DSMC | free-stream rule = pilot | 0.646 × 0.613 × 0.565 | 2.4769 ± 0.015 | 2248 ± 14 |
| | half cell | 0.323 × 0.307 × 0.308 | 2.4515 ± 0.0093 (−1.0 %, z = 1.5) | 2225 ± 8.4 (−1.0 %, z = 1.4) |

The pre-defined criterion (3 σ + 1 %) is met in both cases. The trend is
systematic, though: C_D and Q decrease by 1.0–1.3 % at half the cell size,
for both satellites. On these geometries the pilot rule
(`lambda_fraction = 1.5`) therefore leaves a cell-size error of about 1 %,
larger than the 0.24 % of the M3.17 study on the cube. For ≤ 0.5 %
accuracy, use a smaller `lambda_fraction` or check convergence with
half the cell size (8× the cells). The comparison with SPARTA is not
affected: both codes used the same grid.

**Gate M6, item 2** ("GOCE/CHAMP agree with the references within defined
tolerances"): met against SPARTA as the reference. Published coefficient
tables were not available.
