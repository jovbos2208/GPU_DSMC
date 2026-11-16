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

Results: section 4 (filled in after the runs).
