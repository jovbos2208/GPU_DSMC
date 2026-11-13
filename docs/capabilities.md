# Capability matrix (Gate M4)

Status of every supported combination and the test that validates it.
✔ = implemented and validated, ◐ = implemented, validation pending, – = not supported.

## Solvers and physics

| Feature | TPMC | DSMC | Validation |
| ------- | ---- | ---- | ---------- |
| Free-molecular flow | ✔ | ✔ (n → 0) | `test_tpmc` flat plate vs. theory; `test_dsmc` DSMC ≡ TPMC; dilution study (dsmc.md §8) |
| Monatomic gases O, N, He, Ar | ✔ | ✔ | `test_tpmc`, `test_dsmc`, `test_mixture` |
| Gas mixtures (≤ 8 species) | ✔ | ✔ | `test_mixture` (pair collision rates, injection shares, relaxation) |
| Molecules N2, O2: rotation | ✔ | ✔ | `test_internal` (equipartition, relaxation rate, wall accommodation, heat balance) |
| Molecules N2, O2: vibration (SHO, quantised) | ✔ | ✔ | `test_internal` (equilibrium levels, relaxation) |
| VHS collisions | – | ✔ | `test_vhs`, `test_dsmc` (collision rate, relaxation) |
| VSS collisions | – | ✔ | `test_internal` (deflection distribution); transport properties pending (SPARTA, M4.10) |
| Chemistry, ionisation, radiation | – | – | not in scope |

## Gas-surface interaction

| Model | Translation | Rotation | Vibration | Validation |
| ----- | ----------- | -------- | --------- | ---------- |
| Maxwell specular (f_d = 0) | ✔ | unchanged ✔ | unchanged ✔ | `test_gsi`, `test_tpmc`, `test_internal` |
| Maxwell diffuse / mixed | ✔ | full accommodation ✔ | full accommodation ✔ | `test_gsi` (KS), flat plate incl. internal energy |
| CLL (α_n, σ_t) | ✔ | α_rot ✔ | α_vib ✔ | `test_gsi` moments/limits, `test_internal`, equilibrium Q̇ = 0 |
| Wall velocity (tangential) | ✔ | – | – | `test_gsi` |

## Geometry and numerics

| Feature | Status | Validation |
| ------- | ------ | ---------- |
| Closed triangulated bodies, several components | ✔ | `test_mesh`, `test_ray`, `test_transport` |
| Open two-sided surfaces (`require_closed: false`) | ✔ | flat plate tests |
| Cut DSMC cells | ✔ | `test_grid` (exact / shell bound) |
| Automatic DSMC cell size | ◐ | rule based on λ∞; for blunt bodies use a cell study (dsmc.md §8, open for M5) |

## Backends

| Backend | Status |
| ------- | ------ |
| CPU Serial / OpenMP | ✔ all tests |
| HIP gfx1100 (Radeon 740M), gfx1101 (Radeon PRO W7700) | ✔ all tests |
| HIP gfx942/gfx90a (MI300/MI250) | ◐ presets available, not run |
| CUDA | ◐ preset available, not run (no hardware) |

## Comparison with SPARTA

◐ Pending (M4.10–M4.14): reference cases and automated comparison.
