# S1: free-molecular cube, O, Maxwell diffuse

Acceptance (fixed in docs/sparta.md): |diff| <= 3 sigma_comb + 0.5%

| quantity | vleo_solver | SPARTA | difference | z | limit | result |
| -------- | ----------- | ------ | ---------- | - | ----- | ------ |
| F_x | -4.79479e-05 ± 1.3e-08 | -4.79445e-05 ± 7.5e-09 | -3.37e-09 (+0.01 %) | -0.22 | 2.86e-07 | pass |
| F_y | 3.3432e-06 ± 2e-09 | 3.33924e-06 ± 1.8e-09 | 3.97e-09 (+0.12 %) | 1.45 | 2.49e-08 | pass |
| F_z | -5.31555e-06 ± 2.3e-09 | -5.31736e-06 ± 2.7e-09 | 1.81e-09 (-0.03 %) | 0.51 | 3.72e-08 | pass |
| Q | 0.177119 ± 5.1e-05 | 0.17712 ± 3.2e-05 | -1.7e-06 (-0.00 %) | -0.03 | 0.00107 | pass |

Overall: **PASS**
