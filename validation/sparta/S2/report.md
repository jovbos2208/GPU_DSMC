# S2: free-molecular cube, O, CLL alpha_n = 0.85, sigma_t = 0.65

Acceptance (fixed in docs/sparta.md): |diff| <= 3 sigma_comb + 0.5%

| quantity | vleo_solver | SPARTA | difference | z | limit | result |
| -------- | ----------- | ------ | ---------- | - | ----- | ------ |
| F_x | -5.74604e-05 ± 1.7e-08 | -5.74574e-05 ± 7.7e-09 | -2.97e-09 (+0.01 %) | -0.16 | 3.43e-07 | pass |
| F_y | 2.41545e-06 ± 1.7e-09 | 2.41249e-06 ± 1.6e-09 | 2.95e-09 (+0.12 %) | 1.28 | 1.9e-08 | pass |
| F_z | -3.89052e-06 ± 1.9e-09 | -3.89137e-06 ± 2e-09 | 8.53e-10 (-0.02 %) | 0.30 | 2.79e-08 | pass |
| Q | 0.151473 ± 4.3e-05 | 0.151475 ± 2.8e-05 | -2.12e-06 (-0.00 %) | -0.04 | 0.000913 | pass |

Overall: **PASS**
