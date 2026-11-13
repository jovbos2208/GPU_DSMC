# S4: transitional cube (Kn ~ 1), N2 with rotation (Z_rot = 5), VHS, Maxwell diffuse

Acceptance (fixed in docs/sparta.md): |diff| <= 3 sigma_comb + 3.0%

| quantity | vleo_solver | SPARTA | difference | z | limit | result |
| -------- | ----------- | ------ | ---------- | - | ----- | ------ |
| F_x | -2.54319 ± 0.00068 | -2.53966 ± 0.00052 | -0.00353 (+0.14 %) | -4.15 | 0.0789 | pass |
| F_y | 0.156457 ± 0.00024 | 0.15679 ± 0.00027 | -0.000334 (-0.21 %) | -0.93 | 0.00578 | pass |
| F_z | -0.251318 ± 0.0002 | -0.25065 ± 0.00018 | -0.000668 (+0.27 %) | -2.46 | 0.00836 | pass |
| Q | 8601.93 ± 2.2 | 8594.43 ± 2 | 7.5 (+0.09 %) | 2.56 | 267 | pass |

Overall: **PASS**
