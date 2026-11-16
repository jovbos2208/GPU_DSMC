# GOCE / CHAMP vs. SPARTA (M6.10, M6.11)

Driver: `run_sat.py`; method, acceptance and results: `docs/satellites.md`.
Run on 2026-11-16 on the server `hagen` (`m6-mpi` at `de78517`; free-molecular
cases at `66cc7d3`, identical driver logic): vleo_solver on one Radeon PRO
W7700, SPARTA 27Aug2026 (git 95b9aba, `spa_mpi`, 24 ranks).

Per case: `report.md` and `comparison.json`, and per attitude `a<alpha>_b<beta>/`
with the inputs of both codes (`vleo*.yaml`, `in.case`, `case.species`,
`case.vss`), their logs (`vleo*.log`, `log.sparta`, `sparta.stdout`) and the
vleo_solver outputs (`vleo_out*/`). DSMC cases also contain the runs with the
free-stream cell-size rule and with half the pilot cell size.

Not archived (large, regenerable): SPARTA's per-block surface dumps
(`dump.surf`), the SPARTA surface files (`body.surf`, written from
`examples/geometry/*_march2019.stl` by `run_sat.py`), and vleo_solver's
per-triangle `surface_fields.csv`. Rerun the driver to get them.
