# SPARTA validation cases (M4.10 – M4.14)

Case definitions and acceptance limits (fixed before any run): `docs/sparta.md`.

`run_cases.py` writes identical inputs for both codes (geometry, species data
from this solver's database, cell size, time step, particle weight), runs
`vleo_solver` and SPARTA, evaluates both with block statistics, applies the
acceptance limits and archives, per case, in `S1/` … `S4/`: inputs
(`vleo.yaml`, `in.case`, `case.species`, `case.vss`, `cube.stl`, `cube.surf`),
logs, `vleo_out/`, `log.sparta`, `comparison.json` (incl. versions) and
`report.md`.

```bash
# SPARTA (serial build, release 27Aug2026)
cd /path/to/sparta/src && make -j10 serial
# all cases (~30 min on a laptop CPU)
python3 validation/sparta/run_cases.py --vleo build/cpu-release/vleo_solver \
    --sparta /path/to/sparta/src/spa_serial
```
