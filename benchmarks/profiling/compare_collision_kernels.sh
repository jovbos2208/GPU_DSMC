#!/usr/bin/env bash
# Collision kernel comparison on one GPU: per_cell vs. waves with 8/16/32/64 threads per cell.
#
#   DEVICE=1 ./compare_collision_kernels.sh [case.yaml ...]     (default: p3_reservoir_air.yaml)
#
# Each case runs shortened (100 warm-up + 200 sampled steps, no cell-state output) with the
# fenced phase profiler (VLEO_PROFILE=1) once per kernel variant. Printed per variant:
# dsmc.collide per step, the whole step, and the collision counters, which must be identical
# for all variants (same random streams, same dependency order).
# Results: results/collide/<case>/<variant>/  and  results/collide/summary.txt
set -euo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DSMC=${DSMC:-$(cd "$HERE/../.." && pwd)}
BIN=${BIN:-$DSMC/build/hip-gfx1101-hdf5/vleo_solver}
DEVICE=${DEVICE:-0}
BACKEND=${BACKEND:-hip}
WARMUP=${WARMUP:-100}
SAMPLE=${SAMPLE:-200}
VARIANTS=${VARIANTS:-"per_cell hybrid waves:16 waves:32"}
[[ -x "$BIN" ]] || { echo "error: solver binary not found: $BIN (set BIN=...)"; exit 1; }
CASES=("$@")
[[ ${#CASES[@]} -gt 0 ]] || CASES=(p3_reservoir_air.yaml)
ROOT="$HERE/results/collide"
mkdir -p "$ROOT"
cat > "$ROOT/platform.yaml" <<EOF
platform:
  backend: $BACKEND
  device_id: $DEVICE
EOF

for CASE in "${CASES[@]}"; do
  name=$(basename "$CASE" .yaml)
  for v in $VARIANTS; do
    kernel=${v%%:*}
    team=""; [[ $v == *:* ]] && team=${v##*:}
    tag=${kernel}${team:+_$team}
    d="$ROOT/$name/$tag"
    mkdir -p "$d"
    # absolute geometry path, shortened run, no cell-state output, kernel choice
    python3 - "$CASE" "$d/case.yaml" "$kernel" "$team" "$WARMUP" "$SAMPLE" <<'EOF'
import os, re, sys
src, dst, kernel, team, warm, samp = sys.argv[1:]
s = open(src).read()
here = os.path.dirname(os.path.abspath(src))
s = re.sub(r"(\n  file: )(\S+)", lambda m: m.group(1) + os.path.normpath(os.path.join(here, m.group(2))), s, count=1)
s = re.sub(r"\n  dt_DSMCState_out:[^\n]*", "", s)
s = re.sub(r"(\n  warmup_steps: )\d+", r"\g<1>" + warm, s)
s = re.sub(r"(\n  sample_steps: )\d+", r"\g<1>" + samp, s)
s = re.sub(r"\n  collision_kernel:[^\n]*|\n  collision_team_size:[^\n]*", "", s)
extra = "\n  collision_kernel: " + kernel + ("\n  collision_team_size: " + team if team else "")
s = s.replace("\ndsmc:", "\ndsmc:" + extra, 1)
open(dst, "w").write(s)
EOF
    echo "== $(date -u +%H:%M:%S) $name $tag"
    VLEO_PROFILE=1 "$BIN" --config "$d/case.yaml" --platform "$ROOT/platform.yaml" --output-dir "$d/run" --quiet \
      > "$d/solver.log" 2>&1 || { echo "   failed, see $d/solver.log"; tail -3 "$d/solver.log"; }
  done
done

python3 - "$ROOT" <<'EOF' | tee "$ROOT/summary.txt"
import glob, json, os, sys
root = sys.argv[1]
for case in sorted(os.listdir(root)):
    cdir = os.path.join(root, case)
    if not os.path.isdir(cdir):
        continue
    rows = []
    for d in sorted(glob.glob(os.path.join(cdir, "*", "run", "results.json"))):
        tag = d.split(os.sep)[-3]
        g = json.load(open(d))["diagnostics"]
        p = g.get("profile", {})
        steps = g["steps"] if isinstance(g.get("steps"), (int, float)) else (g["collisions"].get("steps") or 1)
        col = p.get("dsmc.collide", {}).get("seconds", float("nan"))
        loop = p.get("run.steps", {}).get("seconds", float("nan"))
        c = g["collisions"]
        rows.append((tag, 1e3 * col / steps, 1e3 * loop / steps, c["candidates"], c["accepted"],
                     g["particles_mean"], c["candidates"] / max(1, steps)))
    if not rows:
        continue
    print(f"\n{case}  ({rows[0][5]:.3g} particles, {rows[0][6]:.3g} candidates per step)")
    print(f"{'variant':<10} {'collide [ms/step]':>18} {'step [ms]':>10} {'vs waves_32':>12} {'candidates':>12} {'accepted':>12}")
    ref = next((r[1] for r in rows if r[0] == "waves_32"), None)
    for tag, col, loop, cand, acc, *_ in sorted(rows, key=lambda r: r[1]):
        rel = f"{col / ref:.2f}x" if ref else ""
        print(f"{tag:<10} {col:>18.2f} {loop:>10.2f} {rel:>12} {cand:>12} {acc:>12}")
    same = len({(r[3], r[4]) for r in rows}) == 1
    print("collision counters identical for all variants:", "yes" if same else "NO (check!)")
EOF
