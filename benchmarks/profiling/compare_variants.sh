#!/usr/bin/env bash
# Timing of DSMC option variants on one GPU (fenced phase profiler, shortened runs).
#
#   DEVICE=1 VARIANTS="reorder_interval=0 reorder_interval=5 reorder_interval=20" \
#       ./compare_variants.sh [case.yaml ...]            (default case: p3_reservoir_air.yaml)
#
# A variant is a comma-separated list of dsmc options, e.g.
#   "collision_kernel=waves,collision_team_size=16"  or  "reorder_interval=5,state_sample_every=5".
# Each case runs WARMUP + SAMPLE steps (default 100 + 200). The cell-state output is kept
# (KEEP_STATE=1, default) or removed (KEEP_STATE=0). Printed per variant: the phases per step and
# the collision counters, which must agree for all variants (the options change speed, not physics).
# Results: results/variants/<case>/<tag>/  and  results/variants/summary.txt
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
KEEP_STATE=${KEEP_STATE:-1}
VARIANTS=${VARIANTS:-"reorder_interval=0 reorder_interval=2 reorder_interval=5 reorder_interval=20"}
[[ -x "$BIN" ]] || { echo "error: solver binary not found: $BIN (set BIN=...)"; exit 1; }
CASES=("$@")
[[ ${#CASES[@]} -gt 0 ]] || CASES=(p3_reservoir_air.yaml)
ROOT="$HERE/results/variants"
mkdir -p "$ROOT"
cat > "$ROOT/platform.yaml" <<EOF
platform:
  backend: $BACKEND
  device_id: $DEVICE
EOF

for CASE in "${CASES[@]}"; do
  name=$(basename "$CASE" .yaml)
  for v in $VARIANTS; do
    tag=$(echo "$v" | tr ',=' '_-')
    d="$ROOT/$name/$tag"
    mkdir -p "$d"
    python3 - "$CASE" "$d/case.yaml" "$v" "$WARMUP" "$SAMPLE" "$KEEP_STATE" <<'EOF'
import os, re, sys
src, dst, variant, warm, samp, keep = sys.argv[1:]
s = open(src).read()
here = os.path.dirname(os.path.abspath(src))
s = re.sub(r"(\n  file: )(\S+)", lambda m: m.group(1) + os.path.normpath(os.path.join(here, m.group(2))), s, count=1)
if keep != "1":
    s = re.sub(r"\n  (dt_DSMCState_out|state_sample_every):[^\n]*", "", s)
s = re.sub(r"(\n  warmup_steps: )\d+", r"\g<1>" + warm, s)
s = re.sub(r"(\n  sample_steps: )\d+", r"\g<1>" + samp, s)
extra = ""
for kv in filter(None, variant.split(",")):
    k, val = kv.split("=", 1)
    s = re.sub(r"\n  " + re.escape(k) + r":[^\n]*", "", s)
    extra += f"\n  {k}: {val}"
s = s.replace("\ndsmc:", "\ndsmc:" + extra, 1)
open(dst, "w").write(s)
EOF
    echo "== $(date -u +%H:%M:%S) $name $tag"
    VLEO_PROFILE=1 "$BIN" --config "$d/case.yaml" --platform "$ROOT/platform.yaml" --output-dir "$d/run" --quiet \
      > "$d/solver.log" 2>&1 || { echo "   failed, see $d/solver.log"; tail -3 "$d/solver.log"; }
  done
done

python3 - "$ROOT" "$VARIANTS" <<'EOF' | tee "$ROOT/summary.txt"
import json, os, sys
root, variants = sys.argv[1], sys.argv[2].split()
tags = [v.replace(",", "_").replace("=", "-") for v in variants]
phases = ["dsmc.move", "dsmc.collide", "dsmc.cell_state", "dsmc.reorder", "dsmc.bin.histogram", "dsmc.bin.fill",
          "dsmc.bin.sort", "dsmc.step_reduce", "dsmc.compact"]
for case in sorted(os.listdir(root)):
    cdir = os.path.join(root, case)
    if not os.path.isdir(cdir):
        continue
    rows = []
    for tag in tags:  # only the variants of this call
        f = os.path.join(cdir, tag, "run", "results.json")
        if not os.path.exists(f):
            continue
        g = json.load(open(f))["diagnostics"]
        p, steps = g.get("profile", {}), g["steps"]
        ms = lambda k: 1e3 * p.get(k, {}).get("seconds", 0.0) / steps
        rows.append((tag, ms("run.steps"), [ms(k) for k in phases], g["collisions"]["accepted"],
                     g["surface_interactions"], g["particles_mean"]))
    if not rows:
        continue
    print(f"\n{case}  ({rows[0][5]:.3g} particles), ms per step")
    short = [k.replace("dsmc.", "").replace("bin.", "") for k in phases]
    print(f"{'variant':<34} {'step':>7} " + " ".join(f"{h:>10}" for h in short))
    for tag, step, ph, *_ in rows:
        print(f"{tag:<34} {step:>7.2f} " + " ".join(f"{x:>10.2f}" for x in ph))
    same = len({(r[3], r[4]) for r in rows}) == 1
    print("collisions and surface interactions identical for all variants:", "yes" if same else "NO (check!)")
EOF
