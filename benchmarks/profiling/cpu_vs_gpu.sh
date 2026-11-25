#!/usr/bin/env bash
# CPU (Kokkos OpenMP) vs. GPU (Kokkos HIP) on the closed air reservoir P3.
#
#   DEVICE=1 ./cpu_vs_gpu.sh [case.yaml]          (default: p3_reservoir_air.yaml)
#
# Runs the case without cell-state output, with the fenced phase profiler, on
#   - the GPU build (GPU_BIN), once,
#   - the CPU build (CPU_BIN) with THREADS OpenMP threads (default: 1 2 4 ... up to the
#     physical cores, and the physical cores themselves), OMP_PROC_BIND=spread, OMP_PLACES=cores.
# The full-size run (15 M particles, 1.5 M collisions per step) uses STEPS steps on the GPU and
# on the CPU with all cores (default 50 warm-up + 150 sampled); the thread scaling uses
# SCALE_STEPS (default 10 + 20) to keep the 1-thread run short.
# Printed: ms per step, ns per particle step, collisions per second, speed-up, the phase split,
# and whether CPU and GPU produced the same collision counts (same random streams; bitwise
# agreement is expected unless the compilers contract floating-point operations differently).
#
# environment
#   GPU_BIN   default $DSMC/build/hip-gfx1101-hdf5/vleo_solver
#   CPU_BIN   default $DSMC/build/cpu-release/vleo_solver   (cmake --preset cpu-release && cmake --build build/cpu-release -j)
#   DEVICE    HIP device index (default 0);  THREADS  e.g. "1 4 16 32"
#   STEPS     "warmup sample" for the full runs (default "50 150");  SCALE_STEPS (default "10 20")
#   SKIP_GPU=1 / SKIP_SCALING=1
#   FULL_THREADS  threads of the full-size CPU run (default: physical cores)
#   OMP_BIND / OMP_PLACES_  thread placement (default spread / cores); e.g. OMP_BIND=close keeps
#             the first threads on one socket (NUMA test: FULL_THREADS=32 OMP_BIND=close)
#   TAG       suffix of the result directory (several placements side by side)
#   CPU_LAUNCH  launcher of the full-size CPU run, e.g. one MPI rank per socket with
#             FULL_THREADS threads each (CPU_BIN from the cpu-mpi preset):
#             CPU_LAUNCH="mpiexec -n 2 --map-by socket --bind-to socket" FULL_THREADS=32 OMP_BIND=close
set -euo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DSMC=${DSMC:-$(cd "$HERE/../.." && pwd)}
GPU_BIN=${GPU_BIN:-$DSMC/build/hip-gfx1101-hdf5/vleo_solver}
CPU_BIN=${CPU_BIN:-$DSMC/build/cpu-release/vleo_solver}
GPU_BACKEND=${GPU_BACKEND:-hip}
DEVICE=${DEVICE:-0}
OMP_BIND=${OMP_BIND:-spread}
OMP_PLACES_=${OMP_PLACES_:-cores}
CASE=${1:-p3_reservoir_air.yaml}
read -r W_FULL S_FULL <<< "${STEPS:-50 150}"
read -r W_SC S_SC <<< "${SCALE_STEPS:-10 20}"
CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[[ $CORES -gt 0 ]] || CORES=$(nproc)
FULL_THREADS=${FULL_THREADS:-$CORES}
if [[ -z ${THREADS:-} ]]; then
  THREADS=""; t=1
  while (( t < CORES )); do THREADS+="$t "; t=$((t * 2)); done
  THREADS+="$CORES"
fi
[[ -x $CPU_BIN ]] || { echo "error: CPU build not found: $CPU_BIN"; echo "  cmake --preset cpu-release && cmake --build build/cpu-release -j"; exit 1; }
[[ ${SKIP_GPU:-0} == 1 || -x $GPU_BIN ]] || { echo "error: GPU build not found: $GPU_BIN"; exit 1; }

name=$(basename "$CASE" .yaml)
ROOT="$HERE/results/cpu_vs_gpu/$name${TAG:+_$TAG}"
mkdir -p "$ROOT"
echo "host $(hostname), $(lscpu | sed -n 's/^Model name: *//p' | head -1), $CORES physical cores, $(lscpu | sed -n 's/^NUMA node(s): *//p') NUMA nodes; threads: $THREADS; OMP_PROC_BIND=$OMP_BIND OMP_PLACES=$OMP_PLACES_"
lscpu > "$ROOT/lscpu.txt" 2>/dev/null || true

make_case() {  # make_case <dst> <warmup> <sample>
  python3 - "$CASE" "$1" "$2" "$3" <<'EOF'
import os, re, sys
src, dst, warm, samp = sys.argv[1:]
s = open(src).read()
here = os.path.dirname(os.path.abspath(src))
s = re.sub(r"(\n  file: )(\S+)", lambda m: m.group(1) + os.path.normpath(os.path.join(here, m.group(2))), s, count=1)
s = re.sub(r"\n  (dt_DSMCState_out|state_sample_every):[^\n]*", "", s)   # no cell-state output (CPU build without HDF5)
s = re.sub(r"(\n  warmup_steps: )\d+", r"\g<1>" + warm, s)
s = re.sub(r"(\n  sample_steps: )\d+", r"\g<1>" + samp, s)
s = re.sub(r"(\n  blocks: )\d+", r"\g<1>2", s)
open(dst, "w").write(s)
EOF
}

run() {  # run <tag> <backend> <bin> <warmup> <sample> [threads] [launcher]
  local tag=$1 backend=$2 bin=$3 w=$4 s=$5 th=${6:-0} launch=${7:-} d="$ROOT/$1"
  mkdir -p "$d"
  make_case "$d/case.yaml" "$w" "$s"
  printf "platform:\n  backend: %s\n" "$backend" > "$d/platform.yaml"
  [[ $backend != cpu ]] && printf "  device_id: %s\n" "$DEVICE" >> "$d/platform.yaml"
  [[ $th -gt 0 ]] && printf "  num_threads: %s\n" "$th" >> "$d/platform.yaml"
  echo "== $(date -u +%H:%M:%S) $tag"
  OMP_NUM_THREADS=$([[ $th -gt 0 ]] && echo "$th" || echo 1) OMP_PROC_BIND=$OMP_BIND OMP_PLACES=$OMP_PLACES_ VLEO_PROFILE=1 \
    $launch "$bin" --config "$d/case.yaml" --platform "$d/platform.yaml" --output-dir "$d/run" --quiet > "$d/solver.log" 2>&1 \
    || { echo "   failed, see $d/solver.log"; tail -3 "$d/solver.log"; }
}

[[ ${SKIP_GPU:-0} == 1 ]] || run gpu "$GPU_BACKEND" "$GPU_BIN" "$W_FULL" "$S_FULL"
RANKS=$(echo "${CPU_LAUNCH:-}" | sed -n 's/.*-n[p]* *\([0-9]*\).*/\1/p')
run "cpu_full_${RANKS:+${RANKS}x}${FULL_THREADS}t" cpu "$CPU_BIN" "$W_FULL" "$S_FULL" "$FULL_THREADS" "${CPU_LAUNCH:-}"
if [[ ${SKIP_SCALING:-0} != 1 ]]; then
  for t in $THREADS; do run "cpu_scale_${t}t" cpu "$CPU_BIN" "$W_SC" "$S_SC" "$t"; done
fi

python3 - "$ROOT" "$FULL_THREADS" <<'EOF' | tee "$ROOT/summary.txt"
import glob, json, os, re, sys
root, cores = sys.argv[1], int(sys.argv[2])
def load(tag):
    f = os.path.join(root, tag, "run", "results.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); g = d["diagnostics"]; p = g.get("profile", {})
    steps = g["steps"]; loop = p.get("run.steps", {}).get("seconds", g["wall_time_s"])
    ph = {k: p.get(k, {}).get("seconds", 0.0) / steps * 1e3 for k in
          ("dsmc.move", "dsmc.collide", "dsmc.reorder", "dsmc.step_reduce", "dsmc.sample")}
    ph["binning"] = sum(v["seconds"] for k, v in p.items() if k.startswith("dsmc.bin.")) / steps * 1e3
    for k in ("histogram", "scan", "fill", "sort"):
        ph[k] = p.get("dsmc.bin." + k, {}).get("seconds", 0.0) / steps * 1e3
    acc = g["collisions"]["accepted"]
    return dict(tag=tag, steps=steps, ms=loop / steps * 1e3, ns=loop / g["particle_steps"] * 1e9,
                coll_s=acc / loop, acc=acc, cand=g["collisions"]["candidates"],
                ratio=g["collisions"]["collisions_per_particle_per_step"],
                theory=g["collisions"]["free_stream_collisions_per_particle_per_step"],
                particles=g["particles_mean"], CD=d["results"]["C_D"], ph=ph)
gpu = load("gpu")
full = next((load(os.path.basename(t)) for t in sorted(glob.glob(os.path.join(root, "cpu_full_*")))), None)
rows = [r for r in (gpu, full) if r]
print(f"\nFull size ({rows[0]['particles']:.4g} particles, {rows[0]['steps']} steps)")
print(f"{'run':<16} {'ms/step':>9} {'ns/(p*step)':>12} {'collisions/s':>14} {'move':>7} {'collide':>8} {'binning':>8} {'reorder':>8} {'reduce':>7}   [ms/step]")
for r in rows:
    ph = r["ph"]
    print(f"{r['tag']:<16} {r['ms']:>9.1f} {r['ns']:>12.2f} {r['coll_s']:>14.3e} {ph['dsmc.move']:>7.1f} {ph['dsmc.collide']:>8.1f}"
          f" {ph['binning']:>8.1f} {ph['dsmc.reorder']:>8.1f} {ph['dsmc.step_reduce']:>7.1f}")
print("binning split [ms/step]: " + "; ".join(
    f"{r['tag']} " + " ".join(f"{k} {r['ph'][k]:.1f}" for k in ("histogram", "scan", "fill", "sort")) for r in rows))
for r in rows:
    print(f"  {r['tag']}: {r['acc']} collisions ({r['ratio']:.5f} per particle and step, theory {r['theory']:.5f})")
if gpu and full:
    print(f"GPU speed-up over {cores} CPU cores: {full['ms'] / gpu['ms']:.1f}x;"
          f" collision counts {'identical' if gpu['acc'] == full['acc'] and gpu['cand'] == full['cand'] else 'DIFFERENT'}"
          f" (C_D GPU {gpu['CD']:.10g}, CPU {full['CD']:.10g})")
sc = sorted((r for r in (load(os.path.basename(t)) for t in glob.glob(os.path.join(root, "cpu_scale_*t"))) if r),
            key=lambda r: int(re.search(r"_(\d+)t$", r["tag"]).group(1)))
nthreads = lambda r: int(re.search(r"_(\d+)t$", r["tag"]).group(1))
if sc:
    t1 = sc[0]["ms"] * nthreads(sc[0])  # 1-thread time per step (ideal scaling from the first entry)
    print(f"\nCPU thread scaling ({sc[0]['steps']} steps)")
    print(f"{'threads':>8} {'ms/step':>9} {'ns/(p*step)':>12} {'speed-up':>9} {'efficiency':>11}"
          f" {'move':>7} {'collide':>8} {'binning':>8} {'reorder':>8}")
    for r in sc:
        t = nthreads(r)
        ph = r["ph"]
        print(f"{t:>8} {r['ms']:>9.1f} {r['ns']:>12.2f} {t1 / r['ms']:>9.2f} {t1 / r['ms'] / t:>10.0%}"
              f" {ph['dsmc.move']:>7.1f} {ph['dsmc.collide']:>8.1f} {ph['binning']:>8.1f} {ph['dsmc.reorder']:>8.1f}")
    if gpu:
        print(f"GPU vs 1 CPU thread: {t1 / gpu['ms']:.0f}x")
EOF
