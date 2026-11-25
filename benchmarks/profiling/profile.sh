#!/usr/bin/env bash
# Profile vleo_solver on one GPU: which kernel / phase runs when on the GPU and the CPU.
#
#   ./profile.sh <mode> [case.yaml]          (default case: p3_reservoir_air.yaml)
#
# modes
#   phases   built-in fenced phase timings (--profile) -> results.json, no external tool
#   kokkos   kp_trace.cpp (built here with the host C++ compiler, no vendor tool needed):
#            timeline of Kokkos regions, kernels, deep copies and fences
#            -> CSV + Chrome/Perfetto JSON (KP_TRACE_FENCE=0: asynchronous, see kp_trace.cpp)
#   rocprof  rocprofv3: GPU kernel + memory-copy timeline, HIP API calls on the CPU threads,
#            Kokkos regions/kernel labels (--kokkos-trace) -> Perfetto trace + CSV
#   sys      rocprof-sys (omnitrace): CPU call-stack sampling + GPU activity + Kokkos
#            regions in one Perfetto trace
#   perf     Linux perf: CPU call-graph sampling of the host process
#   all      phases, kokkos, rocprof, sys, perf one after the other (skips missing tools)
# Every mode also records monitor.csv (GPU busy %, VRAM, power; CPU % and RSS of the
# solver process; every 0.25 s) and finally runs analyze_profile.py.
#
# environment
#   DSMC    checkout of GPU_DSMC                          (default: ../.. of this script)
#   BIN     solver binary         (default: $DSMC/build/hip-gfx1101-hdf5/vleo_solver)
#   DEVICE  HIP device index (rocm-smi)                   (default: 0)
#   OUT     output root                                   (default: results/<case>)
#   PERIOD  monitor sampling period in s                  (default: 0.25)
#   BACKEND platform backend                              (default: hip; cpu for a CPU build)
set -euo pipefail
export LC_ALL=C   # decimal points in awk output
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
MODE=${1:?usage: ./profile.sh phases|kokkos|rocprof|sys|perf|all [case.yaml]}
CASE=${2:-p3_reservoir_air.yaml}
DSMC=${DSMC:-$(cd "$HERE/../.." && pwd)}
BIN=${BIN:-$DSMC/build/hip-gfx1101-hdf5/vleo_solver}
DEVICE=${DEVICE:-0}
NAME=$(basename "$CASE" .yaml)
OUT=${OUT:-results/$NAME}
PERIOD=${PERIOD:-0.25}
BACKEND=${BACKEND:-hip}
[[ -x "$BIN" ]] || { echo "error: solver binary not found: $BIN (set BIN=...)"; exit 1; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)   # absolute: the profilers run with their own paths

cat > "$OUT/platform_gpu.yaml" <<EOF
platform:
  backend: $BACKEND
  device_id: $DEVICE
  description: profiling run
EOF

# --- monitor: GPU (sysfs, all AMD GPUs) and the solver process (/proc) -------------
monitor() {  # monitor <pid> <csv>
  local pid=$1 csv=$2 cards=() d
  for d in /sys/class/drm/card*/device; do
    [[ -r $d/gpu_busy_percent ]] || continue
    local real; real=$(readlink -f "$d")
    [[ " ${cards[*]-} " == *" $real "* ]] || cards+=("$real")
  done
  {
    printf "t_s,proc_cpu_pct,proc_rss_MB,proc_threads"
    for d in "${cards[@]}"; do local b; b=$(basename "$d"); printf ",gpu_busy_pct[%s],vram_used_MB[%s],power_W[%s]" "$b" "$b" "$b"; done
    echo
  } > "$csv"
  local hz; hz=$(getconf CLK_TCK)
  local t0; t0=$(date +%s.%N)
  local last_ticks="" last_t=""
  while kill -0 "$pid" 2>/dev/null; do
    local now; now=$(date +%s.%N)
    local stat; stat=$(cat "/proc/$pid/stat" 2>/dev/null) || break
    # fields after the command name: utime = 14, stime = 15, num_threads = 20, rss = 24 (pages)
    local rest=${stat##*) }
    read -r -a f <<< "$rest"
    local ticks=$(( f[11] + f[12] )) threads=${f[17]} rss_pages=${f[21]}
    local cpu=""
    if [[ -n $last_ticks ]]; then
      cpu=$(awk -v a="$ticks" -v b="$last_ticks" -v t="$now" -v s="$last_t" -v hz="$hz" \
            'BEGIN{printf "%.1f", 100*(a-b)/hz/(t-s)}')
    fi
    last_ticks=$ticks; last_t=$now
    local line
    line=$(awk -v t="$now" -v t0="$t0" -v c="$cpu" -v r="$rss_pages" -v p="$(getconf PAGESIZE)" -v n="$threads" \
           'BEGIN{printf "%.3f,%s,%.1f,%d", t-t0, c, r*p/1048576, n}')
    for d in "${cards[@]}"; do
      local busy vram pw=""
      busy=$(cat "$d/gpu_busy_percent" 2>/dev/null || echo "")
      vram=$(cat "$d/mem_info_vram_used" 2>/dev/null || echo 0)
      for h in "$d"/hwmon/hwmon*/power1_average "$d"/hwmon/hwmon*/power1_input; do
        [[ -r $h ]] && { pw=$(cat "$h"); break; }
      done
      line+=$(awk -v b="$busy" -v v="$vram" -v p="$pw" 'BEGIN{printf ",%s,%.0f,%s", b, v/1048576, (p==""?"":sprintf("%.1f",p/1e6))}')
    done
    echo "$line" >> "$csv"
    sleep "$PERIOD"
  done
}

# run_with <dir> <command...>: runs the solver (prefixed by a profiler) with the monitor
run_with() {
  local dir=$1; shift
  mkdir -p "$dir"
  echo "== $(date -u +%H:%M:%S) $MODE_NOW -> $dir"
  "$@" --config "$CASE" --platform "$OUT/platform_gpu.yaml" --output-dir "$dir/run" > "$dir/solver.log" 2>&1 &
  local launcher=$!
  # the profiler may fork: monitor the vleo_solver process itself
  local pid="" i p
  for i in $(seq 1 400); do
    for p in $(pgrep -x vleo_solver || true); do
      tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qF -- "--output-dir $dir/run" && pid=$p
    done
    [[ -n $pid ]] && break
    sleep 0.05
  done
  [[ -n $pid ]] || echo "warning: solver process not found, no monitor.csv"
  [[ -n $pid ]] && monitor "$pid" "$dir/monitor.csv" &
  local mon=$!
  local rc=0
  wait "$launcher" || rc=$?
  wait "$mon" 2>/dev/null || true
  tail -2 "$dir/solver.log"
  [[ $rc -eq 0 ]] || echo "warning: exit code $rc (see $dir/solver.log)"
  return 0
}

have() { command -v "$1" >/dev/null 2>&1; }

do_phases() {
  MODE_NOW=phases run_with "$OUT/phases" env VLEO_PROFILE=1 "$BIN"
}

do_kokkos() {
  local d="$OUT/kokkos" cxx
  cxx=$(command -v g++ || command -v c++ || command -v clang++ || true)
  [[ -n $cxx ]] || { echo "skip kokkos: no host C++ compiler"; return; }
  mkdir -p "$d"
  "$cxx" -O2 -std=c++17 -shared -fPIC -o "$d/libkp_trace.so" "$HERE/kp_trace.cpp"
  MODE_NOW=kokkos run_with "$d" env KOKKOS_TOOLS_LIBS="$d/libkp_trace.so" KP_TRACE_OUT="$d/trace" \
      KP_TRACE_FENCE="${KP_TRACE_FENCE:-1}" "$BIN"
}

do_rocprof() {
  if ! have rocprofv3; then
    echo "skip rocprof: rocprofv3 not found (module avail 2>&1 | grep -i rocprof)"; return
  fi
  local d="$OUT/rocprof"
  MODE_NOW=rocprof run_with "$d" rocprofv3 \
      --kernel-trace --memory-copy-trace --hip-runtime-trace --marker-trace --kokkos-trace \
      --stats --output-format csv pftrace -d "$d/trace" -o vleo -- "$BIN"
}

do_sys() {
  local tool="" pre=""
  if have rocprof-sys-sample; then tool=rocprof-sys-sample; pre=ROCPROFSYS
  elif have omnitrace-sample; then tool=omnitrace-sample; pre=OMNITRACE
  else echo "skip sys: neither rocprof-sys-sample nor omnitrace-sample found"; return; fi
  local d="$OUT/sys"
  local -a envs=("${pre}_OUTPUT_PATH=$d/trace" "${pre}_USE_PERFETTO=true" "${pre}_USE_TIMEMORY=true"
                 "${pre}_USE_SAMPLING=true" "${pre}_SAMPLING_FREQ=250" "${pre}_USE_PROCESS_SAMPLING=true"
                 "${pre}_USE_KOKKOSP=true")
  if [[ $pre == ROCPROFSYS ]]; then envs+=("ROCPROFSYS_USE_ROCM=true"); else envs+=("OMNITRACE_USE_ROCTRACER=true"); fi
  MODE_NOW=sys run_with "$d" env "${envs[@]}" "$tool" -- "$BIN"
}

do_perf() {
  if ! have perf; then echo "skip perf: perf not found"; return; fi
  local par; par=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 2)
  if (( par > 2 )); then
    echo "skip perf: kernel.perf_event_paranoid = $par (needs <= 2: sudo sysctl kernel.perf_event_paranoid=2)"; return
  fi
  local d="$OUT/perf"
  MODE_NOW=perf run_with "$d" perf record -F 499 --call-graph dwarf,16384 -o "$d/perf.data" -- "$BIN"
  if [[ -s $d/perf.data ]]; then
    perf report -i "$d/perf.data" --no-children --sort dso,sym --stdio 2>/dev/null | grep -v '^$' | head -80 \
      > "$d/perf_report.txt" || true
    perf report -i "$d/perf.data" --no-children --sort comm,tid --stdio 2>/dev/null | grep -v '^$' | head -40 \
      > "$d/perf_threads.txt" || true
    echo "   $d/perf_report.txt (flame graph: perf script -i $d/perf.data | stackcollapse-perf.pl | flamegraph.pl)"
  fi
}

"$BIN" --check --config "$CASE" --platform "$OUT/platform_gpu.yaml"
case $MODE in
  phases) do_phases ;;
  kokkos) do_kokkos ;;
  rocprof) do_rocprof ;;
  sys) do_sys ;;
  perf) do_perf ;;
  all) do_phases; do_kokkos; do_rocprof; do_sys; do_perf ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

python3 "$HERE/analyze_profile.py" "$OUT" || echo "warning: analyze_profile.py failed"
