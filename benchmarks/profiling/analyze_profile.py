#!/usr/bin/env python3
"""Summary and timeline plots of a profile.sh run.

    python3 analyze_profile.py results/<case>

Reads what is present below results/<case>/:
  */monitor.csv          GPU busy %, VRAM, power, CPU % of the solver over time -> monitor_<mode>.png
  phases/run/results.json   fenced phase timings (diagnostics.profile)
  rocprof/trace/**.csv   rocprofv3 traces: kernel_trace, memory_copy_trace, hip_api_trace,
                         marker_api_trace (Kokkos regions + kernel labels with --kokkos-trace)
                         -> kernel/phase tables, GPU idle gaps, HIP API time per call,
                            timeline_*.png (CPU regions, CPU HIP API calls, GPU kernels, copies)
  perf/perf_report.txt   (only referenced)
Writes results/<case>/summary.md and the figures next to it. Needs numpy and matplotlib.
"""
import bisect, csv, glob, json, os, re, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GROUPS = [  # (group, substrings of the Kokkos label / region name / functor), first match wins
    ("move", ["dsmc_move", "dsmc.move", "MoveKernel"]),
    ("binning", ["dsmc_cell_index", "dsmc_offsets", "dsmc_fill_cells", "dsmc_rank_sort", "dsmc_sort_cells",
                 "dsmc.bin", "CellIndexKernel", "OffsetScan", "FillCells", "RankSortEntries", "SortCells"]),
    ("collide", ["dsmc_candidates", "dsmc_collide", "dsmc_collision_counts", "dsmc.collide", "CellCandidates",
                 "CollideCells", "SumU64x3"]),
    ("reduce", ["dsmc_step_reduce", "dsmc_status_reduce", "dsmc.step_reduce", "StepReduce", "StatusReduce"]),
    ("cell_state", ["dsmc_cell_state", "dsmc.cell_state", "CellMoments", "dsmc_state_output"]),
    ("sample", ["dsmc_count_sum", "dsmc_max_speed", "dsmc.sample", "AccumulateCounts", "MaxSpeed"]),
    ("inject", ["dsmc_inject", "dsmc.inject", "InjectKernel"]),
    ("compact", ["dsmc_alive", "dsmc_compact", "dsmc.compact", "AliveFlag", "CompactCopy"]),
    ("migrate", ["migrant", "dsmc.migrate", "dsmc.mpi", "rebalance"]),
    ("kokkos", ["Kokkos::", "deep_copy", "ViewFill", "ViewCopy"]),
]
COLORS = {"move": "#1f77b4", "binning": "#ff7f0e", "collide": "#d62728", "reduce": "#9467bd",
          "cell_state": "#2ca02c", "sample": "#8c564b", "inject": "#e377c2", "compact": "#17becf",
          "migrate": "#bcbd22", "kokkos": "#7f7f7f", "other": "#c7c7c7"}
SYNC_CALLS = re.compile(r"Synchronize|Memcpy(?!Async)|hipMemcpyDtoH\b|EventSynchronize|hipFree\b")


def group_of(name):
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


REGIONS = {"vleo::dsmc_setup", "vleo::dsmc_fill", "vleo::dsmc_state_output", "vleo::dsmc_progress",
           "vleo::dsmc_result", "vleo::self_test", "vleo::write_outputs"}


def is_region(name):
    """vleo Kokkos regions (profile phases and run stages), as opposed to kernel labels."""
    return name in REGIONS or name.startswith("dsmc.") or name.startswith("vleo::dsmc_step(")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find(root, pattern):
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def col(row, *names, default=None):
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    return default


# --- monitor ----------------------------------------------------------------------
def monitor_plot(csv_path, png, title):
    rows = read_csv(csv_path)
    if len(rows) < 3:
        return None
    keys = rows[0].keys()
    t = np.array([float(r["t_s"]) for r in rows])
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    busy_keys = [k for k in keys if k.startswith("gpu_busy_pct")]
    # the profiled GPU = the one with the highest mean load
    busy = {k: f(k) for k in busy_keys}
    main = max(busy, key=lambda k: np.nanmean(busy[k])) if busy else None
    card = main[main.index("["):] if main else ""
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    if main:
        ax[0].plot(t, busy[main], lw=0.8, color="#d62728", label="GPU busy " + card)
        for k in busy_keys:
            if k != main:
                ax[0].plot(t, busy[k], lw=0.6, color="0.6", label="GPU busy " + k[k.index("["):])
    ax[0].plot(t, f("proc_cpu_pct"), lw=0.8, color="#1f77b4", label="CPU of the solver (100 % = 1 core)")
    ax[0].set_ylabel("%"); ax[0].legend(fontsize=8, loc="upper right")
    if main:
        ax[1].plot(t, f("vram_used_MB" + card) / 1024, color="#2ca02c", label="VRAM used " + card)
    ax[1].plot(t, f("proc_rss_MB") / 1024, color="#1f77b4", label="host RSS of the solver")
    ax[1].set_ylabel("GB"); ax[1].legend(fontsize=8)
    pk = "power_W" + card
    if pk in keys:
        ax[2].plot(t, f(pk), color="#ff7f0e", lw=0.8)
    ax[2].set_ylabel("GPU power [W]"); ax[2].set_xlabel("time since start [s]")
    fig.suptitle(title, fontsize=10)
    fig.savefig(png, dpi=110); plt.close(fig)
    stats = {"duration_s": t[-1] - t[0], "cpu_pct_mean": np.nanmean(f("proc_cpu_pct")),
             "threads_max": int(np.nanmax(f("proc_threads")))}
    if main:
        stats.update(gpu=card, gpu_busy_mean=np.nanmean(busy[main]), vram_max_GB=np.nanmax(f("vram_used_MB" + card)) / 1024)
        if pk in keys:
            stats["power_mean_W"] = np.nanmean(f(pk))
    return stats


# --- rocprofv3 --------------------------------------------------------------------
class Ranges:
    """Host ranges (markers) of one thread with nesting depth; leaf lookup by time."""

    def __init__(self, items):  # items: (start, end, name)
        # equal intervals: the region is the outer range, the kernel label the inner one
        self.items = sorted(items, key=lambda x: (x[0], -x[1], 0 if is_region(x[2]) else 1))
        self.depth, stack = [], []
        for s, e, _ in self.items:
            while stack and stack[-1] <= s:
                stack.pop()
            self.depth.append(len(stack))
            stack.append(e)
        # leaves: ranges that contain no other range
        leaves = [it for i, it in enumerate(self.items)
                  if i + 1 == len(self.items) or self.items[i + 1][0] >= it[1]]
        self.leaf_start = [x[0] for x in leaves]
        self.leaves = leaves

    def leaf_at(self, t):
        i = bisect.bisect_right(self.leaf_start, t) - 1
        if i >= 0 and self.leaves[i][1] >= t:
            return self.leaves[i][2]
        return None


def short_kernel(name):
    m = re.findall(r"vleo::(?:\(anonymous namespace\)::|detail::)?([A-Za-z_]\w*)", name)
    if m:
        return m[0]
    m = re.search(r"Kokkos::Impl::(\w+)", name)
    return "Kokkos::" + m.group(1) if m else name[:60]


def load_rocprof(root):
    kt = find(root, "*kernel_trace.csv")
    if not kt:
        return None
    R = {"files": {os.path.basename(p): p for p in find(root, "*.csv")}}
    kern = read_csv(kt[0])
    api = read_csv(find(root, "*hip_api_trace.csv")[0]) if find(root, "*hip_api_trace.csv") else []
    mark = read_csv(find(root, "*marker_api_trace.csv")[0]) if find(root, "*marker_api_trace.csv") else []
    copy = read_csv(find(root, "*memory_copy_trace.csv")[0]) if find(root, "*memory_copy_trace.csv") else []
    ts = lambda r: (int(col(r, "Start_Timestamp", "start")), int(col(r, "End_Timestamp", "end")))

    markers = defaultdict(list)
    for r in mark:
        s, e = ts(r)
        if e > s:
            markers[col(r, "Thread_Id", "tid", default="0")].append((s, e, col(r, "Function", "Name", default="?")))
    ranges = {tid: Ranges(v) for tid, v in markers.items()}
    api_by_corr = {}
    api_rows = []
    for r in api:
        s, e = ts(r)
        tid = col(r, "Thread_Id", default="0")
        fn = col(r, "Function", "Name", default="?")
        api_rows.append((s, e, fn, tid))
        c = col(r, "Correlation_Id")
        if c is not None:
            api_by_corr[c] = (s, tid)
    kernels = []
    for r in kern:
        s, e = ts(r)
        raw = col(r, "Kernel_Name", "Name", default="?")
        label = None
        c = col(r, "Correlation_Id")
        if c in api_by_corr:  # label of the Kokkos kernel around the launching HIP call
            t, tid = api_by_corr[c]
            if tid in ranges:
                label = ranges[tid].leaf_at(t)
        if label is None or is_region(label):
            label = short_kernel(raw)
        kernels.append((s, e, label))
    copies = []
    for r in copy:
        s, e = ts(r)
        copies.append((s, e, col(r, "Direction", "Kind", default="copy"), int(col(r, "Bytes", "Size", default=0) or 0)))
    R.update(kernels=sorted(kernels), api=sorted(api_rows), ranges=ranges, copies=sorted(copies))
    return R


def union_length(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def gaps(iv):
    out, last = [], None
    for s, e in sorted(iv):
        if last is not None and s > last:
            out.append(s - last)
        last = e if last is None else max(last, e)
    return np.array(out)


def rocprof_summary(R, md, figdir):
    K = R["kernels"]
    main_tid = max(R["ranges"], key=lambda k: len(R["ranges"][k].items)) if R["ranges"] else None
    rg = R["ranges"].get(main_tid)
    steps = [it for it in (rg.items if rg else []) if it[2].startswith("vleo::dsmc_step(")]
    t_lo = steps[0][0] if steps else K[0][0]
    t_hi = steps[-1][1] if steps else K[-1][1]
    loop = [(s, e, n) for s, e, n in K if s >= t_lo and e <= t_hi]
    ktot = sum(e - s for s, e, _ in loop)
    busy = union_length([(s, e) for s, e, _ in loop])
    span = t_hi - t_lo
    md.append("## GPU (rocprofv3), step loop only\n")
    md.append(f"- steps traced: {len(steps)}, loop span {span / 1e9:.2f} s, "
              f"{span / max(1, len(steps)) / 1e6:.2f} ms per step")
    md.append(f"- GPU busy (union of kernel intervals): {100 * busy / span:.1f} % of the loop, "
              f"idle {(span - busy) / 1e9:.2f} s = {(span - busy) / max(1, len(steps)) / 1e3:.0f} µs per step")
    g = gaps([(s, e) for s, e, _ in loop]) / 1e3
    if len(g):
        md.append(f"- gaps between kernels: {len(g)} (median {np.median(g):.1f} µs, "
                  f"{np.sum(g > 100)} longer than 100 µs, together {np.sum(g[g > 100]) / 1e3:.0f} ms)")
    agg = defaultdict(lambda: [0, 0])
    for s, e, n in loop:
        agg[n][0] += 1; agg[n][1] += e - s
    md.append("\n| kernel (Kokkos label) | group | calls | total [ms] | mean [µs] | share |\n|---|---|---:|---:|---:|---:|")
    for n, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
        md.append(f"| `{n}` | {group_of(n)} | {c} | {t / 1e6:.1f} | {t / c / 1e3:.1f} | {100 * t / ktot:.1f} % |")
    gagg = defaultdict(int)
    for n, (c, t) in agg.items():
        gagg[group_of(n)] += t
    md.append("\n| group | kernel time [ms] | share | per step [ms] |\n|---|---:|---:|---:|")
    for gname, t in sorted(gagg.items(), key=lambda x: -x[1]):
        md.append(f"| {gname} | {t / 1e6:.1f} | {100 * t / ktot:.1f} % | {t / max(1, len(steps)) / 1e6:.2f} |")
    if R["api"]:
        a = defaultdict(lambda: [0, 0])
        for s, e, fn, tid in R["api"]:
            if s >= t_lo and e <= t_hi:
                a[fn][0] += 1; a[fn][1] += e - s
        md.append("\n## CPU in HIP API calls (step loop)\n")
        md.append("Time the host thread spends inside HIP calls; synchronising calls "
                  "(Synchronize, blocking Memcpy) = the CPU waiting for the GPU.\n")
        md.append("| HIP call | calls | total [ms] | mean [µs] | waits for GPU |\n|---|---:|---:|---:|:---:|")
        for fn, (c, t) in sorted(a.items(), key=lambda x: -x[1][1])[:15]:
            md.append(f"| `{fn}` | {c} | {t / 1e6:.1f} | {t / c / 1e3:.1f} | {'x' if SYNC_CALLS.search(fn) else ''} |")
    if R["copies"]:
        c = [(s, e, d, b) for s, e, d, b in R["copies"] if s >= t_lo and e <= t_hi]
        dirs = defaultdict(lambda: [0, 0, 0])
        for s, e, d, b in c:
            dirs[d][0] += 1; dirs[d][1] += e - s; dirs[d][2] += b
        md.append("\n| memory copy | count | total [ms] | MB |\n|---|---:|---:|---:|")
        for d, (n, t, b) in dirs.items():
            md.append(f"| {d} | {n} | {t / 1e6:.2f} | {b / 1e6:.1f} |")
    if rg:
        inc = defaultdict(lambda: [0, 0])
        for s, e, n in rg.items:
            inc[n][0] += 1; inc[n][1] += e - s
        md.append("\n## Host ranges (Kokkos regions and kernel launches, not fenced)\n")
        md.append("Inclusive host time; asynchronous launches return immediately, so the time of a phase "
                  "shows up where the host next waits (reductions, copies).\n")
        md.append("| range | count | total [ms] | mean [µs] |\n|---|---:|---:|---:|")
        for n, (c, t) in sorted(inc.items(), key=lambda x: -x[1][1])[:30]:
            md.append(f"| `{n}` | {c} | {t / 1e6:.1f} | {t / c / 1e3:.1f} |")

    # timelines
    figs = []
    t0 = K[0][0]
    if steps:
        samp = [s for s in steps if "sampling" in s[2]] or steps
        mid = samp[len(samp) // 2]
        i = steps.index(mid)
        win = (steps[i][0], steps[min(i + 2, len(steps) - 1)][1])
        figs.append(("timeline_steps.png", win, "3 consecutive time steps (sampling phase)"))
        so = [it for it in rg.items if it[2] == "vleo::dsmc_state_output"]
        if so:
            s = so[len(so) // 2]
            enclosing = [st for st in steps if st[0] <= s[0] <= st[1]]
            lo = enclosing[0][0] if enclosing else s[0]
            figs.append(("timeline_state_output.png", (lo, s[1] + (s[1] - lo) * 0.05),
                         "time step with cell-state output (moments copy + HDF5 write on the host)"))
    figs.append(("timeline_run.png", (K[0][0], K[-1][1]), "whole run"))
    for fname, (lo, hi), title in figs:
        timeline(R, rg, lo, hi, t0, os.path.join(figdir, fname), title, overview=fname == "timeline_run.png")
        md.append(f"\n![{title}]({fname})")


def timeline(R, rg, lo, hi, t0, png, title, overview=False):
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    sc = 1e-6 if (hi - lo) < 5e9 else 1e-9
    unit = "ms" if sc == 1e-6 else "s"
    x = lambda t: (t - t0) * sc
    rows = []
    sel = lambda items: [it for it in items if it[1] >= lo and it[0] <= hi]
    y = 0
    if rg:
        depth_items = defaultdict(list)
        for it, d in zip(rg.items, rg.depth):
            if it[1] >= lo and it[0] <= hi:
                depth_items[d].append(it)
        for d in sorted(depth_items):
            if overview and d > 1:
                continue
            for s, e, n in depth_items[d]:
                ax.broken_barh([(x(s), max(x(e) - x(s), 1e-9))], (y, 0.8), facecolor=COLORS[group_of(n)],
                               edgecolor="k" if d == 0 else "none", lw=0.3)
                if (x(e) - x(s)) > 0.06 * (x(hi) - x(lo)):
                    ax.text(x(s) + 0.002 * (x(hi) - x(lo)), y + 0.4, n.replace("vleo::", ""), fontsize=6,
                            va="center", clip_on=True)
            rows.append({0: "CPU: step / stage", 1: "CPU: phase", 2: "CPU: kernel launch"}.get(d, f"CPU: level {d}"))
            y += 1
    if R["api"]:
        for s, e, fn, tid in sel(R["api"]):
            c = "#d62728" if SYNC_CALLS.search(fn) else "0.55"
            ax.broken_barh([(x(s), max(x(e) - x(s), 1e-9))], (y, 0.8), facecolor=c, edgecolor="none")
        rows.append("CPU HIP API"); y += 1
    for s, e, n in sel(R["kernels"]):
        ax.broken_barh([(x(s), max(x(e) - x(s), 1e-9))], (y, 0.8), facecolor=COLORS[group_of(n)], edgecolor="none")
    rows.append("GPU kernels"); y += 1
    if R["copies"]:
        for s, e, d, b in sel(R["copies"]):
            ax.broken_barh([(x(s), max(x(e) - x(s), 1e-9))], (y, 0.8), facecolor="#000000", edgecolor="none")
        rows.append("GPU copies"); y += 1
    ax.set_yticks(np.arange(len(rows)) + 0.4); ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlim(x(lo), x(hi)); ax.set_xlabel(f"time [{unit}] since the first kernel")
    handles = [Patch(color=c, label=g) for g, c in COLORS.items()]
    handles += [Patch(color="#d62728", label="HIP call waiting for GPU"), Patch(color="0.55", label="other HIP call")]
    ax.legend(handles=handles, fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title(title, fontsize=10)
    fig.savefig(png, dpi=120); plt.close(fig)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    md = [f"# Profile of `{os.path.basename(os.path.abspath(root))}`\n"]
    for mon in sorted(glob.glob(os.path.join(root, "*", "monitor.csv"))):
        mode = os.path.basename(os.path.dirname(mon))
        png = f"monitor_{mode}.png"
        st = monitor_plot(mon, os.path.join(root, png), f"{mode}: GPU and CPU load of vleo_solver")
        if st:
            md.append(f"## Monitor ({mode})\n")
            md.append(", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}" for k, v in st.items()))
            md.append(f"\n![monitor]({png})\n")
    res = os.path.join(root, "phases", "run", "results.json")
    if os.path.exists(res):
        d = json.load(open(res))["diagnostics"]
        p = d.get("profile", {})
        steps = d.get("steps") or 1
        # leaf phases of the step (dsmc.migrate contains dsmc.migrate.*)
        leaf = {k for k in p if k.startswith("dsmc.") and not any(o.startswith(k + ".") for o in p)}
        tot = sum(p[k]["seconds"] for k in leaf)
        md.append("## Fenced phase times (--profile)\n")
        md.append(f"particles mean {d.get('particles_mean', 0):.4g}, wall {d.get('wall_time_s', 0):.1f} s, "
                  f"{d.get('particle_steps_per_s', 0):.3g} particle steps/s "
                  f"({1e9 / max(d.get('particle_steps_per_s', 1), 1):.2f} ns per particle step)\n")
        md.append("| phase | calls | total [s] | per step [ms] | share of step phases |\n|---|---:|---:|---:|---:|")
        for k, v in sorted(p.items(), key=lambda x: -x[1]["seconds"]):
            share = f"{100 * v['seconds'] / tot:.1f} %" if k in leaf else ""
            md.append(f"| `{k}` | {v['calls']} | {v['seconds']:.2f} | {1e3 * v['seconds'] / steps:.2f} | {share} |")
        md.append("")
    R = load_rocprof(os.path.join(root, "rocprof"))
    if R:
        rocprof_summary(R, md, root)
        pf = find(os.path.join(root, "rocprof"), "*.pftrace")
        if pf:
            md.append(f"\nPerfetto trace: `{os.path.relpath(pf[0], root)}` (open in https://ui.perfetto.dev)")
    for pat in ("sys/trace/**/*.proto", "sys/trace/**/*.pftrace"):
        for p in glob.glob(os.path.join(root, pat), recursive=True)[:1]:
            md.append(f"\nrocprof-sys Perfetto trace: `{os.path.relpath(p, root)}`")
    if os.path.exists(os.path.join(root, "perf", "perf_report.txt")):
        md.append("\nperf: `perf/perf_report.txt` (CPU hot spots), `perf/perf_threads.txt` (per thread)")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", os.path.join(root, "summary.md"))


if __name__ == "__main__":
    main()
