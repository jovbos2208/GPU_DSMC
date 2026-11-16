#!/usr/bin/env python3
"""Performance benchmarks (M5.1, M5.2, M5.4).

Runs the benchmark cases with --profile and records, per case: wall time,
throughput (TPMC histories/s, DSMC particle steps/s), the phase profile and the
efficiency  E = SE(C_D)^2 * wall time  (time needed per unit of statistical
accuracy; lower is better, independent of the run length).

  benchmark.py run <vleo_solver> <label> [--cases B1 B2 ...]   -> benchmarks/results/<label>.json
  benchmark.py compare <before.json> <after.json>                -> speed-up table

Several GPUs (M6.7, MPI builds): --ranks N [--mpiexec mpiexec] [--mpiexec-flags "..."]
starts the solver with N ranks; --scale K multiplies the problem size (DSMC
particles per cell, TPMC histories) for weak scaling, e.g.
  benchmark.py run <solver> w7700-np1 --cases B3 B4
  benchmark.py run <solver> w7700-np2 --cases B3 B4 --ranks 2           (strong)
  benchmark.py run <solver> w7700-np2-weak --cases B3 B4 --ranks 2 --scale 2
"""
import re
import argparse
import json
import math
import pathlib
import platform
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"


def sphere_stl(path, n_theta=128, n_phi=256, r=0.5):
    pts = lambda it, ip: (
        (0.0, 0.0, r) if it == 0 else (0.0, 0.0, -r) if it == n_theta else
        (r * math.sin(math.pi * it / n_theta) * math.cos(2 * math.pi * (ip % n_phi) / n_phi),
         r * math.sin(math.pi * it / n_theta) * math.sin(2 * math.pi * (ip % n_phi) / n_phi),
         r * math.cos(math.pi * it / n_theta)))
    with open(path, "w") as f:
        f.write("solid sphere\n")
        for it in range(n_theta):
            for ip in range(n_phi):
                a, b, c, d = pts(it, ip), pts(it + 1, ip), pts(it + 1, ip + 1), pts(it, ip + 1)
                tris = []
                if it > 0:
                    tris.append((a, b, d))
                if it < n_theta - 1:
                    tris.append((d, b, c))
                for t in tris:
                    f.write("facet normal 0 0 0\nouter loop\n")
                    for p in t:
                        f.write("vertex %r %r %r\n" % p)
                    f.write("endloop\nendfacet\n")
        f.write("endsolid sphere\n")


SAT = str(ROOT / "examples" / "geometry" / "satellite.stl")
COMMON = ("reference: {area_m2: 1.0, length_m: 1.0, moment_center_m: [0, 0, 0], lift_axis_body: [0, 0, 1]}\n")


def cases(tmp):
    sphere = tmp / "sphere.stl"
    sphere_stl(sphere)
    return {
        "B1": ("TPMC, satellite, VLEO air (O, N2, O2, N, He), CLL", f"""simulation: {{name: B1, mode: tpmc, random_seed: 1}}
geometry: {{file: {SAT}}}
{COMMON}gas: {{number_density_m3: 1.0e14, translational_temperature_K: 900, bulk_velocity_body_mps: [-7800, 0, 0],
      species: {{O: 0.6, N2: 0.3, O2: 0.02, N: 0.03, He: 0.05}}}}
surface: {{model: cll, temperature_K: 300, cll: {{alpha_n: 0.85, sigma_t: 0.65}}}}
tpmc: {{particle_histories: 20000000}}
"""),
        "B2": ("TPMC, sphere with 65 024 triangles (BVH load), Maxwell", f"""simulation: {{name: B2, mode: tpmc, random_seed: 2}}
geometry: {{file: {sphere}}}
{COMMON}gas: {{number_density_m3: 1.0e14, translational_temperature_K: 900, bulk_velocity_body_mps: [-7800, 300, -200],
      species: {{O: 1.0}}}}
surface: {{model: maxwell, temperature_K: 300, maxwell: {{diffuse_fraction: 0.9}}}}
tpmc: {{particle_histories: 10000000}}
"""),
        "B3": ("DSMC, 0.5 m cube, He, Kn ~ 1, dx = 0.05 m, 30 particles/cell", f"""simulation: {{name: B3, mode: dsmc, random_seed: 3}}
geometry: {{file: {tmp / 'cube.stl'}}}
domain: {{margin_m: 0.15}}
{COMMON}gas: {{number_density_m3: 1.0e19, translational_temperature_K: 900, bulk_velocity_body_mps: [-7500, 0, 0],
      species: {{He: 1.0}}}}
surface: {{model: maxwell, temperature_K: 300, maxwell: {{diffuse_fraction: 1.0}}}}
dsmc: {{cell_size_mode: fixed, cell_size_m: 0.05, timestep_mode: fixed, timestep_s: 2.0e-6,
       particles_per_cell_target: 30, warmup_steps: 200, sample_steps: 1000, blocks: 20}}
"""),
        "B4": ("DSMC, satellite, air with N2/O2 (rotation, vibration), Kn ~ 0.5", f"""simulation: {{name: B4, mode: dsmc, random_seed: 4}}
geometry: {{file: {SAT}}}
domain: {{margin_m: 0.2}}
{COMMON}gas: {{number_density_m3: 2.0e19, translational_temperature_K: 900, bulk_velocity_body_mps: [-7800, 0, 0],
      species: {{O: 0.5, N2: 0.4, O2: 0.1}}}}
surface: {{model: cll, temperature_K: 300, cll: {{alpha_n: 0.85, sigma_t: 0.65}}}}
dsmc: {{cell_size_mode: fixed, cell_size_m: 0.1, particles_per_cell_target: 20, warmup_steps: 200,
       sample_steps: 600, blocks: 20}}
"""),
        # M6.7: large enough to fill several GPUs (B4 with 8x the cells, ~3 M particles); not in the default set
        "B5": ("DSMC, satellite, air with N2/O2, dx = 0.05 m (~3 M particles, multi-GPU)", f"""simulation: {{name: B5, mode: dsmc, random_seed: 5}}
geometry: {{file: {SAT}}}
domain: {{margin_m: 0.2}}
{COMMON}gas: {{number_density_m3: 2.0e19, translational_temperature_K: 900, bulk_velocity_body_mps: [-7800, 0, 0],
      species: {{O: 0.5, N2: 0.4, O2: 0.1}}}}
surface: {{model: cll, temperature_K: 300, cll: {{alpha_n: 0.85, sigma_t: 0.65}}}}
dsmc: {{cell_size_mode: fixed, cell_size_m: 0.05, particles_per_cell_target: 20, warmup_steps: 100,
       sample_steps: 200, blocks: 20}}
"""),
    }

DEFAULT_CASES = ["B1", "B2", "B3", "B4"]


def cube_stl(path, h=0.25):
    v = lambda i, j, k: ((h if i else -h), (h if j else -h), (h if k else -h))
    quads = [[(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
             [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
             [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]]
    with open(path, "w") as f:
        f.write("solid cube\n")
        for q in quads:
            a, b, c, d = (v(*p) for p in q)
            for t in ((a, b, c), (a, c, d)):
                f.write("facet normal 0 0 0\nouter loop\n" + "".join("vertex %r %r %r\n" % p for p in t) +
                        "endloop\nendfacet\n")
        f.write("endsolid cube\n")


def run_cmd(args):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vleo_bench_"))
    cube_stl(tmp / "cube.stl")
    all_cases = cases(tmp)
    selected = args.cases or DEFAULT_CASES
    version = subprocess.run([args.solver, "--version"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    out = {"label": args.label, "host": platform.node(), "solver_version": version, "ranks": args.ranks,
           "scale": args.scale, "cases": {}}
    for name in selected:
        desc, text = all_cases[name]
        if args.scale != 1:
            text = re.sub(r"(particles_per_cell_target|particle_histories): (\d+)",
                          lambda m: "%s: %d" % (m.group(1), int(m.group(2)) * args.scale), text)
        cfg = tmp / f"{name}.yaml"
        cfg.write_text(text)
        print(f"[{name}] {desc} ...", flush=True)
        walls = []
        for rep in range(args.repeat):
            prof = [] if args.no_profile else ["--profile"]
            launch = [] if args.ranks == 1 and not args.mpiexec_flags and args.mpiexec is None else \
                [args.mpiexec or "mpiexec", *shlex.split(args.mpiexec_flags), "-n", str(args.ranks)]
            subprocess.run([*launch, args.solver, "-q", *prof, "-c", str(cfg), "-o", str(tmp / name)], check=True,
                           stderr=subprocess.DEVNULL)
            r = json.loads((tmp / name / "results.json").read_text())
            walls.append(r["diagnostics"]["wall_time_s"])
        d = r["diagnostics"]
        walls.sort()
        wall = walls[len(walls) // 2]  # median of the repetitions
        se = r["std_error"]["C_D"]
        entry = {"description": desc, "mode": r["simulation"]["mode"], "wall_time_s": wall,
                 "wall_times_s": walls,
                 "C_D": r["results"]["C_D"], "C_D_std_error": se, "heat_rate_W": r["results"]["heat_rate_W"],
                 "efficiency_se2_time": se * se * wall, "profile": d.get("profile", {}),
                 "parallel": d.get("parallel", {})}
        if entry["mode"] == "tpmc":
            entry["throughput"] = d["histories_per_s"] * d["wall_time_s"] / wall
            entry["throughput_unit"] = "histories/s"
        else:
            # solver versions before M5 do not report particle steps
            entry["throughput"] = d["particle_steps_per_s"] * d["wall_time_s"] / wall if "particle_steps_per_s" in d \
                else None
            entry["throughput_unit"] = "particle steps/s"
        out["cases"][name] = entry
        tp = f"{entry['throughput']:.4g}" if entry["throughput"] else "n/a"
        print(f"    {wall:.2f} s, {tp} {entry['throughput_unit']}, C_D = {entry['C_D']:.5f} ± {se:.2g}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    for name, e in out["cases"].items():
        tot = sum(p["seconds"] for p in e["profile"].values()) or 1.0
        top = sorted(e["profile"].items(), key=lambda kv: -kv[1]["seconds"])[:5]
        print(f"  {name} profile: " + ", ".join(f"{k} {100 * v['seconds'] / tot:.0f} %" for k, v in top))


def compare_cmd(args):
    a, b = (json.loads(pathlib.Path(p).read_text()) for p in (args.before, args.after))
    print(f"| case | wall {a['label']} [s] | wall {b['label']} [s] | throughput after | speed-up | C_D before | C_D after |")
    print("| ---- | ---- | ---- | ---- | -------- | ---------- | --------- |")
    for name in a["cases"]:
        if name not in b["cases"]:
            continue
        x, y = a["cases"][name], b["cases"][name]
        z = abs(x["C_D"] - y["C_D"]) / math.hypot(x["C_D_std_error"], y["C_D_std_error"])
        tp = f"{y['throughput']:.4g} {y['throughput_unit']}" if y.get("throughput") else "n/a"
        # same case and run length: the speed-up is the ratio of the (median) wall times
        print(f"| {name} | {x['wall_time_s']:.2f} | {y['wall_time_s']:.2f} | {tp} | "
              f"{x['wall_time_s'] / y['wall_time_s']:.2f}x | {x['C_D']:.5f} | {y['C_D']:.5f} (z = {z:.1f}) |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")  # Python 3.6 compatible (no required=)
    r = sub.add_parser("run")
    r.add_argument("solver")
    r.add_argument("label")
    r.add_argument("--cases", nargs="+")
    r.add_argument("--repeat", type=int, default=3, help="repetitions per case (median is reported)")
    r.add_argument("--no-profile", action="store_true", help="for solver versions without --profile (before M5)")
    r.add_argument("--ranks", type=int, default=1, help="MPI ranks (one per GPU)")
    r.add_argument("--mpiexec", help="MPI launcher (default: mpiexec when --ranks > 1)")
    r.add_argument("--mpiexec-flags", default="", help="extra launcher flags")
    r.add_argument("--scale", type=int, default=1, help="problem size factor (weak scaling)")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = ap.parse_args()
    if args.cmd is None:
        ap.error("a command is required: run or compare")
    return run_cmd(args) if args.cmd == "run" else compare_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
