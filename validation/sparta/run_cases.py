#!/usr/bin/env python3
"""SPARTA comparison cases S1-S4 (implementation plan M4.10-M4.14).

For every case this script writes identical inputs for both codes (geometry,
species data from this solver's database, numerical parameters), runs
vleo_solver and SPARTA, evaluates both with block statistics, applies the
acceptance limits fixed beforehand in docs/sparta.md and archives everything in
validation/sparta/<case>/ (inputs, logs, results, report, versions).

Usage:
  run_cases.py --vleo <vleo_solver> --sparta <spa_serial> [--cases S1 S2 S3 S4] [--quick]
--quick runs ~10x fewer steps (smoke test of the pipeline, statistics too weak
for the acceptance decision).
"""
import argparse
import datetime
import json
import math
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AMU = 1.66053906892e-27
KB = 1.380649e-23

# Species data of this solver (src/core/Species.cpp, docs/species.md).
SPECIES = {
    #        M [g/mol], d_vhs,    omega, T_ref, rot_dof, Z_rot, vib_dof, theta_v, Z_vib
    "O":  (15.999, 3.0e-10, 0.80, 273.15, 0, 0.0, 0, 0.0, 0.0),
    "Ar": (39.948, 4.17e-10, 0.81, 273.0, 0, 0.0, 0, 0.0, 0.0),
    "N2": (28.014, 4.17e-10, 0.74, 273.0, 2, 5.0, 2, 3371.0, 1.0 / 1.90114e-5),
}

# Body and domain: 0.5 m cube centred at the origin, domain [-0.4, 0.4]^3.
HALF, BOX = 0.25, 0.4
VSTREAM = (-7500.0, 500.0, -800.0)  # generic direction: all force components non-zero

CASES = {
    # name: (gas, n, T, wall, collisions, acceptance systematic, description)
    "S1": dict(gas="O", n=1e14, T=900.0, wall=("diffuse", 1.0), dsmc=False, sys=0.005,
               desc="free-molecular cube, O, Maxwell diffuse"),
    "S2": dict(gas="O", n=1e14, T=900.0, wall=("cll", 0.85, 0.65), dsmc=False, sys=0.005,
               desc="free-molecular cube, O, CLL alpha_n = 0.85, sigma_t = 0.65"),
    "S3": dict(gas="Ar", n=3e18, T=900.0, wall=("diffuse", 1.0), dsmc=True, sys=0.02,
               desc="transitional cube (Kn ~ 1), Ar, VHS, Maxwell diffuse"),
    "S4": dict(gas="N2", n=3e18, T=900.0, wall=("diffuse", 1.0), dsmc=True, sys=0.03,
               desc="transitional cube (Kn ~ 1), N2 with rotation (Z_rot = 5), VHS, Maxwell diffuse"),
}
TW = 300.0
QUANTITIES = ["F_x", "F_y", "F_z", "Q"]


def numerics(case, quick):
    """Identical numerical parameters for both codes."""
    T, n = case["T"], case["n"]
    m = SPECIES[case["gas"]][0] * AMU
    c_mp = math.sqrt(2 * KB * T / m)
    v = math.sqrt(sum(x * x for x in VSTREAM))
    cells = 16 if not case["dsmc"] else 13         # DSMC: dx = 0.8/13 = 0.062 m ~ lambda/10
    dx = 2 * BOX / cells
    dt = dx / (4.0 * (v + 3.0 * c_mp))
    ppc = 20
    fnum = n * dx ** 3 / ppc
    block = 100 if quick else 500
    warm_blocks = 4 if quick else 8                # warm-up: 8 x 500 steps (> 3 flow-through times)
    blocks = 6 if quick else 40
    return dict(cells=cells, dx=dx, dt=dt, fnum=fnum, ppc=ppc, block=block,
                warmup=warm_blocks * block, sample=blocks * block, blocks=blocks)


def cube_triangles():
    h = HALF
    v = lambda i, j, k: ((h if i else -h), (h if j else -h), (h if k else -h))
    quads = [[(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
             [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
             [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]]
    tris = []
    for q in quads:
        a, b, c, d = (v(*p) for p in q)
        tris += [(a, b, c), (a, c, d)]
    return tris


def write_stl(path, tris):
    with open(path, "w") as f:
        f.write("solid cube\n")
        for t in tris:
            f.write("facet normal 0 0 0\nouter loop\n")
            for p in t:
                f.write("vertex %r %r %r\n" % p)
            f.write("endloop\nendfacet\n")
        f.write("endsolid cube\n")


def write_sparta_surf(path, tris):
    pts, index = [], {}
    tri_idx = []
    for t in tris:
        ids = []
        for p in t:
            if p not in index:
                index[p] = len(pts) + 1
                pts.append(p)
            ids.append(index[p])
        tri_idx.append(ids)
    with open(path, "w") as f:
        f.write("# 0.5 m cube, outward normals (counter-clockwise from outside)\n\n")
        f.write(f"{len(pts)} points\n{len(tri_idx)} triangles\n\nPoints\n\n")
        for i, p in enumerate(pts, 1):
            f.write(f"{i} {p[0]!r} {p[1]!r} {p[2]!r}\n")
        f.write("\nTriangles\n\n")
        for i, t in enumerate(tri_idx, 1):
            f.write(f"{i} {t[0]} {t[1]} {t[2]}\n")


def sparta_species_files(d, gas):
    M, dvhs, omega, tref, rot, zrot, vib, theta, zvib = SPECIES[gas]
    (d / "case.species").write_text(
        "# species data written by run_cases.py from this solver's database\n"
        "# ID molwt mass rotdof rotrel vibdof vibrel vibtemp specwt charge\n"
        f"{gas} {M} {M * AMU!r} {rot} {1.0 / zrot if zrot else 0.0!r} {vib} "
        f"{1.0 / zvib if zvib else 0.0!r} {theta} 1.0 0.0\n")
    (d / "case.vss").write_text(
        "# VHS parameters (VSS alpha = 1) written by run_cases.py\n"
        f"{gas} {dvhs!r} {omega} {tref} 1.0\n")


def sparta_input(case, num):
    wall = case["wall"]
    if wall[0] == "diffuse":
        collide_surf = f"surf_collide 1 diffuse {TW} {wall[1]}"
    else:  # cll: acc_t = sigma_t (2 - sigma_t)  (docs/sparta.md)
        an, st = wall[1], wall[2]
        collide_surf = f"surf_collide 1 cll {TW} {an} {st * (2 - st)!r} {an} {an}"
    lines = [
        "# generated by validation/sparta/run_cases.py",
        "seed 12345",
        "dimension 3",
        "global gridcut 0.0 comm/sort yes",
        "boundary o o o",
        f"create_box {-BOX} {BOX} {-BOX} {BOX} {-BOX} {BOX}",
        f"create_grid {num['cells']} {num['cells']} {num['cells']}",
        "balance_grid rcb cell",
        f"global nrho {case['n']!r} fnum {num['fnum']!r}",
        f"species case.species {case['gas']}",
        f"mixture gas {case['gas']} vstream {VSTREAM[0]} {VSTREAM[1]} {VSTREAM[2]} temp {case['T']}",
        "read_surf cube.surf",
        collide_surf,
        "surf_modify all collide 1",
    ]
    if case["dsmc"]:
        lines += ["collide vss gas case.vss",
                  "collide_modify vremax 100 yes rotate smooth vibrate discrete"]
    lines += [
        "fix in emit/face gas all",
        "create_particles gas n 0",
        f"timestep {num['dt']!r}",
        f"stats {num['block']}",
        "stats_style step cpu np",
        f"run {num['warmup']}",
        "compute 1 surf all gas fx fy fz etot norm flow",
        f"fix 1 ave/surf all 1 {num['block']} {num['block']} c_1[*]",
        "compute 2 reduce sum f_1[1] f_1[2] f_1[3] f_1[4]",
        "stats_style step cpu np c_2[1] c_2[2] c_2[3] c_2[4]",
        f"run {num['sample']}",
    ]
    return "\n".join(lines) + "\n"


def vleo_config(case, num, name):
    wall = case["wall"]
    if wall[0] == "diffuse":
        surf = f"{{model: maxwell, temperature_K: {TW}, maxwell: {{diffuse_fraction: {wall[1]}}}}}"
    else:
        surf = (f"{{model: cll, temperature_K: {TW}, cll: {{alpha_n: {wall[1]}, sigma_t: {wall[2]}, "
                f"alpha_rot: {wall[1]}, alpha_vib: {wall[1]}}}}}")
    mode = "dsmc" if case["dsmc"] else "tpmc"
    text = (f"simulation: {{name: {name}, mode: {mode}, random_seed: 2026}}\n"
            "geometry: {file: cube.stl}\n"
            f"domain: {{box_min_m: [{-BOX}, {-BOX}, {-BOX}], box_max_m: [{BOX}, {BOX}, {BOX}]}}\n"
            "reference: {area_m2: 0.25, length_m: 0.5, moment_center_m: [0, 0, 0], lift_axis_body: [0, 0, 1]}\n"
            f"gas: {{number_density_m3: {case['n']!r}, translational_temperature_K: {case['T']}, "
            f"bulk_velocity_body_mps: [{VSTREAM[0]}, {VSTREAM[1]}, {VSTREAM[2]}], species: {{{case['gas']}: 1.0}}}}\n"
            f"surface: {surf}\n"
            "tpmc: {particle_histories: 20000000}\n"
            f"dsmc: {{collision_model: VHS, cell_size_mode: fixed, cell_size_m: {num['dx']!r}, "
            f"timestep_mode: fixed, timestep_s: {num['dt']!r}, particles_per_cell_target: {num['ppc']}, "
            f"warmup_steps: {num['warmup']}, sample_steps: {num['sample']}, blocks: {num['blocks']}}}\n")
    return text


def parse_sparta_log(path):
    """Block averages of (F_x, F_y, F_z, Q) from the sampling run."""
    rows, in_sample = [], False
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("Step") and "c_2[1]" in line:
            in_sample, first = True, True
            continue
        if in_sample:
            if line.startswith("Loop time"):
                break
            parts = line.split()
            if len(parts) == 7 and re.match(r"^\d+$", parts[0]):
                if first:  # values at the start of the run carry no samples
                    first = False
                    continue
                rows.append([float(x) for x in parts[3:7]])
    return rows


def mean_se(values):
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n)


def run(cmd, cwd, log):
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)


def version(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vleo", required=True)
    ap.add_argument("--sparta", required=True)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    vleo = str(pathlib.Path(args.vleo).resolve())
    sparta = str(pathlib.Path(args.sparta).resolve())

    summary = []
    for name in args.cases:
        case = CASES[name]
        num = numerics(case, args.quick)
        d = HERE / name
        d.mkdir(exist_ok=True)
        tris = cube_triangles()
        write_stl(d / "cube.stl", tris)
        write_sparta_surf(d / "cube.surf", tris)
        sparta_species_files(d, case["gas"])
        (d / "in.case").write_text(sparta_input(case, num))
        (d / "vleo.yaml").write_text(vleo_config(case, num, name))
        print(f"[{name}] {case['desc']}: running vleo_solver ...", flush=True)
        run([vleo, "-q", "-c", "vleo.yaml", "-o", "vleo_out"], d, d / "vleo.log")
        print(f"[{name}] running SPARTA ...", flush=True)
        run([sparta, "-in", "in.case", "-log", "log.sparta", "-echo", "none", "-screen", "none"], d, d / "sparta.stdout")

        r = json.loads((d / "vleo_out" / "results.json").read_text())
        ours = {"F_x": (r["results"]["force_N"][0], r["std_error"]["force_N"][0]),
                "F_y": (r["results"]["force_N"][1], r["std_error"]["force_N"][1]),
                "F_z": (r["results"]["force_N"][2], r["std_error"]["force_N"][2]),
                "Q": (r["results"]["heat_rate_W"], r["std_error"]["heat_rate_W"])}
        blocks = parse_sparta_log(d / "log.sparta")
        if len(blocks) < 2:
            raise SystemExit(f"[{name}] could not read SPARTA block averages from log.sparta")
        theirs = {q: mean_se([b[i] for b in blocks]) for i, q in enumerate(QUANTITIES)}

        rows, ok = [], True
        for q in QUANTITIES:
            (a, sa), (b, sb) = ours[q], theirs[q]
            comb = math.hypot(sa, sb)
            limit = 3.0 * comb + case["sys"] * max(abs(a), abs(b))
            passed = abs(a - b) <= limit
            ok = ok and passed
            rows.append(dict(quantity=q, vleo=a, vleo_se=sa, sparta=b, sparta_se=sb, diff=a - b,
                             rel=(a - b) / b if b else float("nan"), z=(a - b) / comb if comb else float("nan"),
                             limit=limit, passed=passed))
        report = {
            "case": name, "description": case["desc"], "date": datetime.date.today().isoformat(),
            "quick": args.quick, "numerics": num, "acceptance": f"|diff| <= 3 sigma_comb + {case['sys']:.1%}",
            "versions": {"vleo_solver": version([vleo, "--version"]),
                         "sparta": "SPARTA 27Aug2026 (git 95b9abaa8bd548991cc3c3f1c58b34722f7ade74), spa_serial"},
            "sparta_blocks": len(blocks), "rows": rows, "passed": ok}
        (d / "comparison.json").write_text(json.dumps(report, indent=2))
        md = [f"# {name}: {case['desc']}", "",
              f"Acceptance (fixed in docs/sparta.md): {report['acceptance']}" + ("  — QUICK RUN, not a decision" if args.quick else ""), "",
              "| quantity | vleo_solver | SPARTA | difference | z | limit | result |",
              "| -------- | ----------- | ------ | ---------- | - | ----- | ------ |"]
        for x in rows:
            md.append(f"| {x['quantity']} | {x['vleo']:.6g} ± {x['vleo_se']:.2g} | {x['sparta']:.6g} ± {x['sparta_se']:.2g} | "
                      f"{x['diff']:.3g} ({100 * x['rel']:+.2f} %) | {x['z']:.2f} | {x['limit']:.3g} | {'pass' if x['passed'] else 'FAIL'} |")
        md += ["", f"Overall: **{'PASS' if ok else 'FAIL'}**"]
        (d / "report.md").write_text("\n".join(md) + "\n")
        print("\n".join(md), flush=True)
        summary.append((name, ok))

    print("\nSummary: " + ", ".join(f"{n} {'PASS' if ok else 'FAIL'}" for n, ok in summary))
    return 0 if all(ok for _, ok in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
