#!/usr/bin/env python3
"""GOCE and CHAMP: vleo_solver against SPARTA with identical inputs (M6.10, M6.11).

No published coefficient tables are available, so the reference is SPARTA
(27Aug2026, git 95b9aba) run with exactly the same geometry (March et al.
2019 models, examples/geometry/*_march2019.stl), gas, wall model, domain
and - for DSMC - cell size, time step and particle weight.

Free-molecular cases (TPMC vs. SPARTA without collisions) are run over an
attitude sweep (scripts/attitude_sweep.py convention: flight direction
d = (cos a cos b, sin b, sin a cos b), +x = flight direction at a = b = 0).
DSMC cases use the automatic cell size of vleo_solver with pilot runs; SPARTA
then uses the grid chosen by the pilot. In addition vleo_solver runs with the
free-stream cell-size rule (no pilot) and with half the pilot cell size, which
checks the pilot rule on a real geometry (M6.11).

Compared: F_x, F_y, F_z, heat rate Q (block statistics, 20 blocks), and the
moments about the CAD origin computed in both codes from the per-triangle
forces at the triangle centroids (vleo_solver also reports its exact moment).
Acceptance (fixed before the runs, docs/satellites.md):
    free-molecular |diff| <= 3 sigma_comb + 1 % max(|a|, |b|)
    DSMC           |diff| <= 3 sigma_comb + 3 % max(|a|, |b|)
Moments use the vleo_solver standard error of its exact moment as sigma of
both centroid moments.

  run_sat.py --vleo <vleo_solver> --sparta <spa_mpi> [--mpiexec mpiexec --sparta-ranks 32]
             [--cases G-FM-D G-FM-C C-FM-D C-FM-C G-DSMC C-DSMC] [--quick]
Results: validation/satellites/<case>/ (inputs, logs, comparison.json, report.md).
Python 3.6 compatible.
"""
import argparse
import datetime
import json
import math
import pathlib
import re
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "validation" / "sparta"))
import run_cases as rc  # noqa: E402  (species database, SPARTA surface writer)

GEOMETRY = {"GOCE": ROOT / "examples" / "geometry" / "goce_march2019.stl",
            "CHAMP": ROOT / "examples" / "geometry" / "champ_march2019.stl"}
SPEED, T_GAS, TW = 7700.0, 900.0, 300.0
CLL = (0.85, 0.65)
FM_ATT = [(0.0, 0.0), (10.0, 0.0), (30.0, 0.0), (0.0, 15.0)]
FM_ATT_CLL = [(0.0, 0.0), (10.0, 0.0)]
CASES = {
    "G-FM-D": dict(sat="GOCE", dsmc=False, gas={"O": 0.7, "N2": 0.3}, wall="diffuse", att=FM_ATT),
    "G-FM-C": dict(sat="GOCE", dsmc=False, gas={"O": 1.0}, wall="cll", att=FM_ATT_CLL),
    "C-FM-D": dict(sat="CHAMP", dsmc=False, gas={"O": 0.7, "N2": 0.3}, wall="diffuse", att=FM_ATT),
    "C-FM-C": dict(sat="CHAMP", dsmc=False, gas={"O": 1.0}, wall="cll", att=FM_ATT_CLL),
    "G-DSMC": dict(sat="GOCE", dsmc=True, gas={"O": 1.0}, wall="diffuse", att=[(0.0, 0.0)]),
    "C-DSMC": dict(sat="CHAMP", dsmc=True, gas={"O": 1.0}, wall="diffuse", att=[(0.0, 0.0)]),
}
N_FM, N_DSMC = 1.0e15, 3.0e18     # DSMC: lambda_inf ~ 0.8 m (Kn ~ 0.15 on GOCE, 0.1 on CHAMP)
SYS = {False: 0.01, True: 0.03}
BLOCKS = 20


# --- geometry ------------------------------------------------------------------
def read_stl_mm(path):
    pts = [tuple(float(x) / 1000.0 for x in m)
           for m in re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", pathlib.Path(path).read_text())]
    return [tuple(pts[i:i + 3]) for i in range(0, len(pts), 3)]


def centroids(tris):
    return [tuple(sum(p[k] for p in t) / 3.0 for k in range(3)) for t in tris]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def centroid_moment(cents, forces):
    m = [0.0, 0.0, 0.0]
    for c, f in zip(cents, forces):
        x = cross(c, f)
        for k in range(3):
            m[k] += x[k]
    return m


def flight_dir(a_deg, b_deg):
    a, b = math.radians(a_deg), math.radians(b_deg)
    return (math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b))


# --- numerics --------------------------------------------------------------------
def domain_box(tris, margin):
    lo = [min(p[k] for t in tris for p in t) - margin for k in range(3)]
    hi = [max(p[k] for t in tris for p in t) + margin for k in range(3)]
    return lo, hi


def c_mp(gas):
    m_min = min(rc.SPECIES[s][0] for s in gas) * rc.AMU
    return math.sqrt(2 * rc.KB * T_GAS / m_min)


def fm_numerics(lo, hi, gas, quick):
    dx = 0.2
    cells = [max(1, int(round((hi[k] - lo[k]) / dx))) for k in range(3)]
    dt = dx / (4.0 * (SPEED + 3.0 * c_mp(gas)))
    volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    fnum = N_FM * volume / (2.0e5 if quick else 1.0e6)   # simulated particles in the domain
    diag = math.sqrt(sum((hi[k] - lo[k]) ** 2 for k in range(3)))
    flow = diag / SPEED / dt
    block = int(math.ceil(2 * flow / 100.0)) * 100        # block >= 2 flow-through times
    return dict(cells=cells, dt=dt, fnum=fnum, block=block, warmup=2 * block,
                sample=(5 if quick else BLOCKS) * block, histories=2_000_000 if quick else 20_000_000)


# --- SPARTA ----------------------------------------------------------------------
def sparta_files(d, tris, gas, label):
    rc.write_sparta_surf(d / "body.surf", tris)
    surf = (d / "body.surf").read_text().split("\n", 1)[1]  # replace the header comment of the cube writer
    (d / "body.surf").write_text("# %s (March et al. 2019 model, metres), outward normals\n%s" % (label, surf))
    lines_sp = ["# species data written by run_sat.py from vleo_solver's database",
                "# ID molwt mass rotdof rotrel vibdof vibrel vibtemp specwt charge"]
    lines_vss = ["# VHS parameters (VSS alpha = 1) written by run_sat.py"]
    for s in gas:
        M, dvhs, omega, tref, rot, zrot, vib, theta, zvib = rc.SPECIES[s]
        lines_sp.append("%s %r %r %d %r %d %r %r 1.0 0.0" % (s, M, M * rc.AMU, rot, 1.0 / zrot if zrot else 0.0, vib,
                                                         1.0 / zvib if zvib else 0.0, theta))
        lines_vss.append("%s %r %r %r 1.0" % (s, dvhs, omega, tref))
    (d / "case.species").write_text("\n".join(lines_sp) + "\n")
    (d / "case.vss").write_text("\n".join(lines_vss) + "\n")


def sparta_input(case, lo, hi, num, att, n):
    v = [-SPEED * x for x in flight_dir(*att)]
    gas = case["gas"]
    if case["wall"] == "diffuse":
        sc = "surf_collide 1 diffuse %r 1.0" % TW
    else:  # cll: acc_t = sigma_t (2 - sigma_t)  (docs/sparta.md)
        an, st = CLL
        sc = "surf_collide 1 cll %r %r %r %r %r" % (TW, an, st * (2 - st), an, an)
    lines = ["# generated by validation/satellites/run_sat.py", "seed 12345", "dimension 3",
             "global gridcut 0.0 comm/sort yes", "boundary o o o",
             "create_box %r %r %r %r %r %r" % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]),
             "create_grid %d %d %d" % tuple(num["cells"]), "balance_grid rcb cell",
             "global nrho %r fnum %r" % (n, num["fnum"]),
             "species case.species " + " ".join(gas),
             "mixture gas %s vstream %r %r %r temp %r" % (" ".join(gas), v[0], v[1], v[2], T_GAS)]
    lines += ["mixture gas %s frac %r" % (s, x / sum(gas.values())) for s, x in gas.items()]
    lines += ["read_surf body.surf", sc, "surf_modify all collide 1"]
    if case["dsmc"]:
        lines += ["collide vss gas case.vss", "collide_modify vremax 100 yes rotate smooth vibrate discrete"]
    b = num["block"]
    # warm-up rounded up to whole blocks, so that every sampling block is complete
    num["sparta_warmup"] = int(math.ceil(num["warmup"] / float(b))) * b
    lines += ["fix in emit/face gas all", "create_particles gas n 0" if case["dsmc"] else "",
              "timestep %r" % num["dt"], "stats %d" % b, "stats_style step cpu np", "run %d" % num["sparta_warmup"],
              "compute 1 surf all gas fx fy fz etot norm flow",
              "fix 1 ave/surf all 1 %d %d c_1[*]" % (b, b),
              "dump 1 surf all %d dump.surf id f_1[1] f_1[2] f_1[3] f_1[4]" % b,
              "run %d" % num["sample"]]
    return "\n".join(x for x in lines if x) + "\n"


def parse_dump(path, ntri, first_step):
    """Per-block (forces per triangle, Q) from the surf dump."""
    blocks, cur, step, reading = [], None, None, False
    lines = pathlib.Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
            i += 2
            continue
        if lines[i].startswith("ITEM: SURFS"):
            f = [None] * ntri
            q = 0.0
            for k in range(ntri):
                p = lines[i + 1 + k].split()
                f[int(p[0]) - 1] = (float(p[1]), float(p[2]), float(p[3]))
                q += float(p[4])
            if step > first_step:
                blocks.append((f, q))
            i += 1 + ntri
            continue
        i += 1
    return blocks


def mean_se(values):
    n = len(values)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n)


# --- vleo_solver -----------------------------------------------------------------
def vleo_config(case, sat, lo, hi, n, num=None, dsmc_cells=None, pilot=True, name="case"):
    gas = case["gas"]
    if case["wall"] == "diffuse":
        surf = "{model: maxwell, temperature_K: %r, maxwell: {diffuse_fraction: 1.0}}" % TW
    else:
        surf = ("{model: cll, temperature_K: %r, cll: {alpha_n: %r, sigma_t: %r, alpha_rot: %r, alpha_vib: %r}}"
                % (TW, CLL[0], CLL[1], CLL[0], CLL[0]))
    text = ("simulation: {name: %s, mode: %s, random_seed: 2026}\n" % (name, "dsmc" if case["dsmc"] else "tpmc") +
            "geometry: {file: %s, units: mm}\n" % GEOMETRY[sat] +
            "domain: {box_min_m: [%r, %r, %r], box_max_m: [%r, %r, %r]}\n" % tuple(lo + hi) +
            "reference: {area_m2: 1.0, length_m: 1.0, moment_center_m: [0, 0, 0], lift_axis_body: [0, 0, 1]}\n"
            "gas: {number_density_m3: %r, translational_temperature_K: %r, " % (n, T_GAS) +
            "bulk_velocity_body_mps: [%r, 0, 0], species: {%s}}\n" % (
                -SPEED, ", ".join("%s: %r" % kv for kv in gas.items())) +
            "surface: %s\n" % surf +
            "output: {save_surface_fields: true}\n")
    if case["dsmc"]:
        if dsmc_cells is None:
            text += "dsmc: {collision_model: VHS, cell_size_mode: automatic, cell_size_pilot: %s, " % (
                "true" if pilot else "false")
        else:
            text += "dsmc: {collision_model: VHS, cell_size_mode: fixed, cell_size_m: %r, " % dsmc_cells
        text += "particles_per_cell_target: 20, sample_steps: %d, blocks: %d}\n" % (num["sample"], BLOCKS)
    else:
        text += "tpmc: {particle_histories: %d}\n" % num["histories"]
    return text


def read_surface_forces(path):
    rows = list(pathlib.Path(path).read_text().splitlines())
    head = [h.strip() for h in rows[0].split(",")]
    ix = [head.index(c) for c in ("force_x_N", "force_y_N", "force_z_N")]
    idx = head.index("source_index") if "source_index" in head else None
    out = {}
    for k, line in enumerate(rows[1:]):
        p = line.split(",")
        out[int(p[idx]) if idx is not None else k] = tuple(float(p[i]) for i in ix)
    return [out[k] for k in sorted(out)]


def run(cmd, cwd, log):
    print("$ " + " ".join(cmd), flush=True)
    with open(str(log), "w") as f:
        if subprocess.call(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT) != 0:
            raise SystemExit("command failed, see %s" % log)


def compare(ours, theirs, sys_frac):
    rows, ok = [], True
    for q in ours:
        (a, sa), (b, sb) = ours[q], theirs[q]
        comb = math.hypot(sa, sb)
        limit = 3.0 * comb + sys_frac * max(abs(a), abs(b))
        passed = abs(a - b) <= limit
        ok = ok and passed
        rows.append(dict(quantity=q, vleo=a, vleo_se=sa, sparta=b, sparta_se=sb, diff=a - b,
                         rel=(a - b) / b if b else float("nan"), z=(a - b) / comb if comb else float("nan"),
                         limit=limit, passed=passed))
    return rows, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vleo", required=True)
    ap.add_argument("--vleo-mpiexec", default="", help="launcher prefix for vleo_solver, e.g. 'mpiexec -n 2'")
    ap.add_argument("--sparta", required=True)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--sparta-ranks", type=int, default=1)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--quick", action="store_true", help="short runs (pipeline test, no acceptance decision)")
    args = ap.parse_args()
    vleo = shlex.split(args.vleo_mpiexec) + [str(pathlib.Path(args.vleo).resolve())]
    sparta = [args.mpiexec, "-n", str(args.sparta_ranks), str(pathlib.Path(args.sparta).resolve())]

    summary = []
    for name in args.cases:
        case = CASES[name]
        sat = case["sat"]
        tris = read_stl_mm(GEOMETRY[sat])
        cents = centroids(tris)
        lo, hi = domain_box(tris, 1.0 if case["dsmc"] else 0.3)
        n = N_DSMC if case["dsmc"] else N_FM
        d = HERE / name
        d.mkdir(exist_ok=True)
        results = []
        extra_runs = {}
        for att in case["att"]:
            tag = "a%+g_b%+g" % att
            w = d / tag
            w.mkdir(exist_ok=True)
            att_args = ["--attitude", repr(att[0]), repr(att[1])]
            if case["dsmc"]:
                # vleo first: its pilot run chooses the grid that SPARTA then uses
                num = dict(sample=(400 if args.quick else 2000))
                (w / "vleo.yaml").write_text(vleo_config(case, sat, lo, hi, n, num, name=name))
                run(vleo + ["-q", "-c", "vleo.yaml", "-o", "vleo_out"] + att_args, w, w / "vleo.log")
                r = json.loads((w / "vleo_out" / "results.json").read_text())
                dg = r["diagnostics"]
                dxyz = dg["cell_size_m"]
                cells = [int(round((hi[k] - lo[k]) / dxyz[k])) for k in range(3)]
                steps = r["diagnostics"]["steps"]
                num.update(cells=cells, dt=dg["timestep_s"], fnum=dg["particle_weight"],
                           warmup=steps - num["sample"])
                num["block"] = num["sample"] // BLOCKS if not args.quick else num["sample"] // 5
                # pilot-rule check: free-stream rule (no pilot) and half the pilot cell size
                for label, kw in (("freestream_rule", dict(pilot=False)), ("half_cell", dict(dsmc_cells=max(dxyz) / 2))):
                    (w / ("vleo_%s.yaml" % label)).write_text(vleo_config(case, sat, lo, hi, n, num, name=name, **kw))
                    run(vleo + ["-q", "-c", "vleo_%s.yaml" % label, "-o", "vleo_%s" % label] + att_args, w,
                        w / ("vleo_%s.log" % label))
                    rr = json.loads((w / ("vleo_%s" % label) / "results.json").read_text())
                    extra_runs[label] = dict(cell_size_m=rr["diagnostics"]["cell_size_m"], C_D=rr["results"]["C_D"],
                                             C_D_se=rr["std_error"]["C_D"], Q=rr["results"]["heat_rate_W"],
                                             Q_se=rr["std_error"]["heat_rate_W"])
                extra_runs["pilot"] = dict(cell_size_m=dxyz, C_D=r["results"]["C_D"], C_D_se=r["std_error"]["C_D"],
                                           Q=r["results"]["heat_rate_W"], Q_se=r["std_error"]["heat_rate_W"],
                                           pilot=dg.get("cell_size_pilot"))
            else:
                num = fm_numerics(lo, hi, case["gas"], args.quick)
                (w / "vleo.yaml").write_text(vleo_config(case, sat, lo, hi, n, num, name=name))
                run(vleo + ["-q", "-c", "vleo.yaml", "-o", "vleo_out"] + att_args, w, w / "vleo.log")
                r = json.loads((w / "vleo_out" / "results.json").read_text())
            sparta_files(w, tris, case["gas"], GEOMETRY[sat].name)
            (w / "in.case").write_text(sparta_input(case, lo, hi, num, att, n))
            run(sparta + ["-in", "in.case", "-log", "log.sparta", "-echo", "none", "-screen", "none"], w,
                w / "sparta.stdout")
            blocks = parse_dump(w / "dump.surf", len(tris), num["sparta_warmup"])
            if len(blocks) < 2:
                raise SystemExit("%s: could not read SPARTA block averages from dump.surf" % w)
            sp = {}
            for k, q in enumerate(("F_x", "F_y", "F_z")):
                sp[q] = mean_se([sum(f[k] for f in b[0]) for b in blocks])
            sp["Q"] = mean_se([b[1] for b in blocks])
            mom = [centroid_moment(cents, b[0]) for b in blocks]
            ours_m = centroid_moment(cents, read_surface_forces(w / "vleo_out" / "surface_fields.csv"))
            res, se = r["results"], r["std_error"]
            ours = {"F_x": (res["force_N"][0], se["force_N"][0]), "F_y": (res["force_N"][1], se["force_N"][1]),
                    "F_z": (res["force_N"][2], se["force_N"][2]), "Q": (res["heat_rate_W"], se["heat_rate_W"])}
            for k, c in enumerate("xyz"):
                sm = mean_se([m[k] for m in mom])
                ours["M_%s (centroids)" % c] = (ours_m[k], se["moment_Nm"][k])
                sp["M_%s (centroids)" % c] = (sm[0], se["moment_Nm"][k])
            rows, ok = compare(ours, sp, SYS[case["dsmc"]])
            results.append(dict(attitude_deg=att, rows=rows, passed=ok, numerics=num,
                                vleo_C=dict(C_D=res["C_D"], C_L=res["C_L"], C_Y=res["C_Y"], C_M=res["C_M"]),
                                vleo_exact_moment_Nm=res["moment_Nm"], sparta_blocks=len(blocks)))
        all_ok = all(x["passed"] for x in results)
        report = dict(case=name, description="%s, %s, %s, %s" % (
            sat, "DSMC" if case["dsmc"] else "free-molecular", "/".join(case["gas"]), case["wall"]),
            date=datetime.date.today().isoformat(), quick=args.quick,
            acceptance="|diff| <= 3 sigma_comb + %g %%" % (100 * SYS[case["dsmc"]]),
            reference="A_ref = 1 m^2, L_ref = 1 m, moment centre = CAD origin, +x = flight direction",
            n_m3=n, speed_mps=SPEED, T_K=T_GAS, T_wall_K=TW, domain=[lo, hi],
            results=results, cell_size_check=extra_runs or None, passed=all_ok)
        (d / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        md = ["# %s: %s" % (name, report["description"]), "",
              "Acceptance (fixed in docs/satellites.md): %s%s" % (
                  report["acceptance"], "  - QUICK RUN, not a decision" if args.quick else ""), ""]
        for x in results:
            md += ["## alpha = %g deg, beta = %g deg (C_D = %.4f, C_L = %.4f, C_Y = %.4f)" % (
                x["attitude_deg"][0], x["attitude_deg"][1], x["vleo_C"]["C_D"], x["vleo_C"]["C_L"], x["vleo_C"]["C_Y"]),
                   "", "| quantity | vleo_solver | SPARTA | difference | z | limit | result |",
                   "| -------- | ----------- | ------ | ---------- | - | ----- | ------ |"]
            for y in x["rows"]:
                md.append("| %s | %.6g ± %.2g | %.6g ± %.2g | %.3g (%+.2f %%) | %.2f | %.3g | %s |" % (
                    y["quantity"], y["vleo"], y["vleo_se"], y["sparta"], y["sparta_se"], y["diff"], 100 * y["rel"],
                    y["z"], y["limit"], "pass" if y["passed"] else "FAIL"))
            md.append("")
        if extra_runs:
            md += ["## Cell size (vleo_solver only)", "", "| grid | cell size [m] | C_D | Q [W] |",
                   "| ---- | ------------- | --- | ----- |"]
            for label in ("freestream_rule", "pilot", "half_cell"):
                e = extra_runs[label]
                md.append("| %s | %s | %.4f ± %.2g | %.4g ± %.2g |" % (
                    label, " x ".join("%.4f" % v for v in e["cell_size_m"]), e["C_D"], e["C_D_se"], e["Q"], e["Q_se"]))
            md.append("")
        md.append("Overall: **%s**" % ("PASS" if all_ok else "FAIL"))
        (d / "report.md").write_text("\n".join(md) + "\n")
        print("\n".join(md), flush=True)
        summary.append((name, all_ok))
    print("\nSummary: " + ", ".join("%s %s" % (nm, "PASS" if ok else "FAIL") for nm, ok in summary))
    return 0 if all(ok for _, ok in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
