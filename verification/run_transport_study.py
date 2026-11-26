#!/usr/bin/env python3
"""Discretisation study (docs/verification.md, stage 2): viscosity and thermal
conductivity of argon (Couette / Fourier, Kn = 0.05, VHS) against cell size,
time step and particles per cell.

    verification/run_transport_study.py run  <build>/tests/test_dsmc_walls [--csv study.csv] [--only dx dt ppc]
    verification/run_transport_study.py plot [--csv study.csv] [--png study.png]

`run` calls the near-continuum test in study mode (VLEO_STUDY_* environment,
see tests/unit/test_dsmc_walls.cpp) once per point and case and appends the
STUDY lines to the CSV; points already in the CSV are skipped, so an
interrupted study resumes. Each sweep varies one parameter, the others stay at
the reference dx = lambda/3, dt = 0.2 tau, 100 particles per cell, 24 000
particles. The simulated time is the same for every time step.

`plot` fits ratio = r0 + b x^2 (x = dx/lambda or dt/tau) to each sweep and
compares b with the hard-sphere theory
    viscosity:    cells 16/(45 pi) = 0.113   (Alexander, Garcia & Alder 1998)
                  time  16/(75 pi) = 0.068   (Hadjiconstantinou 2000)
    conductivity: cells 32/(225 pi) = 0.045  (Garcia & Wagner 2000)
                  time  64/(675 pi) = 0.030  (Garcia & Wagner 2000)
(VHS argon is softer than hard spheres; the coefficients are a guide to the
size and the quadratic form, not an exact reference.)
"""
import argparse
import csv
import math
import os
import subprocess
import sys

SWEEPS = {
    "dx": [dict(dx=x) for x in (1 / 3, 2 / 3, 1.0, 1.5, 2.0)],
    "dt": [dict(dt=x) for x in (0.05, 0.1, 0.2, 0.4, 0.8)],
    "ppc": [dict(ppc=x) for x in (5, 10, 20, 50, 100)],
}
CASES = {"couette": "Couette: walls at -100 and +100 m/s", "fourier": "Fourier: walls at 323 K and 223 K"}
THEORY = {("couette", "dx"): 16 / (45 * math.pi), ("couette", "dt"): 16 / (75 * math.pi),
          ("fourier", "dx"): 32 / (225 * math.pi), ("fourier", "dt"): 64 / (675 * math.pi)}
FIELDS = ["sweep", "case", "model", "dx_lambda", "dt_tau", "ppc", "particles", "seed", "ratio", "se"]


def point_key(sweep, case, p):
    return (sweep, case, round(p.get("dx", 1 / 3), 4), round(p.get("dt", 0.2), 4), int(p.get("ppc", 0)))


def run(a):
    done = set()
    if os.path.exists(a.csv):
        for r in csv.DictReader(open(a.csv)):
            ppc = int(r["ppc"]) if r["sweep"] == "ppc" else 0
            done.add((r["sweep"], r["case"], round(float(r["dx_lambda"]), 4), round(float(r["dt_tau"]), 4), ppc))
    new = not os.path.exists(a.csv)
    out = open(a.csv, "a", newline="")
    w = csv.DictWriter(out, FIELDS)
    if new:
        w.writeheader()
    for sweep in a.only:
        for p in SWEEPS[sweep]:
            for case, section in CASES.items():
                key = point_key(sweep, case, p)
                if key in done:
                    continue
                env = dict(os.environ, VLEO_STUDY_MODEL="VHS")
                if "dx" in p:
                    env["VLEO_STUDY_DX"] = repr(p["dx"])
                if "dt" in p:
                    env["VLEO_STUDY_DT"] = repr(p["dt"])
                if "ppc" in p:
                    env["VLEO_STUDY_PPC"] = repr(p["ppc"])
                print(f"{sweep} {case} {p} ...", flush=True)
                res = subprocess.run([a.test, "[transport]", "-c", section], env=env, capture_output=True, text=True)
                line = next((l for l in res.stdout.splitlines() if l.startswith("STUDY ")), None)
                if line is None:
                    print(res.stdout[-2000:], res.stderr[-2000:], file=sys.stderr)
                    sys.exit(f"error: no STUDY line for {sweep} {case} {p}")
                kv = dict(f.split("=", 1) for f in line.split()[1:])
                row = {k: kv.get(k) for k in FIELDS}
                row["sweep"], row["case"] = sweep, case  # the keys of the resume check
                w.writerow(row)
                out.flush()
                print("   ", line, flush=True)


def plot(a):
    import numpy as np
    rows = list(csv.DictReader(open(a.csv)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=120)
    for ax, sweep, xkey, xlabel in ((axes[0], "dx", "dx_lambda", "cell size dx / lambda"),
                                    (axes[1], "dt", "dt_tau", "time step dt / tau"),
                                    (axes[2], "ppc", "ppc", "particles per cell")):
        for case, col, name in (("couette", "C0", "viscosity mu / mu_CE"), ("fourier", "C3", "conductivity kappa / kappa_CE")):
            rs = sorted((r for r in rows if r["sweep"] == sweep and r["case"] == case), key=lambda r: float(r[xkey]))
            if not rs:
                continue
            x = np.array([float(r[xkey]) for r in rs])
            y = np.array([float(r["ratio"]) for r in rs])
            e = np.array([float(r["se"]) for r in rs])
            ax.errorbar(x, y, yerr=2 * e, color=col, marker="o", ls="none", capsize=3, label=name)
            line = f"{sweep} {case}: " + ", ".join(f"{xi:g}: {yi:.4f}+-{ei:.4f}" for xi, yi, ei in zip(x, y, e))
            if sweep in ("dx", "dt"):
                W = 1 / e ** 2
                A = np.vstack([np.ones_like(x), x ** 2]).T
                cov = np.linalg.inv(A.T @ (A * W[:, None]))
                r0, b = cov @ (A.T @ (W * y))
                xf = np.linspace(0, x.max() * 1.05, 100)
                ax.plot(xf, r0 + b * xf ** 2, color=col, lw=1.5)
                th = THEORY[(case, sweep)]
                ax.plot(xf, r0 * (1 + th * xf ** 2), color=col, lw=1, ls=":")
                line += (f"\n    fit r0 = {r0:.4f} +- {math.sqrt(cov[0, 0]):.4f}, b = {b:.4f} +- {math.sqrt(cov[1, 1]):.4f}"
                         f" (b / r0 = {b / r0:.4f}; hard-sphere theory {th:.4f})")
            print(line)
        ax.axhline(1.0, color="0.6", lw=1)
        ax.set_xlabel(xlabel)
        if sweep == "ppc":
            ax.set_xscale("log")
        ax.set_ylabel("DSMC / Chapman-Enskog (1st order)")
        ax.set_title({"dx": "cell size (dt = 0.2 tau)", "dt": "time step (dx = lambda/3)",
                      "ppc": "particles per cell (dx = lambda/3, dt = 0.2 tau)"}[sweep], loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
    axes[0].text(0.02, 0.02, "solid: fit r0 + b x^2, dotted: hard-sphere theory", transform=axes[0].transAxes,
                 fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(a.png)
    print("wrote", a.png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("test", help="path to tests/test_dsmc_walls")
    r.add_argument("--csv", default="results/transport_study.csv")
    r.add_argument("--only", nargs="+", default=["dx", "dt", "ppc"], choices=list(SWEEPS))
    p = sub.add_parser("plot")
    p.add_argument("--csv", default="results/transport_study.csv")
    p.add_argument("--png", default="results/transport_study.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
    run(a) if a.cmd == "run" else plot(a)


if __name__ == "__main__":
    main()
