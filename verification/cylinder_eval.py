#!/usr/bin/env python3
"""Evaluate the Lofthouse cylinder runs (make_cylinder_configs.py, docs/verification.md).

    verification/cylinder_eval.py results/cylinder_ma10_kn0p25 [...] [--png cylinder.png]

Per run: drag per unit length 2 F_x / dz (half model, slab thickness dz) with its
standard error, and the surface distributions over the angle from the stagnation
line: pressure, shear stress and heat flux per facet (both triangles of a facet,
area inside the slab = chord x dz). The peak heat flux is the mean of the facets
within 2 deg of the stagnation line. Reference: MONACO DSMC of Lofthouse (PhD
thesis, University of Michigan 2008), tables 4.3 and 4.4.
Needs numpy and pyyaml (matplotlib for --png).
"""
import argparse
import csv
import json
import math
import os
import sys

try:
    import numpy as np
    import yaml
except ImportError as e:
    sys.exit(f"error: {e.name} is required (pip install numpy pyyaml)")

# Lofthouse 2008, table 4.3 (drag per length, N/m) and 4.4 (peak heating, kW/m2), DSMC (MONACO)
REF_DRAG = {(10, "0.002"): 187.6, (10, "0.01"): 40.02, (10, "0.05"): 8.866, (10, "0.25"): 2.092,
            (25, "0.002"): 1171.0, (25, "0.01"): 250.8, (25, "0.05"): 56.92, (25, "0.25"): 13.19}
REF_HEAT = {(10, "0.002"): 89.80, (10, "0.01"): 39.13, (10, "0.05"): 15.85, (10, "0.25"): 5.926,
            (25, "0.002"): 1746.0, (25, "0.01"): 749.6, (25, "0.05"): 304.5, (25, "0.25"): 108.2}
N_INF = {"0.002": 2.124e21, "0.01": 4.247e20, "0.05": 8.494e19, "0.25": 1.699e19}
N_FACETS = 360
R = 0.1524


def evaluate(run):
    res = json.load(open(os.path.join(run, "results.json")))
    cfg = yaml.safe_load(open(os.path.join(run, "resolved_config.yaml")))
    lo = [float(v) for v in cfg["domain"]["box_min_m"]]
    hi = [float(v) for v in cfg["domain"]["box_max_m"]]
    dz = hi[2] - lo[2]
    n = float(cfg["gas"]["number_density_m3"])
    U = float(cfg["gas"]["bulk_velocity_body_mps"][0])
    ma = 10 if U < 4000 else 25
    kn = min(N_INF, key=lambda k: abs(math.log(N_INF[k] / n)))
    fx = res["results"]["force_N"][0]
    fx_se = res["std_error"]["force_N"][0]
    drag, drag_se = 2 * fx / dz, 2 * fx_se / dz
    # surface distributions per facet (upper half: facets 0..179, facet i spans [i, i+1] deg)
    acc = {}
    with open(os.path.join(run, "surface_fields.csv")) as f:
        for r in csv.DictReader(f):
            src = int(r["source_index"])
            if src >= 2 * N_FACETS:
                continue  # caps (outside the slab)
            i = src // 2
            a = acc.setdefault(i, [0.0, 0.0, 0.0, 0.0])
            a[0] += float(r["force_x_N"])
            a[1] += float(r["force_y_N"])
            a[2] += float(r["heat_rate_W"])
    chord = 2 * R * math.sin(math.pi / N_FACETS)
    area = chord * dz
    phi, p, tau, q = [], [], [], []
    for i in range(N_FACETS // 2):
        if i not in acc:
            continue
        th = math.radians(i + 0.5)
        nx, ny = math.cos(th), math.sin(th)  # outward normal
        fxi, fyi, qi = acc[i][0], acc[i][1], acc[i][2]
        fn = fxi * nx + fyi * ny  # force on the wall along the outward normal (pressure pushes inwards: < 0)
        ft = -fxi * ny + fyi * nx
        phi.append(180.0 - (i + 0.5))
        p.append(-fn / area)
        tau.append(abs(ft) / area)
        q.append(qi / area)
    order = np.argsort(phi)
    phi, p, tau, q = (np.array(v)[order] for v in (phi, p, tau, q))
    peak = float(np.mean(q[phi < 2.0]))
    return dict(run=run, ma=ma, kn=kn, drag=drag, drag_se=drag_se, phi=phi, p=p, tau=tau, q=q, peak=peak)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        if not os.path.exists(os.path.join(run, "results.json")):
            print(f"{run}: no results.json - skipped")
            continue
        d = evaluate(run)
        rows.append(d)
        rd, rq = REF_DRAG.get((d["ma"], d["kn"])), REF_HEAT.get((d["ma"], d["kn"]))
        print(f"{run}: Ma {d['ma']}, Kn {d['kn']}")
        print(f"  drag/length {d['drag']:.4g} +- {d['drag_se']:.2g} N/m, Lofthouse DSMC {rd} N/m "
              f"({100 * (d['drag'] / rd - 1):+.2f} %)")
        print(f"  peak heat flux (Phi < 2 deg) {d['peak'] / 1e3:.4g} kW/m2, Lofthouse DSMC {rq} kW/m2 "
              f"({100 * (d['peak'] / 1e3 / rq - 1):+.2f} %); max over Phi {d['q'].max() / 1e3:.4g} kW/m2")
        print(f"  stagnation pressure {np.mean(d['p'][d['phi'] < 2.0]):.4g} Pa, max shear {d['tau'].max():.4g} Pa "
              f"at Phi = {d['phi'][int(np.argmax(d['tau']))]:.1f} deg")
        out = os.path.join(run, "surface_profile.csv")
        with open(out, "w") as f:
            f.write("phi_deg,pressure_Pa,shear_Pa,heat_flux_W_m2\n")
            for row in zip(d["phi"], d["p"], d["tau"], d["q"]):
                f.write("%.2f,%.6g,%.6g,%.6g\n" % row)
    if a.png and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120)
        for d in rows:
            lab = f"Ma {d['ma']}, Kn {d['kn']}"
            axes[0].plot(d["phi"], d["p"], label=lab)
            axes[1].plot(d["phi"], d["tau"], label=lab)
            axes[2].plot(d["phi"], d["q"] / 1e3, label=lab)
            rq = REF_HEAT.get((d["ma"], d["kn"]))
            if rq:
                axes[2].plot([0], [rq], marker="*", ms=12, color=axes[2].lines[-1].get_color())
        for ax, t in zip(axes, ("pressure [Pa]", "shear stress [Pa]", "heat flux [kW/m2] (star: Lofthouse peak)")):
            ax.set_xlabel("angle from the stagnation line [deg]")
            ax.set_title(t, loc="left", fontsize=11)
            ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(a.png)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
