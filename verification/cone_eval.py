#!/usr/bin/env python3
"""Evaluate the 70-deg blunted cone runs (make_cone_configs.py, docs/verification.md).

    verification/cone_eval.py results/cone_case1_dx0p4 [...] [--png cone.png]

Per run: drag coefficient (reference area pi d^2 / 4; the configs use a quarter of
it on the quarter model), the stagnation-point heat transfer coefficient
C_H = 2 q / (rho V^3) (facets within r < 1.5 mm of the axis on the nose) and the
distributions of pressure and heat flux along the forebody contour (s / R_n from
the stagnation point; area-weighted over the facets of each contour segment).

Reference: Moss, Dogra & Wilmoth, NASA TM-107762 (1993), DSMC (Bird's code),
table 4 "forebody & wake" runs (no sting): case 1 C_D = 1.63 / 1.62 (runs 2, 3),
C_H = 0.353 / 0.350; case 2 C_D = 1.56, C_H = 0.203 (run 10). With sting
(table 2): 1.68 / 1.59, C_H 0.349 / 0.202. The surface distribution of case 1
is taken from table 5 (cone with sting; the forebody is unaffected by the
sting, TM-107762 section "Effect of afterbody on forebody solution").
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_cone_configs import R_N, profile  # noqa: E402

K_B = 1.380649e-23
M_N2 = 28.02e-3 / 6.02214076e23
REF = {"case1": dict(cd=(1.63, 1.62), ch=(0.353, 0.350)), "case2": dict(cd=(1.56,), ch=(0.203,))}
# TM-107762 table 5 (case 1, cone with sting): element midpoint x, y [m], pressure [Pa], net heat flux [W/m2]
MOSS_CASE1 = [
    (5.396e-6, 2.597e-4, 36.44, 1.009e4), (2.321e-4, 2.382e-3, 36.37, 9.928e3),
    (1.078e-3, 5.165e-3, 34.98, 8.655e3), (2.168e-3, 8.160e-3, 34.25, 7.513e3),
    (3.275e-3, 1.120e-2, 33.67, 7.016e3), (4.400e-3, 1.429e-2, 32.93, 6.651e3),
    (5.542e-3, 1.743e-2, 31.57, 6.441e3), (6.702e-3, 2.062e-2, 29.75, 6.359e3),
    (7.406e-3, 2.255e-2, 27.74, 6.437e3), (7.879e-3, 2.385e-2, 25.49, 6.823e3),
    (8.083e-3, 2.436e-2, 19.76, 5.968e3), (8.278e-3, 2.462e-2, 14.13, 4.784e3),
    (8.597e-3, 2.485e-2, 7.402, 2.646e3), (9.046e-3, 2.499e-2, 2.986, 1.054e3),
]


def arc_length(pts):
    s = [0.0]
    for (x0, r0), (x1, r1) in zip(pts[:-1], pts[1:]):
        s.append(s[-1] + math.hypot(x1 - x0, r1 - r0))
    return np.array(s)


def s_of_point(pts, s_nodes, x, r):
    """Arc length of the contour point closest to (x, r)."""
    best = (1e30, 0.0)
    for k in range(len(pts) - 1):
        (x0, r0), (x1, r1) = pts[k], pts[k + 1]
        dx, dr = x1 - x0, r1 - r0
        L2 = dx * dx + dr * dr
        t = min(1.0, max(0.0, ((x - x0) * dx + (r - r0) * dr) / L2))
        d = math.hypot(x0 + t * dx - x, r0 + t * dr - r)
        if d < best[0]:
            best = (d, s_nodes[k] + t * math.sqrt(L2))
    return best[1]


def evaluate(run):
    res = json.load(open(os.path.join(run, "results.json")))
    cfg = yaml.safe_load(open(os.path.join(run, "resolved_config.yaml")))
    n = float(cfg["gas"]["number_density_m3"])
    U = float(cfg["gas"]["bulk_velocity_body_mps"][0])
    rho = n * M_N2
    case = "case1" if rho < 3e-5 else "case2"
    pts, _ = profile()
    s_nodes = arc_length(pts)
    nseg = len(pts) - 1
    acc = np.zeros((nseg, 3))  # area, heat rate, pressure * area
    q_st, a_st = 0.0, 0.0
    with open(os.path.join(run, "surface_fields.csv")) as f:
        for row in csv.DictReader(f):
            if float(row["impingement_rate_per_s"]) <= 0.0:
                continue  # outside the quarter model
            x = float(row["centroid_x_m"])
            r = math.hypot(float(row["centroid_y_m"]), float(row["centroid_z_m"]))
            area = float(row["area_m2"])
            q = float(row["heat_rate_W"])
            if r < 1.5e-3 and x < 1e-3:
                q_st += q
                a_st += area
            s = s_of_point(pts, s_nodes, x, r)
            k = min(nseg - 1, int(np.searchsorted(s_nodes, s, side="right")) - 1)
            acc[k] += (area, q, float(row["pressure_Pa"]) * area)
    ok = acc[:, 0] > 0
    s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
    d = dict(run=run, case=case, rho=rho, U=U, cd=res["results"]["C_D"], cd_se=res["std_error"]["C_D"],
             q_st=q_st / a_st, s=s_mid[ok] / R_N, q=acc[ok, 1] / acc[ok, 0], p=acc[ok, 2] / acc[ok, 0],
             dx=float(cfg["dsmc"]["cell_size_m"]))
    d["ch"] = 2 * d["q_st"] / (rho * U ** 3)
    return d


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
        ref = REF[d["case"]]
        cd_ref, ch_ref = np.mean(ref["cd"]), np.mean(ref["ch"])
        print(f"{run}: {d['case']}, dx = {d['dx'] * 1e3:g} mm")
        print(f"  C_D = {d['cd']:.4f} +- {d['cd_se']:.1g}, TM-107762 {'/'.join(f'{v:g}' for v in ref['cd'])} "
              f"({100 * (d['cd'] / cd_ref - 1):+.1f} %)")
        print(f"  stagnation q = {d['q_st'] / 1e4:.3f} W/cm2, C_H = {d['ch']:.4f}, TM-107762 "
              f"{'/'.join(f'{v:g}' for v in ref['ch'])} ({100 * (d['ch'] / ch_ref - 1):+.1f} %)")
        if d["case"] == "case1":
            pts, _ = profile()
            s_nodes = arc_length(pts)
            print("  s/R_n    p [Pa]  (Moss)    q [kW/m2]  (Moss)")
            for x, y, pm, qm in MOSS_CASE1[::2]:
                s = s_of_point(pts, s_nodes, x, y) / R_N
                print(f"  {s:5.2f}  {np.interp(s, d['s'], d['p']):7.2f}  ({pm:6.2f})  {np.interp(s, d['s'], d['q']) / 1e3:7.3f}"
                      f"  ({qm / 1e3:6.3f})")
    if a.png and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=120)
        pts, _ = profile()
        s_nodes = arc_length(pts)
        sm = [s_of_point(pts, s_nodes, x, y) / R_N for x, y, _, _ in MOSS_CASE1]
        for d in rows:
            lab = f"{d['case']}, dx {d['dx'] * 1e3:g} mm"
            fs = d["s"] < 3.2
            axes[0].plot(d["s"][fs], d["p"][fs] / (d["rho"] * d["U"] ** 2 / 2), label=lab)
            axes[1].plot(d["s"][fs], 2 * d["q"][fs] / (d["rho"] * d["U"] ** 3), label=lab)
        rho1, u1 = 1.729e-5, 1502.0
        axes[0].plot(sm, [p / (rho1 * u1 ** 2 / 2) for _, _, p, _ in MOSS_CASE1], "ko", ms=4, label="TM-107762 case 1")
        axes[1].plot(sm, [2 * q / (rho1 * u1 ** 3) for _, _, _, q in MOSS_CASE1], "ko", ms=4, label="TM-107762 case 1")
        for ax, t in zip(axes, ("pressure coefficient C_p", "heat transfer coefficient C_H")):
            ax.set_xlabel("s / R_n (forebody)")
            ax.set_title(t, loc="left", fontsize=11)
            ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(a.png)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
