#!/usr/bin/env python3
"""Evaluate the sphere-drag series (make_sphere_configs.py, docs/verification.md).

    verification/sphere_eval.py results/sphere_ma4_fm results/sphere_ma4_kn* [--png sphere.png]

References, all functions of the Mach number M, the Reynolds number
Re = rho U D / mu(T_inf) and T_w / T_inf (mu: first-order Chapman-Enskog
viscosity of the VHS argon that the solver uses):
  - free-molecular sphere, diffuse wall (Patterson 1971; exact):
      C_fm = (1 + 2 s^2) exp(-s^2) / (sqrt(pi) s^3) + (4 s^4 + 4 s^2 - 1) erf(s) / (2 s^4)
             + 2 sqrt(pi) / (3 s) sqrt(T_w / T_inf),          s = M sqrt(gamma / 2)
  - Henderson (AIAA J. 14, 1976), M > 1.75:
      C_D = [0.9 + 0.34/M^2 + 1.86 sqrt(M/Re) (2 + 2/s^2 + 1.058/s sqrt(T_w/T_inf) - 1/s^4)]
            / (1 + 1.86 sqrt(M/Re))
  - Loth (AIAA J. 46, 2008), rarefaction regime Re <= 45:
      C_D = C_Kn,Re / (1 + M^4) + M^4 C_fm,Re / (1 + M^4)
      C_Kn,Re = 24/Re (1 + 0.15 Re^0.687) f_Kn,  f_Kn = 1 / (1 + Kn (2.514 + 0.8 exp(-0.55/Kn)))
      C_fm,Re = C_fm / (1 + (C'_fm / 1.63 - 1) sqrt(Re / 45)),  C'_fm = C_fm without the wall term
      Kn = sqrt(pi gamma / 2) M / Re
The correlations are fits to measurements (sphere drag in ballistic ranges and
low-density tunnels, among them argon, Aroesty 1963) with errors of 5-15 %; they
are a plausibility band, not an exact reference. The free-molecular value is
exact (the faceted sphere differs from it by about 0.1 %).
Needs numpy and pyyaml (matplotlib for --png).
"""
import argparse
import json
import math
import os
import sys

try:
    import numpy as np
    import yaml
except ImportError as e:
    sys.exit(f"error: {e.name} is required (pip install numpy pyyaml)")

K_B = 1.380649e-23
M_AR = 39.948 * 1.66053906660e-27
GAMMA = 5.0 / 3.0
D_REF, OMEGA, T_REF = 4.17e-10, 0.81, 273.0  # VHS argon (species database)
D_SPH = 0.1


def mu_vhs(T):
    """First-order Chapman-Enskog viscosity of VHS argon."""
    return (7.5 * math.sqrt(math.pi * M_AR * K_B * T_REF) / ((5 - 2 * OMEGA) * (7 - 2 * OMEGA) * math.pi * D_REF ** 2)
            * (T / T_REF) ** OMEGA)


def lambda_vhs(n, T):
    mr = 0.5 * M_AR
    sg = 2 * math.sqrt(math.pi) * D_REF ** 2 * (T / T_REF) ** (1 - OMEGA) * math.sqrt(2 * K_B * T_REF / mr)
    return math.sqrt(8 * K_B * T / (math.pi * M_AR)) / (n * sg)


def cd_fm(M, tw, wall=True):
    s = M * math.sqrt(GAMMA / 2)
    c = ((1 + 2 * s * s) * math.exp(-s * s) / (math.sqrt(math.pi) * s ** 3)
         + (4 * s ** 4 + 4 * s * s - 1) * math.erf(s) / (2 * s ** 4))
    return c + (2 * math.sqrt(math.pi) / (3 * s) * math.sqrt(tw) if wall else 0.0)


def cd_henderson(M, Re, tw):
    s = M * math.sqrt(GAMMA / 2)
    r = 1.86 * math.sqrt(M / Re)
    return (0.9 + 0.34 / M ** 2 + r * (2 + 2 / s ** 2 + 1.058 / s * math.sqrt(tw) - 1 / s ** 4)) / (1 + r)


def cd_loth(M, Re, tw):
    if Re > 45:
        return float("nan")
    kn = math.sqrt(math.pi * GAMMA / 2) * M / Re
    fkn = 1 / (1 + kn * (2.514 + 0.8 * math.exp(-0.55 / kn)))
    c_kn = 24 / Re * (1 + 0.15 * Re ** 0.687) * fkn
    c_fm_re = cd_fm(M, tw) / (1 + (cd_fm(M, tw, wall=False) / 1.63 - 1) * math.sqrt(Re / 45))
    return (c_kn + M ** 4 * c_fm_re) / (1 + M ** 4)


def evaluate(run):
    res = json.load(open(os.path.join(run, "results.json")))
    cfg = yaml.safe_load(open(os.path.join(run, "resolved_config.yaml")))
    n = float(cfg["gas"]["number_density_m3"])
    T = float(cfg["gas"]["translational_temperature_K"])
    U = float(cfg["gas"]["bulk_velocity_body_mps"][0])
    tw = float(cfg["surface"]["temperature_K"]) / T
    M = U / math.sqrt(GAMMA * K_B * T / M_AR)
    mode = cfg["simulation"]["mode"]
    re = n * M_AR * U * D_SPH / mu_vhs(T) if mode == "dsmc" else 0.0
    kn = lambda_vhs(n, T) / D_SPH if mode == "dsmc" else math.inf
    d = dict(run=run, mode=mode, M=M, tw=tw, Re=re, kn=kn, cd=res["results"]["C_D"], se=res["std_error"]["C_D"],
             fm=cd_fm(M, tw))
    d["henderson"] = cd_henderson(M, re, tw) if re > 0 else cd_henderson(M, 1e-12, tw)
    d["loth"] = cd_loth(M, re, tw) if re > 0 else d["fm"]
    d["diag"] = res.get("diagnostics", {})
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
        rows.append(evaluate(run))
    rows.sort(key=lambda d: -d["kn"])
    if not rows:
        return
    M, tw = rows[0]["M"], rows[0]["tw"]
    print(f"Ma {M:.3f}, T_w/T_inf {tw:.3f}: free-molecular C_D {cd_fm(M, tw):.4f} "
          f"(Henderson's free-molecular limit {cd_henderson(M, 1e-12, tw):.4f})")
    print(f"{'run':<30} {'Kn':>7} {'Re':>8} {'C_D':>8} {'SE':>7} {'/fm':>7} {'Henderson':>10} {'dev':>7} {'Loth':>8} {'dev':>7}")
    for d in rows:
        kn = "inf" if math.isinf(d["kn"]) else f"{d['kn']:.3g}"
        loth = f"{d['loth']:8.4f} {100 * (d['cd'] / d['loth'] - 1):+6.1f}%" if not math.isnan(d["loth"]) else f"{'-':>8} {'':>7}"
        print(f"{os.path.basename(d['run']):<30} {kn:>7} {d['Re']:8.3g} {d['cd']:8.4f} {d['se']:7.1g} {d['cd'] / d['fm']:7.4f} "
              f"{d['henderson']:10.4f} {100 * (d['cd'] / d['henderson'] - 1):+6.1f}% {loth}")
    fm_rows = [d for d in rows if d["mode"] == "tpmc"]
    for d in fm_rows:
        print(f"free-molecular check: TPMC {d['cd']:.4f} +- {d['se']:.1g} vs exact {d['fm']:.4f} "
              f"({100 * (d['cd'] / d['fm'] - 1):+.2f} %)")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        kn_c = np.logspace(-2.5, 1.5, 200)
        # Re from Kn at fixed Ma via the measured Re * Kn product of the DSMC runs (constant at fixed M, T)
        prod = np.mean([d["Re"] * d["kn"] for d in rows if d["mode"] == "dsmc"])
        re_c = prod / kn_c
        ax.plot(kn_c, [cd_henderson(M, r, tw) for r in re_c], color="C0", lw=1.5, label="Henderson (1976)")
        ax.plot(kn_c, [cd_loth(M, r, tw) for r in re_c], color="C2", lw=1.5, label="Loth (2008), Re <= 45")
        ax.axhline(cd_fm(M, tw), color="0.5", lw=1, ls="--", label="free-molecular (exact)")
        ds = [d for d in rows if d["mode"] == "dsmc"]
        ax.errorbar([d["kn"] for d in ds], [d["cd"] for d in ds], yerr=[2 * d["se"] for d in ds], color="C3",
                    marker="o", ls="none", capsize=3, label="DSMC (vleo_solver)")
        for d in fm_rows:
            ax.plot([kn_c[-1]], [d["cd"]], marker=">", color="C3", ms=8, ls="none", label="TPMC (Kn = inf)")
        ax.set_xscale("log")
        ax.set_xlabel("Kn = lambda_VHS,inf / D")
        ax.set_ylabel("C_D")
        ax.set_title(f"sphere in argon, Ma {M:.3g}, T_w / T_inf = {tw:.3g}, diffuse wall", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(a.png)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
