#!/usr/bin/env python3
"""Normal-shock structure from the cell-state files of shock runs
(make_shock_configs.py, docs/verification.md).

    verification/shock_profile.py results/shock_ma1p4 results/shock_ma2 ... [--png shock.png]

Per run: the cell-state files of the sampling phase are averaged over y and z,
each file is shifted so that its shock centre (n = (n1 + n2) / 2) lies at
x = 0, and the aligned profiles are averaged. Thickness delta = 2 w of the
least-squares fit n = (n1+n2)/2 + (n2-n1)/2 tanh((x - x0) / w) over |x| < 20 lambda1
(same estimator as test_dsmc_shock), standard error from 5 groups of files.
The Navier-Stokes structure with the same mu(T) (first-order Chapman-Enskog,
VHS) and kappa = 15 k mu / (4 m) is integrated and fitted the same way.
Prints Rankine-Hugoniot checks and lambda1 / delta; --png draws the density
and temperature profiles and lambda1 / delta over the Mach number.
Needs numpy, h5py, pyyaml (and matplotlib for --png).
"""
import argparse
import glob
import json
import math
import os
import sys

try:
    import numpy as np
    import h5py
    import yaml
except ImportError as e:
    sys.exit(f"error: {e.name} is required (pip install numpy h5py pyyaml)")

K_B = 1.380649e-23
M_AR = 39.948 * 1.66053906660e-27
D_REF, OMEGA, T_REF = 4.17e-10, 0.81, 273.0
GAMMA = 5.0 / 3.0


def rankine_hugoniot(M):
    g, M2 = GAMMA, M * M
    r = (g + 1) * M2 / ((g - 1) * M2 + 2)
    t = (2 * g * M2 - (g - 1)) * ((g - 1) * M2 + 2) / ((g + 1) ** 2 * M2)
    return r, 1.0 / r, t


def viscosity(T):
    w = OMEGA
    return 15 * math.sqrt(math.pi * M_AR * K_B * T_REF) / (2 * math.pi * D_REF ** 2 * (5 - 2 * w) * (7 - 2 * w)) * \
        (T / T_REF) ** w


def navier_stokes(n1, T1, u1, M, xs):
    """Density and temperature of the NS shock on xs (centre n = (n1+n2)/2 at 0)."""
    m, R = M_AR, K_B / M_AR
    cv = 1.5 * R
    r, ur, tr = rankine_hugoniot(M)
    rho1 = n1 * m
    m0 = rho1 * u1
    P = m0 * u1 + rho1 * R * T1
    E = m0 * (2.5 * R * T1 + 0.5 * u1 * u1)

    def rhs(y):
        u, T = y
        mu = viscosity(T)
        kappa = 15 * K_B / (4 * m) * mu
        return np.array([0.75 / mu * (m0 * u + m0 * R * T / u - P), (m0 * cv * T - 0.5 * m0 * u * u + P * u - E) / kappa])

    u2, T2 = u1 * ur, T1 * tr
    y2 = np.array([u2, T2])
    J = np.zeros((2, 2))
    f0 = rhs(y2)
    for k, h in enumerate((1e-6 * u2, 1e-6 * T2)):
        d = np.zeros(2)
        d[k] = h
        J[:, k] = (rhs(y2 + d) - f0) / h
    lam, vec = np.linalg.eig(J)
    e = vec[:, int(np.argmin(lam.real))].real
    e = e / math.hypot(e[0] / u2, e[1] / T2)
    if e[0] < 0:
        e = -e
    y = y2 + 1e-6 * e
    h = -2e-7 * 1e21 / n1  # 0.2 um at n1 = 1e21 (scales with the mean free path)
    xs_ns, ns, Ts = [0.0], [n1 * r], [T2]
    x = 0.0
    for _ in range(3000000):
        if y[0] >= u1 * (1 - 1e-6):
            break
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
        xs_ns.append(x)
        ns.append(m0 / (m * y[0]))
        Ts.append(y[1])
    xs_ns, ns, Ts = np.array(xs_ns[::-1]), np.array(ns[::-1]), np.array(Ts[::-1])
    nm = 0.5 * n1 * (1 + r)
    i = int(np.argmax(ns >= nm))
    x0 = xs_ns[i - 1] + (nm - ns[i - 1]) / (ns[i] - ns[i - 1]) * (xs_ns[i] - xs_ns[i - 1])
    return np.interp(xs + x0, xs_ns, ns), np.interp(xs + x0, xs_ns, Ts)


def tanh_thickness(xs, n, n1, n2, rng):
    a, b = 0.5 * (n1 + n2), 0.5 * (n2 - n1)
    sel = np.abs(xs) < rng

    def sse(x0, w):
        return float(np.sum((n[sel] - (a + b * np.tanh((xs[sel] - x0) / w))) ** 2))

    x0, w, dx0, dw = 0.0, 0.2 * rng, 0.1 * rng, 0.1 * rng
    best = sse(x0, w)
    for _ in range(400):
        if dx0 < 1e-6 * rng and dw < 1e-6 * rng:
            break
        moved = False
        for px, pw in ((dx0, 0), (-dx0, 0), (0, dw), (0, -dw)):
            if w + pw <= 0:
                continue
            e = sse(x0 + px, w + pw)
            if e < best:
                best, x0, w, moved = e, x0 + px, w + pw, True
        if not moved:
            dx0 *= 0.5
            dw *= 0.5
    return 2 * w


def load_run(run):
    res = json.load(open(os.path.join(run, "results.json")))
    cfg = yaml.safe_load(open(os.path.join(run, "resolved_config.yaml")))
    lo, hi = np.array(cfg["domain"]["box_min_m"], float), np.array(cfg["domain"]["box_max_m"], float)
    dx = np.array(res["diagnostics"]["cell_size_m"], float)
    nx = int(round((hi[0] - lo[0]) / dx[0]))
    faces = cfg["dsmc"]["faces"]
    up, down = faces["x_min"], faces["x_max"]
    n1, T1, u1 = up["number_density_m3"], up["temperature_K"], up["velocity_mps"][0]
    M = u1 / math.sqrt(GAMMA * K_B * T1 / M_AR)
    xc = lo[0] + (np.arange(nx) + 0.5) * dx[0]
    profiles = []
    for f in sorted(glob.glob(os.path.join(run, "dsmc_state", "dsmc_state_*.vtkhdf"))):
        with h5py.File(f, "r") as h:
            g = h["VTKHDF"]
            if int(np.asarray(g["FieldData"]["SamplingPhase"]).ravel()[0]) != 1:
                continue
            c = np.asarray(g["CellData"]["cell_index"]).astype(np.int64).ravel()
            nd = np.asarray(g["CellData"]["number_density_m3"]).ravel()
            vel = np.asarray(g["CellData"]["velocity_mps"]).reshape(-1, 3)
            tt = np.asarray(g["CellData"]["temperature_trans_K"]).ravel()
        ix = c % nx
        ok = np.isfinite(tt) & (nd > 0)
        w = np.bincount(ix[ok], weights=nd[ok], minlength=nx)
        cnt = np.bincount(ix[ok], minlength=nx)
        n = w / np.maximum(cnt, 1)
        u = np.bincount(ix[ok], weights=nd[ok] * vel[ok, 0], minlength=nx) / np.maximum(w, 1e-300)
        T = np.bincount(ix[ok], weights=nd[ok] * tt[ok], minlength=nx) / np.maximum(w, 1e-300)
        profiles.append((n, u, T))
    if not profiles:
        sys.exit(f"error: {run}: no cell-state files of the sampling phase")
    return dict(res=res, n1=n1, T1=T1, u1=u1, n2=down["number_density_m3"], T2=down["temperature_K"],
                u2=down["velocity_mps"][0], M=M, xc=xc, dx=dx[0], profiles=profiles,
                lam=res["diagnostics"]["mean_free_path_inf_m"])


def align(d):
    """Shift every file to its shock centre; returns aligned profiles on xs."""
    xc, n1, n2 = d["xc"], d["n1"], d["n2"]
    nm = 0.5 * (n1 + n2)
    xs = xc - 0.5 * (xc[0] + xc[-1])
    out = []
    for n, u, T in d["profiles"]:
        sm = np.convolve(n, np.ones(5) / 5, mode="same")
        i = int(np.argmax((sm[2:-2] >= nm))) + 2
        x0 = xc[i - 1] + (nm - sm[i - 1]) / (sm[i] - sm[i - 1]) * (xc[i] - xc[i - 1])
        out.append([np.interp(xs, xc - x0, v, left=np.nan, right=np.nan) for v in (n, u, T)])
    return xs, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        d = load_run(run)
        xs, prof = align(d)
        lam, n1, n2, T1, T2 = d["lam"], d["n1"], d["n2"], d["T1"], d["T2"]
        arr = np.array(prof)  # files x 3 x bins
        groups = np.array_split(np.arange(len(arr)), 5)
        deltas = []
        for g in groups:
            ng = np.nanmean(arr[g, 0, :], axis=0)
            ng = np.where(np.isfinite(ng), ng, np.where(xs < 0, n1, n2))
            deltas.append(tanh_thickness(xs, ng, n1, n2, 20 * lam))
        n_all = np.nanmean(arr[:, 0, :], axis=0)
        T_all = np.nanmean(arr[:, 2, :], axis=0)
        u_all = np.nanmean(arr[:, 1, :], axis=0)
        n_fill = np.where(np.isfinite(n_all), n_all, np.where(xs < 0, n1, n2))
        delta = tanh_thickness(xs, n_fill, n1, n2, 20 * lam)
        delta_se = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
        n_ns, T_ns = navier_stokes(n1, T1, d["u1"], d["M"], xs)
        delta_ns = tanh_thickness(xs, n_ns, n1, n2, 20 * lam)
        up = (xs < -15 * lam) & (xs > -25 * lam)
        dn = (xs > 15 * lam) & (xs < 25 * lam)
        r, _, t = rankine_hugoniot(d["M"])
        print(f"{run}: Ma {d['M']:.3f}, {len(arr)} files, lambda1 {lam * 1e3:.4f} mm")
        print(f"  plateaus: n2/n1 {np.nanmean(n_all[dn]) / np.nanmean(n_all[up]):.4f} (RH {r:.4f}), "
              f"T2/T1 {np.nanmean(T_all[dn]) / np.nanmean(T_all[up]):.4f} (RH {t:.4f}), "
              f"mass flux ratio {np.nanmean(n_all[dn] * u_all[dn]) / np.nanmean(n_all[up] * u_all[up]):.4f}")
        print(f"  thickness {delta / lam:.3f} +- {delta_se / lam:.3f} lambda1, Navier-Stokes {delta_ns / lam:.3f}; "
              f"lambda1/delta {lam / delta:.4f} +- {lam * delta_se / delta ** 2:.4f} (NS {lam / delta_ns:.4f})")
        rows.append(dict(M=d["M"], xs=xs / lam, n=(n_all - n1) / (n2 - n1), T=(T_all - T1) / (T2 - T1),
                         n_ns=(n_ns - n1) / (n2 - n1), T_ns=(T_ns - T1) / (T2 - T1), inv=lam / delta,
                         inv_se=lam * delta_se / delta ** 2, inv_ns=lam / delta_ns))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (p1, p2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)
        cols = plt.cm.viridis(np.linspace(0, 0.85, len(rows)))
        for row, c in zip(sorted(rows, key=lambda r: r["M"]), cols):
            p1.plot(row["xs"], row["n"], color=c, lw=2, label=f"Ma {row['M']:.3g} DSMC")
            p1.plot(row["xs"], row["T"], color=c, lw=1.2, ls=":")
            p1.plot(row["xs"], row["n_ns"], color=c, lw=1, ls="--")
        p1.set_xlim(-20, 20)
        p1.set_xlabel("x / lambda1")
        p1.set_ylabel("(q - q1) / (q2 - q1)")
        p1.set_title("density (solid: DSMC, dashed: Navier-Stokes), temperature (dotted)", loc="left", fontsize=10)
        p1.legend(frameon=False, fontsize=9)
        rs = sorted(rows, key=lambda r: r["M"])
        p2.errorbar([r["M"] for r in rs], [r["inv"] for r in rs], yerr=[2 * r["inv_se"] for r in rs], marker="o",
                    label="DSMC (VHS argon)")
        p2.plot([r["M"] for r in rs], [r["inv_ns"] for r in rs], marker="s", ls="--", label="Navier-Stokes")
        p2.set_xlabel("Mach number")
        p2.set_ylabel("lambda1 / delta  (lambda1: VHS mean free path)")
        p2.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(a.png)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
