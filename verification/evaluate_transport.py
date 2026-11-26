#!/usr/bin/env python3
"""Viscosity (Couette) or thermal conductivity (Fourier) of a verification run
against first-order Chapman-Enskog (docs/verification.md).

    verification/evaluate_transport.py results/couette_argon [--core 0.3]

Reads results.json (wall force / heat rate with standard errors),
resolved_config.yaml (domain, walls, species, collision model) and the
cell-state files of the sampling phase (dsmc_state/*.vtkhdf, needs numpy and
h5py). Profiles are averaged over x and z, the gradient is fitted in the core
|y| < core * H. Fourier: kappa ~ T^omega is handled exactly by fitting
T^(1 + omega). The case is detected from the walls: moving walls -> Couette,
different wall temperatures -> Fourier.
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
AMU = 1.66053906660e-27
# Bird (1994), Table A1: molar mass, VHS d, omega, VSS d, alpha (T_ref = 273 K)
SPECIES = {"Ar": (39.948, 4.17e-10, 0.81, 4.11e-10, 1.40), "He": (4.0026, 2.33e-10, 0.66, 2.30e-10, 1.26)}


def viscosity_ce(sp, model, T):
    M, d_vhs, w, d_vss, a = SPECIES[sp]
    m = M * AMU
    if model != "VSS":
        a, d = 1.0, d_vhs
    else:
        d = d_vss
    return (5 * (a + 1) * (a + 2) * math.sqrt(math.pi * m * K_B * 273.0) /
            (4 * a * (5 - 2 * w) * (7 - 2 * w) * math.pi * d * d) * (T / 273.0) ** w)


def profiles(run, nx, ny):
    """Mean u_x(y) and T_tr(y) over the sampling-phase windows (averaged over x, z)."""
    files = sorted(glob.glob(os.path.join(run, "dsmc_state", "dsmc_state_*.vtkhdf")))
    u = np.zeros(ny)
    T = np.zeros(ny)
    w = np.zeros(ny)
    used = 0
    for f in files:
        with h5py.File(f, "r") as h:
            g = h["VTKHDF"]
            if int(np.asarray(g["FieldData"]["SamplingPhase"]).ravel()[0]) != 1:
                continue
            c = np.asarray(g["CellData"]["cell_index"]).astype(np.int64).ravel()
            vel = np.asarray(g["CellData"]["velocity_mps"]).reshape(-1, 3)
            tt = np.asarray(g["CellData"]["temperature_trans_K"]).ravel()
            n = np.asarray(g["CellData"]["number_density_m3"]).ravel()
        iy = (c // nx) % ny
        ok = np.isfinite(tt) & (n > 0)
        np.add.at(u, iy[ok], n[ok] * vel[ok, 0])
        np.add.at(T, iy[ok], n[ok] * tt[ok])
        np.add.at(w, iy[ok], n[ok])
        used += 1
    if used == 0:
        sys.exit("error: no cell-state files of the sampling phase (dsmc.dt_DSMCState_out, HDF5 build)")
    return u / w, T / w, used


def fit(y, v):
    A = np.vstack([y, np.ones_like(y)]).T
    (k, b), res, *_ = np.linalg.lstsq(A, v, rcond=None)
    n = len(y)
    s2 = float(res[0]) / (n - 2) if len(res) else 0.0
    se = math.sqrt(s2 / np.sum((y - y.mean()) ** 2))
    return k, se


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--core", type=float, default=0.3, help="fit region |y| < core * H")
    a = ap.parse_args()

    res = json.load(open(os.path.join(a.run, "results.json")))
    cfg = yaml.safe_load(open(os.path.join(a.run, "resolved_config.yaml")))
    lo, hi = np.array(cfg["domain"]["box_min_m"], float), np.array(cfg["domain"]["box_max_m"], float)
    dx = np.array(res["diagnostics"]["cell_size_m"], float)
    nx, ny, nz = (int(round(e)) for e in (hi - lo) / dx)
    species = list(cfg["gas"]["species"].keys())
    if len(species) != 1 or species[0] not in SPECIES:
        sys.exit("error: single-species Ar or He runs only")
    sp = species[0]
    model = cfg["dsmc"].get("collision_model", "VHS")
    walls = {w["face"]: w for w in res["diagnostics"].get("walls", [])}
    if "y_min" not in walls or "y_max" not in walls:
        sys.exit("error: needs wall faces y_min and y_max")
    H = hi[1] - lo[1]
    yc = lo[1] + (np.arange(ny) + 0.5) * dx[1]
    ymid = 0.5 * (lo[1] + hi[1])
    core = np.abs(yc - ymid) < a.core * H
    u, T, nfiles = profiles(a.run, nx, ny)
    top = walls["y_max"]
    A = top["area_m2"]
    lam = res["diagnostics"]["mean_free_path_inf_m"]
    print(f"{a.run}: {sp} {model}, H = {H * 1e3:.2f} mm, Kn = {lam / H:.4f}, {nfiles} cell-state files, "
          f"{int(core.sum())} core cells")

    moving = abs(top["velocity_mps"][0] - walls["y_min"]["velocity_mps"][0]) > 0
    if moving:  # Couette
        tau = -top["force_x_N"]["value"] / A
        tau_se = top["force_x_N"]["std_error"] / A
        dudy, dudy_se = fit(yc[core], u[core])
        Tc = float(T[core].mean())
        mu = tau / dudy
        se = mu * math.hypot(tau_se / tau, dudy_se / dudy)
        ref = viscosity_ce(sp, model, Tc)
        print(f"  wall shear {tau:.5g} +- {tau_se:.2g} Pa, du/dy {dudy:.5g} +- {dudy_se:.2g} 1/s, core T {Tc:.2f} K")
        print(f"  mu = {mu:.5g} +- {se:.2g} Pa s, Chapman-Enskog {ref:.5g} Pa s: {100 * (mu / ref - 1):+.2f} % "
              f"({(mu - ref) / se:+.1f} sigma)")
    else:  # Fourier
        M, _, w, _, _ = SPECIES[sp]
        q = top["heat_rate_W"]["value"] / A
        q_se = top["heat_rate_W"]["std_error"] / A
        dth, dth_se = fit(yc[core], T[core] ** (1 + w))
        kref = 15 * K_B / (4 * M * AMU) * viscosity_ce(sp, model, 273.0)
        q_ce = -kref * 273.0 ** (-w) / (1 + w) * dth
        r = q / q_ce
        se = r * math.hypot(q_se / q, dth_se / dth)
        print(f"  heat flux into the cold wall {q:.5g} +- {q_se:.2g} W/m2, Chapman-Enskog with the core profile "
              f"{q_ce:.5g} W/m2")
        print(f"  kappa / kappa_CE = {r:.4f} +- {se:.4f} ({(r - 1) / se:+.1f} sigma)")


if __name__ == "__main__":
    main()
