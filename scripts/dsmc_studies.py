#!/usr/bin/env python3
"""DSMC verification studies (implementation plan M3.16, M3.17).

  dilution     DSMC at decreasing density converges to the free-molecular TPMC result
  cells        cell-size convergence (time step and particles per cell fixed)
  timestep     time-step convergence (cells and particles per cell fixed)
  particles    particles-per-cell convergence (cells and time step fixed)

Body: 0.5 m cube, helium at 900 K, 7.5 km/s, diffuse wall at 300 K.
Usage: dsmc_studies.py <vleo_solver> <work dir> [study ...]   (prints Markdown tables)
"""
import json
import pathlib
import subprocess
import sys

CUBE = """solid cube
{facets}endsolid cube
"""


def cube_stl(path, h=0.25):
    v = lambda i, j, k: ((h if i else -h), (h if j else -h), (h if k else -h))
    quads = [[(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
             [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
             [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]]
    out = []
    for q in quads:
        a, b, c, d = (v(*p) for p in q)
        for t in ((a, b, c), (a, c, d)):
            out.append("facet normal 0 0 0\nouter loop\n" + "".join(
                f"vertex {x} {y} {z}\n" for x, y, z in t) + "endloop\nendfacet\n")
    path.write_text(CUBE.format(facets="".join(out)))


def config(mode, n, extra_dsmc="", seed=7, stl="cube.stl"):
    return f"""simulation: {{name: study, mode: {mode}, random_seed: {seed}}}
geometry: {{file: {stl}}}
domain: {{margin_m: 0.15}}
reference: {{area_m2: 0.25, length_m: 0.5, moment_center_m: [0, 0, 0], lift_axis_body: [0, 0, 1]}}
gas: {{number_density_m3: {n:.6e}, translational_temperature_K: 900, bulk_velocity_body_mps: [-7500, 0, 0], species: {{He: 1.0}}}}
surface: {{model: maxwell, temperature_K: 300, maxwell: {{diffuse_fraction: 1.0}}}}
tpmc: {{particle_histories: 20000000}}
dsmc: {{blocks: 20{extra_dsmc}}}
"""


def run(solver, work, name, text):
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(text)
    subprocess.run([solver, "-q", "-c", str(d / "config.yaml"), "-o", str(d / "out")], check=True,
                   stderr=subprocess.DEVNULL)
    return json.loads((d / "out" / "results.json").read_text())


def row(label, r, ref=None):
    res, se, dg = r["results"], r["std_error"], r.get("diagnostics", {})
    cd, q = res["C_D"], res["heat_rate_W"]
    s = f"| {label} | {cd:.4f} ± {se['C_D']:.4f} | {q:.4g} ± {se['heat_rate_W']:.2g} |"
    if ref is not None:
        s += f" {100 * (cd / ref['results']['C_D'] - 1):+.2f} % |"
    if "mean_free_path_inf_m" in dg:
        s += (f" {dg['mean_free_path_inf_m']:.3g} | {dg['cell_size_m'][0]:.3g} | {dg['timestep_s']:.3g} |"
              f" {dg['mean_particles_per_gas_cell']:.1f} | {dg['collisions']['collisions_per_particle_per_step']:.3g} |")
    return s


HEADER = ("| case | C_D | Q̇ [W] | ΔC_D | λ∞ [m] | Δx [m] | Δt [s] | particles/cell | coll./particle/step |\n"
          "| ---- | --- | ----- | ---- | ------ | ------ | ------ | -------------- | ------------------- |")


def main():
    solver, work = sys.argv[1], pathlib.Path(sys.argv[2]).resolve()
    studies = sys.argv[3:] or ["dilution", "cells", "timestep", "particles"]
    work.mkdir(parents=True, exist_ok=True)
    cube_stl(work / "cube.stl")
    stl = str(work / "cube.stl")
    tpmc = run(solver, work, "tpmc", config("tpmc", 1e14, stl=stl))
    print(f"TPMC (free-molecular reference): C_D = {tpmc['results']['C_D']:.4f} ± {tpmc['std_error']['C_D']:.4f}\n")
    steps = ", warmup_steps: 400, sample_steps: 4000"
    if "dilution" in studies:
        print("### Dilution: DSMC → TPMC as n → 0 (ΔC_D relative to TPMC)\n\n" + HEADER)
        for n in (1e14, 1e17, 1e18, 1e19, 1e20):
            r = run(solver, work, f"dil_{n:.0e}", config("dsmc", n, steps, stl=stl))
            print(row(f"n = {n:.0e} m⁻³", r, tpmc))
        print()
    n = 1e19  # Kn ~ 1 based on the cube edge
    base = ", particles_per_cell_target: 30"
    if "cells" in studies:
        print(f"### Cell size (n = {n:.0e}, fixed Δt = 2e-6 s, 30 particles/cell; ΔC_D relative to the finest)\n\n" + HEADER)
        rows = []
        for dx in (0.2, 0.1, 0.05, 0.025):
            rows.append((dx, run(solver, work, f"dx_{dx}", config(
                "dsmc", n, steps + base + f", timestep_mode: fixed, timestep_s: 2.0e-6, cell_size_mode: fixed, cell_size_m: {dx}", stl=stl))))
        for dx, r in rows:
            print(row(f"Δx = {dx} m", r, rows[-1][1]))
        print()
    if "timestep" in studies:
        print(f"### Time step (n = {n:.0e}, Δx = 0.05 m, 30 particles/cell; ΔC_D relative to the smallest Δt)\n\n" + HEADER)
        rows = []
        for dt in (8e-6, 4e-6, 2e-6, 1e-6):
            rows.append((dt, run(solver, work, f"dt_{dt}", config(
                "dsmc", n, steps + base + f", timestep_mode: fixed, timestep_s: {dt}, cell_size_mode: fixed, cell_size_m: 0.05", stl=stl))))
        for dt, r in rows:
            print(row(f"Δt = {dt:g} s", r, rows[-1][1]))
        print()
    if "particles" in studies:
        print(f"### Particles per cell (n = {n:.0e}, Δx = 0.05 m, Δt = 2e-6 s; ΔC_D relative to the largest)\n\n" + HEADER)
        rows = []
        for ppc in (5, 15, 45, 135):
            rows.append((ppc, run(solver, work, f"ppc_{ppc}", config(
                "dsmc", n, steps + f", particles_per_cell_target: {ppc}, timestep_mode: fixed, timestep_s: 2.0e-6, "
                "cell_size_mode: fixed, cell_size_m: 0.05", stl=stl))))
        for ppc, r in rows:
            print(row(f"{ppc} particles/cell", r, rows[-1][1]))
        print()


if __name__ == "__main__":
    main()
