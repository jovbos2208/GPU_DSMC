#!/usr/bin/env python3
"""Sphere drag over the Knudsen number in supersonic argon (docs/verification.md, stage 3).

    verification/make_sphere_configs.py [--mach 4] [--kn 10 3 1 0.3 0.1 0.03] [--out verification]

Conditions of a low-density wind tunnel (Aroesty 1963 measured argon spheres in
such a tunnel): stagnation temperature T0 = 300 K, wall at T0, fully diffuse;
Mach 4 by default (T_inf = 47.4 K, U_inf = 513 m/s). VHS argon from the species
database (Bird 1994: d_ref = 4.17e-10 m, omega = 0.81, T_ref = 273 K). The sphere
diameter is 0.1 m; Kn = lambda_VHS,inf / D sets the density.

Set-up in the solver: quarter model (y >= 0, z >= 0, the symmetry planes are
specular faces), the sphere cut by the box (domain.clip_geometry), inflow faces
x_min, x_max, y_max, z_max with the free stream; the box reaches
min(4, 1 + 6 lambda_inf / D) D upstream and sideways and 1.5 D downstream. reference.area_m2 is the
quarter of pi R^2, so the solver's C_D is the drag coefficient of the sphere.
The free-molecular limit is a TPMC run of the full sphere (sphere_fm.yaml).
Cells D/20, refined to lambda_wall/2 near the body where needed (uniform grid);
lambda_wall from the Rayleigh pitot pressure at the wall temperature.

Writes sphere_d100mm.stl, sphere_ma<M>_kn<Kn>.yaml, sphere_fm.yaml and
batch_sphere.txt. Evaluate with verification/sphere_eval.py.
"""
import argparse
import math
import os

K_B = 1.380649e-23
M_AR = 39.948 * 1.66053906660e-27
GAMMA = 5.0 / 3.0
D_REF, OMEGA, T_REF = 4.17e-10, 0.81, 273.0
D_SPH = 0.1
R_SPH = 0.5 * D_SPH
T0 = 300.0
N_THETA, N_PHI = 90, 180  # 2 deg facets; N_PHI multiple of 4: facet edges on y = 0 and z = 0


def vhs_lambda_tau(n, T):
    mr = 0.5 * M_AR
    sg = 2 * math.sqrt(math.pi) * D_REF ** 2 * (T / T_REF) ** (1 - OMEGA) * math.sqrt(2 * K_B * T_REF / mr)
    return math.sqrt(8 * K_B * T / (math.pi * M_AR)) / (n * sg), 1.0 / (n * sg)


def pitot_ratio(M, g=GAMMA):
    """Rayleigh pitot formula p02 / p_inf."""
    a = ((g + 1) ** 2 * M * M / (4 * g * M * M - 2 * (g - 1))) ** (g / (g - 1))
    return a * (1 - g + 2 * g * M * M) / (g + 1)


def write_stl(path):
    """UV sphere, axis along x (flow direction), outward normals."""
    out = ["solid sphere"]

    def p(i, j):  # shared vertices are bitwise identical (closed mesh)
        if i == 0 or i == N_THETA:
            return (-R_SPH if i == 0 else R_SPH, 0.0, 0.0)
        j %= N_PHI
        th = math.pi * i / N_THETA
        ph = 2 * math.pi * j / N_PHI
        return (-R_SPH * math.cos(th), R_SPH * math.sin(th) * math.cos(ph), R_SPH * math.sin(th) * math.sin(ph))

    def tri(a, b, c):
        ux, uy, uz = (b[k] - a[k] for k in range(3))
        vx, vy, vz = (c[k] - a[k] for k in range(3))
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        cx, cy, cz = ((a[k] + b[k] + c[k]) / 3 for k in range(3))
        if nx * cx + ny * cy + nz * cz < 0:  # make the normal point outwards
            b, c = c, b
            nx, ny, nz = -nx, -ny, -nz
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        out.append(f" facet normal {nx / ln:.9e} {ny / ln:.9e} {nz / ln:.9e}")
        out.append("  outer loop")
        for q in (a, b, c):
            out.append(f"   vertex {q[0]:.9e} {q[1]:.9e} {q[2]:.9e}")
        out.append("  endloop")
        out.append(" endfacet")

    for i in range(N_THETA):
        for j in range(N_PHI):
            a, b = p(i, j), p(i, j + 1)
            c, d = p(i + 1, j), p(i + 1, j + 1)
            if i > 0:
                tri(a, c, b)
            if i < N_THETA - 1:
                tri(b, c, d)
    out.append("endsolid sphere")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


COMMON = """surface:
  model: maxwell
  temperature_K: {Tw:g}
  maxwell:
    diffuse_fraction: 1.0

statistics:
  confidence_level: 0.95
  report_uncertainty: true
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mach", type=float, default=4.0)
    ap.add_argument("--kn", nargs="+", type=float, default=[10, 3, 1, 0.3, 0.1, 0.03])
    ap.add_argument("--ppc", type=int, default=10, help="particles per cell in the free stream")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    M = a.mach
    T = T0 / (1 + 0.5 * (GAMMA - 1) * M * M)
    Tw = T0
    U = M * math.sqrt(GAMMA * K_B * T / M_AR)
    c_mp = math.sqrt(2 * K_B * T / M_AR)
    c_mp_w = math.sqrt(2 * K_B * Tw / M_AR)
    mtag = f"{M:g}".replace(".", "p")
    stl = "sphere_d100mm.stl"
    batch = []
    for kn in a.kn:
        lam_ref, _ = vhs_lambda_tau(1.0, T)
        n = lam_ref / (kn * D_SPH)
        lam, tau = vhs_lambda_tau(n, T)
        nw = pitot_ratio(M) * T / Tw * n
        lam_w, tau_w = vhs_lambda_tau(nw, Tw)
        dx = min(D_SPH / 20, 0.5 * lam_w) if kn > 0.05 else min(D_SPH / 20, 0.75 * lam_w)
        dx = D_SPH / math.ceil(D_SPH / dx)  # integer cells per diameter
        dt = min(0.2 * tau_w, dx / (U + 3 * c_mp), 0.5 * dx / (3 * c_mp_w))
        # upstream: stand-off and shock thickness (a few lambda_inf), at most 4 D; downstream 1.5 D
        # (supersonic wake, it does not act on the fore-body)
        up = min(4.0, 1.0 + 6.0 * lam / D_SPH) * D_SPH
        side = up
        down = 1.5 * D_SPH
        x0, x1, ys = -up, down, side
        nx, ny = round((x1 - x0) / dx), round(ys / dx)
        cells = nx * ny * ny
        flow = (x1 - x0) / U
        steps = max(4000, int(math.ceil(5 * flow / dt / 1000.0)) * 1000)
        ktag = f"{kn:g}".replace(".", "p")
        tag = f"sphere_ma{mtag}_kn{ktag}"
        text = f"""# Sphere in argon, Ma {M:g}, Kn = lambda_inf / D = {kn:g} (docs/verification.md), generated by
# make_sphere_configs.py. n_inf = {n:.4g} m^-3, T_inf = {T:.4g} K, U_inf = {U:.5g} m/s, wall {Tw:g} K diffuse.
# VHS lambda_inf = {lam * 1e3:.4g} mm; wall at the stagnation point ~ {nw / n:.2f} n_inf, lambda_w ~ {lam_w * 1e3:.4g} mm.
# Quarter model y, z >= 0: {nx} x {ny} x {ny} cells of {dx * 1e3:.4g} mm. C_D refers to pi R^2 (area = quarter).
simulation:
  name: {tag}
  mode: dsmc
  random_seed: {20261127 + int(round(kn * 100))}

geometry:
  file: {stl}
  units: m

domain:
  box_min_m: [{x0:.6g}, 0.0, 0.0]
  box_max_m: [{x1:.6g}, {ys:.6g}, {ys:.6g}]
  clip_geometry: true          # the sphere crosses the symmetry planes y = 0 and z = 0

reference:
  area_m2: {math.pi * R_SPH ** 2 / 4:.8g}       # quarter of pi R^2
  length_m: {D_SPH}
  moment_center_m: [0.0, 0.0, 0.0]
  lift_axis_body: [0.0, 1.0, 0.0]  # quarter model: C_L is not meaningful

gas:
  number_density_m3: {n:.6g}
  translational_temperature_K: {T:.6g}
  bulk_velocity_body_mps: [{U:.6g}, 0.0, 0.0]
  species:
    Ar: 1.0

{COMMON.format(Tw=Tw)}
dsmc:
  boundary: specular            # y_min, z_min: symmetry planes
  faces:
    x_min: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:.6g}, velocity_mps: [{U:.6g}, 0, 0]}}
    x_max: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:.6g}, velocity_mps: [{U:.6g}, 0, 0]}}
    y_max: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:.6g}, velocity_mps: [{U:.6g}, 0, 0]}}
    z_max: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:.6g}, velocity_mps: [{U:.6g}, 0, 0]}}
  collision_model: VHS
  timestep_mode: fixed
  timestep_s: {dt:.4g}
  cell_size_mode: fixed
  cell_size_m: {dx:.6g}
  particles_per_cell_target: {a.ppc}
  warmup_steps: {steps}
  sample_steps: {steps}
  blocks: 10

output:
  directory: results/{tag}
  format: json
  save_surface_fields: true
"""
        with open(os.path.join(a.out, tag + ".yaml"), "w") as f:
            f.write(text)
        batch.append(f"{tag}.yaml results/{tag}")
        print(f"{tag}: n {n:.3g}, lambda_inf {lam * 1e3:.3g} mm, lambda_w ~ {lam_w * 1e3:.3g} mm, dx {dx * 1e3:.3g} mm "
              f"(D/{D_SPH / dx:.0f}, dx/lambda_w {dx / lam_w:.2f}), {nx} x {ny} x {ny} = {cells / 1e6:.2f} M cells, "
              f"~{cells * a.ppc / 1e6:.1f} M particles, dt {dt:.3g} s (dt/tau_w {dt / tau_w:.2f}), 2 x {steps} steps")
    # free-molecular limit: TPMC of the full sphere
    fm = f"""# Free-molecular limit of the sphere series (TPMC, full sphere), generated by make_sphere_configs.py.
simulation:
  name: sphere_ma{mtag}_fm
  mode: tpmc
  random_seed: 20261127

geometry:
  file: {stl}
  units: m

reference:
  area_m2: {math.pi * R_SPH ** 2:.8g}       # pi R^2
  length_m: {D_SPH}
  moment_center_m: [0.0, 0.0, 0.0]
  lift_axis_body: [0.0, 1.0, 0.0]  # quarter model: C_L is not meaningful

gas:
  number_density_m3: 1.0e16
  translational_temperature_K: {T:.6g}
  bulk_velocity_body_mps: [{U:.6g}, 0.0, 0.0]
  species:
    Ar: 1.0

{COMMON.format(Tw=Tw)}
tpmc:
  particle_histories: 20000000
  max_reflections: 100

output:
  directory: results/sphere_ma{mtag}_fm
  format: json
  save_surface_fields: false
"""
    with open(os.path.join(a.out, f"sphere_ma{mtag}_fm.yaml"), "w") as f:
        f.write(fm)
    batch.insert(0, f"sphere_ma{mtag}_fm.yaml results/sphere_ma{mtag}_fm")
    write_stl(os.path.join(a.out, stl))
    with open(os.path.join(a.out, "batch_sphere.txt"), "w") as f:
        f.write("\n".join(batch) + "\n")
    print(f"T_inf {T:.4g} K, U_inf {U:.5g} m/s, pitot p02/p_inf {pitot_ratio(M):.3g}")


if __name__ == "__main__":
    main()
