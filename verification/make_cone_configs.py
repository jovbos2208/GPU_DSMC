#!/usr/bin/env python3
"""70-deg blunted cone in Mach 20 nitrogen: the SR3 wind-tunnel conditions computed
with DSMC by Moss, Dogra & Wilmoth (NASA TM-107762, 1993).

    verification/make_cone_configs.py [--cases case1 case2] [--dx 0.4 0.25] [--ppc 5] [--out verification]

Conditions (TM-107762, tables 1 and 5): nitrogen, wall 300 K, fully diffuse,
    case 1: rho = 1.729e-5 kg/m3, V = 1502 m/s, T = 13.32 K (Ma 20.2, Kn 0.032)
    case 2: rho = 5.19e-5 kg/m3,  V = 1502 m/s, T = 14.0 K  (Ma 19.7, Kn 0.011)
(case 3, Kn 0.001, needs cells far below what a uniform grid allows). VHS
d_ref = 4.07e-10 m, T_ref = 300 K, viscosity exponent 0.75 (gas.collision_data);
Larsen-Borgnakke with Z_rot = 5 (as TM-107762; vibration: species database,
TM-107762 used Z_vib = 50 - vibration is frozen at these temperatures).

Geometry (reconstructed from the surface coordinates of table 5, base diameter
d = 0.05 m): spherical nose R_n = 12.5 mm, 70-deg half-angle cone, corner radius
R_c = 1.25 mm, flat base at x = 10.42 mm (nose at x = 0); no sting (TM-107762
"forebody & wake" runs). Body of revolution about x, 120 facets around.

Set-up in the solver: quarter model (y, z >= 0, symmetry planes specular), the
body cut by the box (domain.clip_geometry), inflow faces x_min, x_max, y_max,
z_max with the free stream; reference area = quarter of pi d^2 / 4, so the
solver's C_D is the full-body value. Uniform cells of --dx mm; one config per
case and cell size.

Writes cone70.stl, cone_<case>_dx<dx>.yaml and batch_cone.txt; evaluate with
verification/cone_eval.py.
"""
import argparse
import math
import os

K_B = 1.380649e-23
M_N2 = 28.02e-3 / 6.02214076e23  # TM-107762: M = 28.02 g/mol
D_REF, OMEGA, T_REF = 4.07e-10, 0.75, 300.0
T_W = 300.0
R_B, R_N, R_C = 0.025, 0.0125, 0.00125
HALF = math.radians(70.0)
N_PHI = 120  # multiple of 4: facet edges on y = 0 and z = 0

CASES = {
    # rho, V, T, domain [x_min, x_max, r_max] (m)
    "case1": (1.729e-5, 1502.0, 13.32, (-0.025, 0.025, 0.045)),
    "case2": (5.19e-5, 1502.0, 14.0, (-0.015, 0.022, 0.04)),
}


def profile():
    """(x, r) points of the body contour from the nose to the axis behind the base."""
    a = math.pi / 2 - HALF  # 20 deg: angle of the cone surface normal to the axis
    pts = []
    for k in range(0, 11):  # nose arc 0 .. 20 deg
        t = a * k / 10
        pts.append((R_N * (1 - math.cos(t)), R_N * math.sin(t)))
    x0, r0 = pts[-1]
    yc = R_B - R_C
    xc = x0 + math.tan(a) * (yc - r0) + R_C / math.cos(a)
    xt, rt = xc - R_C * math.cos(a), yc + R_C * math.sin(a)  # cone / corner tangent point
    for k in range(1, 25):  # cone
        f = k / 24
        pts.append((x0 + f * (xt - x0), r0 + f * (rt - r0)))
    for k in range(1, 12):  # corner arc: normal angle 110 deg -> 0 deg
        al = math.radians(110.0 * (1 - k / 11))
        pts.append((xc + R_C * math.cos(al), yc + R_C * math.sin(al)))
    xb = xc + R_C
    for k in range(1, 13):  # base plane down to the axis
        pts.append((xb, yc * (1 - k / 12)))
    pts[-1] = (xb, 0.0)
    return pts, xb


def write_stl(path):
    pts, _ = profile()
    out = ["solid cone70"]

    def v(i, j):
        x, r = pts[i]
        if r == 0.0:
            return (x, 0.0, 0.0)
        ph = 2 * math.pi * (j % N_PHI) / N_PHI
        return (x, r * math.cos(ph), r * math.sin(ph))

    def tri(a, b, c):
        ux, uy, uz = (b[k] - a[k] for k in range(3))
        wx, wy, wz = (c[k] - a[k] for k in range(3))
        nx, ny, nz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln == 0.0:
            return
        out.append(f" facet normal {nx / ln:.9e} {ny / ln:.9e} {nz / ln:.9e}")
        out.append("  outer loop")
        for q in (a, b, c):
            out.append(f"   vertex {q[0]:.9e} {q[1]:.9e} {q[2]:.9e}")
        out.append("  endloop")
        out.append(" endfacet")

    # orientation: d/dphi x d/ds points outwards (s: from the nose to the base)
    for i in range(len(pts) - 1):
        for j in range(N_PHI):
            a, b = v(i, j), v(i, j + 1)
            c, d = v(i + 1, j), v(i + 1, j + 1)
            if pts[i][1] > 0.0:
                tri(a, b, c)
            if pts[i + 1][1] > 0.0:
                tri(b, d, c)
    out.append("endsolid cone70")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def vhs_lambda_tau(n, T):
    mr = 0.5 * M_N2
    sg = 2 * math.sqrt(math.pi) * D_REF ** 2 * (T / T_REF) ** (1 - OMEGA) * math.sqrt(2 * K_B * T_REF / mr)
    return math.sqrt(8 * K_B * T / (math.pi * M_N2)) / (n * sg), 1.0 / (n * sg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--dx", nargs="+", type=float, default=[0.4, 0.25], help="cell sizes in mm")
    ap.add_argument("--ppc", type=int, default=5, help="particles per cell in the free stream")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    stl = "cone70.stl"
    batch = []
    for case in a.cases:
        rho, U, T, (x0, x1, rm) = CASES[case]
        n = rho / M_N2
        lam, _ = vhs_lambda_tau(n, T)
        # wall density at the stagnation point: pitot pressure ~ 0.92 rho U^2 (gamma 1.4, hypersonic) at T_w
        nw = 0.92 * rho * U * U / (K_B * T_W)
        lam_w, tau_w = vhs_lambda_tau(nw, T_W)
        c_mp = math.sqrt(2 * K_B * T / M_N2)
        c_mp_w = math.sqrt(2 * K_B * T_W / M_N2)
        mach = U / math.sqrt(1.4 * K_B * T / M_N2)
        for dxmm in a.dx:
            dx = dxmm * 1e-3
            dt = min(0.2 * tau_w, dx / (U + 3 * c_mp), 0.5 * dx / (3 * c_mp_w))
            nx, ny = round((x1 - x0) / dx), round(rm / dx)
            cells = nx * ny * ny
            steps = max(4000, int(math.ceil(8 * (x1 - x0) / U / dt / 1000.0)) * 1000)
            tag = f"cone_{case}_dx{dxmm:g}".replace(".", "p")
            text = f"""# 70-deg blunted cone, nitrogen, Moss, Dogra & Wilmoth (NASA TM-107762) {case} (docs/verification.md),
# generated by make_cone_configs.py. rho = {rho:.4g} kg/m3, n = {n:.4g} m^-3, V = {U:g} m/s, T = {T:g} K (Ma {mach:.1f}),
# wall {T_W:g} K diffuse. VHS lambda_inf = {lam * 1e3:.3f} mm; stagnation wall ~ {nw / n:.0f} n_inf, lambda_w ~ {lam_w * 1e3:.3f} mm.
# Quarter model y, z >= 0: {nx} x {ny} x {ny} cells of {dxmm:g} mm (dx / lambda_w = {dx / lam_w:.1f}).
simulation:
  name: {tag}
  mode: dsmc
  random_seed: {20261127 + int(dxmm * 100) + (7 if case == 'case2' else 0)}

geometry:
  file: {stl}
  units: m

domain:
  box_min_m: [{x0:g}, 0.0, 0.0]
  box_max_m: [{x1:g}, {rm:g}, {rm:g}]
  clip_geometry: true          # the cone crosses the symmetry planes y = 0 and z = 0

reference:
  area_m2: {math.pi * R_B ** 2 / 4:.8g}        # quarter of pi d^2 / 4
  length_m: {2 * R_B}
  moment_center_m: [0.0, 0.0, 0.0]
  lift_axis_body: [0.0, 1.0, 0.0]  # quarter model: C_L is not meaningful

gas:
  number_density_m3: {n:.6g}
  translational_temperature_K: {T:g}
  bulk_velocity_body_mps: [{U:g}, 0.0, 0.0]
  species:
    N2: 1.0
  collision_data:
    N2: {{d_ref_m: {D_REF}, omega: {OMEGA}, T_ref_K: {T_REF}}}

surface:
  model: maxwell
  temperature_K: {T_W:g}
  maxwell:
    diffuse_fraction: 1.0

dsmc:
  boundary: specular            # y_min, z_min: symmetry planes
  faces:
    x_min: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:g}, velocity_mps: [{U:g}, 0, 0]}}
    x_max: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:g}, velocity_mps: [{U:g}, 0, 0]}}
    y_max: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:g}, velocity_mps: [{U:g}, 0, 0]}}
    z_max: {{type: inflow, number_density_m3: {n:.6g}, temperature_K: {T:g}, velocity_mps: [{U:g}, 0, 0]}}
  collision_model: VHS
  timestep_mode: fixed
  timestep_s: {dt:.4g}
  cell_size_mode: fixed
  cell_size_m: {dx:.6g}
  particles_per_cell_target: {a.ppc}
  warmup_steps: {steps}
  sample_steps: {steps}
  blocks: 10

statistics:
  confidence_level: 0.95
  report_uncertainty: true

output:
  directory: results/{tag}
  format: json
  save_surface_fields: true
"""
            with open(os.path.join(a.out, tag + ".yaml"), "w") as f:
                f.write(text)
            batch.append(f"{tag}.yaml results/{tag}")
            print(f"{tag}: Ma {mach:.2f}, lambda_inf {lam * 1e3:.3f} mm, lambda_w ~ {lam_w * 1e3:.3f} mm, "
                  f"dx/lambda_w {dx / lam_w:.1f}, {nx} x {ny} x {ny} = {cells / 1e6:.2f} M cells, "
                  f"~{cells * a.ppc / 1e6:.0f} M free-stream particles, dt {dt:.3g} s (dt/tau_w {dt / tau_w:.2f}), "
                  f"2 x {steps} steps")
    write_stl(os.path.join(a.out, stl))
    with open(os.path.join(a.out, "batch_cone.txt"), "w") as f:
        f.write("\n".join(batch) + "\n")
    _, xb = profile()
    print(f"base plane at x = {xb * 1e3:.2f} mm (TM-107762: 10.4 mm)")


if __name__ == "__main__":
    main()
