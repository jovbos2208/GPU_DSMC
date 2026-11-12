#!/usr/bin/env python3
"""Generates examples/geometry/satellite.stl: a simple box-wing satellite.

Body frame: the satellite flies along +x (gas velocity relative to the body is
-x in the example configurations), +z is the nadir-opposite "up" axis.

  * bus:          1.0 x 1.0 x 1.5 m box centred at the origin
  * panel_left:   2.0 x 0.02 x 1.0 m box at -y, separated from the bus by 0.05 m
  * panel_right:  mirror image at +y

Each part is a separate closed solid (material ids 0, 1, 2). All triangles are
counter-clockwise seen from outside (outward normals). Illustrative geometry
only.
"""
import pathlib
import sys


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = {
        (i, j, k): (x1 if i else x0, y1 if j else y0, z1 if k else z0)
        for i in (0, 1) for j in (0, 1) for k in (0, 1)
    }
    quads = [  # outward-facing, counter-clockwise seen from outside
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],  # z = z0, normal -z
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],  # z = z1, normal +z
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],  # y = y0, normal -y
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],  # y = y1, normal +y
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],  # x = x0, normal -x
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],  # x = x1, normal +x
    ]
    tris = []
    for q in quads:
        a, b, c, d = (v[i] for i in q)
        tris += [(a, b, c), (a, c, d)]
    return tris


def normal(t):
    a, b, c = t
    u = [b[i] - a[i] for i in range(3)]
    w = [c[i] - a[i] for i in range(3)]
    n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
    length = sum(x * x for x in n) ** 0.5
    return [x / length for x in n]


def write(path, solids):
    fmt = lambda p: " ".join(repr(float(x)) for x in p)
    with open(path, "w", newline="\n") as f:
        for name, tris in solids:
            f.write(f"solid {name}\n")
            for t in tris:
                f.write(f"  facet normal {fmt(normal(t))}\n    outer loop\n")
                for p in t:
                    f.write(f"      vertex {fmt(p)}\n")
                f.write("    endloop\n  endfacet\n")
            f.write(f"endsolid {name}\n")


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    out = root / "examples" / "geometry" / "satellite.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    solids = [
        ("bus", box((-0.5, -0.5, -0.75), (0.5, 0.5, 0.75))),
        ("panel_left", box((-0.5, -2.55, -0.01), (0.5, -0.55, 0.01))),
        ("panel_right", box((-0.5, 0.55, -0.01), (0.5, 2.55, 0.01))),
    ]
    write(out, solids)
    print(f"wrote {out} ({sum(len(t) for _, t in solids)} triangles)")


if __name__ == "__main__":
    sys.exit(main())
