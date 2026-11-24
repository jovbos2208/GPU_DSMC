#!/usr/bin/env python3
"""Convert DSMC cell-state files (VTKHDF unstructured grids) to VTK XML .vtu files.

The solver writes dsmc.dt_DSMCState_out output as VTKHDF (HDF5) files, which
ParaView >= 5.11 opens directly. This script produces classic .vtu files for
older ParaView versions or other tools, plus a .pvd collection with the
physical times when a .series file is given.

Usage:
  scripts/vtkhdf_to_vtu.py results/run/dsmc_state/dsmc_state.vtkhdf.series
  scripts/vtkhdf_to_vtu.py results/run/dsmc_state/dsmc_state_000003.vtkhdf [...]
  options: --output-dir DIR (default: next to the input files)

Requires numpy and h5py. Output: binary (base64) VTU, UInt64 headers.
"""

import argparse
import base64
import json
import pathlib
import sys
import xml.sax.saxutils

try:
    import h5py
    import numpy as np
except ImportError as e:  # pragma: no cover
    sys.exit(f"error: {e.name} is required (pip install h5py numpy)")

VTK_TYPES = {
    np.dtype("float64"): "Float64",
    np.dtype("float32"): "Float32",
    np.dtype("int64"): "Int64",
    np.dtype("int32"): "Int32",
    np.dtype("uint8"): "UInt8",
}


def encode(a: np.ndarray) -> str:
    raw = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
    header = np.array([len(raw)], dtype="<u8").tobytes()
    return base64.b64encode(header + raw).decode("ascii")


def data_array(name: str, a: np.ndarray, components: int = 1) -> str:
    t = VTK_TYPES[a.dtype]
    attr = f' Name="{xml.sax.saxutils.escape(name)}"' if name else ""
    return (f'        <DataArray type="{t}"{attr} NumberOfComponents="{components}" format="binary">'
            f"{encode(a)}</DataArray>\n")


def convert(src: pathlib.Path, dst: pathlib.Path) -> None:
    with h5py.File(src, "r") as f:
        g = f["VTKHDF"]
        kind = g.attrs["Type"]
        kind = kind.decode() if isinstance(kind, bytes) else str(kind)
        if kind != "UnstructuredGrid":
            raise ValueError(f"{src}: VTKHDF type {kind!r} is not supported (UnstructuredGrid expected)")
        points = g["Points"][()]
        conn = g["Connectivity"][()].astype(np.int64)
        offsets = g["Offsets"][()].astype(np.int64)
        types = g["Types"][()].astype(np.uint8)
        cell = {k: g["CellData"][k][()] for k in g["CellData"]} if "CellData" in g else {}
        field = {k: g["FieldData"][k][()] for k in g["FieldData"]} if "FieldData" in g else {}

    n_points, n_cells = len(points), len(types)
    out = ['<?xml version="1.0"?>\n',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n',
           "  <UnstructuredGrid>\n"]
    if field:
        out.append("    <FieldData>\n")
        for k, v in field.items():
            v = np.atleast_1d(np.asarray(v, dtype=np.float64))
            out.append(f'      <DataArray type="Float64" Name="{xml.sax.saxutils.escape(k)}" '
                       f'NumberOfTuples="{len(v)}" format="binary">{encode(v)}</DataArray>\n')
        out.append("    </FieldData>\n")
    out.append(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n')
    out.append("      <Points>\n" + data_array("Points", points.astype(np.float64), 3) + "      </Points>\n")
    # VTU offsets are the end positions of each cell (VTKHDF: begin positions, length C + 1)
    out.append("      <Cells>\n" + data_array("connectivity", conn) + data_array("offsets", offsets[1:])
               + data_array("types", types) + "      </Cells>\n")
    out.append("      <CellData>\n")
    for k, v in cell.items():
        v = np.asarray(v, dtype=np.float64)
        out.append(data_array(k, v.reshape(n_cells, -1), 1 if v.ndim == 1 else v.shape[1]))
    out.append("      </CellData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")
    dst.write_text("".join(out), encoding="ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=pathlib.Path, help=".vtkhdf files or one .vtkhdf.series file")
    ap.add_argument("--output-dir", type=pathlib.Path, help="directory for the .vtu (and .pvd) files")
    a = ap.parse_args()

    series = None
    files = []
    for p in a.inputs:
        if p.name.endswith(".series"):
            series = json.loads(p.read_text())
            files += [(p.parent / e["name"], float(e["time"])) for e in series["files"]]
            series_name = p.name[: -len(".vtkhdf.series")] if p.name.endswith(".vtkhdf.series") else p.stem
            series_dir = p.parent
        else:
            files.append((p, None))
    if not files:
        print("error: no input files", file=sys.stderr)
        return 1

    written = []
    for src, t in files:
        out_dir = a.output_dir or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (src.stem + ".vtu")
        convert(src, dst)
        written.append((dst, t))
        print(f"wrote {dst}")

    if series is not None:
        out_dir = a.output_dir or series_dir
        pvd = out_dir / f"{series_name}.pvd"
        lines = ['<?xml version="1.0"?>\n', '<VTKFile type="Collection" version="1.0">\n', "  <Collection>\n"]
        for dst, t in written:
            try:
                rel = dst.relative_to(out_dir)
            except ValueError:
                rel = dst.resolve()
            lines.append(f'    <DataSet timestep="{t!r}" part="0" file="{rel}"/>\n')
        lines.append("  </Collection>\n</VTKFile>\n")
        pvd.write_text("".join(lines), encoding="ascii")
        print(f"wrote {pvd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
