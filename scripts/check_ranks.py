#!/usr/bin/env python3
"""Multi-rank consistency check (M6.6, docs/parallel.md).

Runs one configuration with 1, 2, ... MPI ranks and compares every run with
the single-rank run:

  * integer counters (diagnostics, collisions, species): identical;
  * main quantities: |a - b| <= rel_tol * max(|a|, |b|) or <= se_frac * SE
    (sums over ranks are rounded differently; default 1e-9 and 1e-6);
  * other floating-point diagnostics and surface_fields.csv: rel_tol, with an
    absolute floor of rel_tol * (largest magnitude of the same field).

  check_ranks.py --solver vleo_solver --mpiexec mpiexec [--mpiexec-flags "..."]
                 --config sim.yaml --out DIR [--ranks 1 2 3 4] [-- solver args]
Exit code 1 on any difference. Python 3.6 compatible.
"""
import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys

# timing and decomposition entries legitimately differ between rank counts
SKIP = {"wall_time_s", "particle_steps_per_s", "histories_per_s", "profile", "parallel"}


def flatten(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in SKIP:
                flatten(v, path + "." + k if path else k, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            flatten(v, "%s[%d]" % (path, i), out)
    else:
        out[path] = node
    return out


def field_of(path):
    """Path without list indices (vector components share one scale)."""
    out, depth = [], 0
    for ch in path:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def compare_json(a, b, rel_tol, se_frac):
    fa = flatten({k: a[k] for k in ("results", "std_error", "diagnostics") if k in a}, "", {})
    fb = flatten({k: b[k] for k in ("results", "std_error", "diagnostics") if k in b}, "", {})
    scale = {}
    for p, v in fa.items():
        if isinstance(v, float):
            f = field_of(p)
            scale[f] = max(scale.get(f, 0.0), abs(v))
    errors = []
    for p in sorted(set(fa) | set(fb)):
        if p not in fa or p not in fb:
            errors.append("%s: only in one run" % p)
            continue
        va, vb = fa[p], fb[p]
        if is_int(va) and is_int(vb) or isinstance(va, (bool, str)) or va is None:
            if va != vb:
                errors.append("%s: %r != %r (must be identical)" % (p, va, vb))
            continue
        va, vb = float(va), float(vb)
        d = abs(va - vb)
        tol = rel_tol * max(abs(va), abs(vb), scale.get(field_of(p), 0.0) if not p.startswith("results.") else 0.0)
        if p.startswith("results."):
            se = fa.get("std_error." + p[len("results."):])
            if isinstance(se, (int, float)):
                tol = max(tol, se_frac * abs(float(se)))
        if not d <= tol:
            errors.append("%s: %.17g vs %.17g (|diff| %.3g > tol %.3g)" % (p, va, vb, d, tol))
    return errors


def read_csv(path):
    with open(path) as f:
        return list(csv.reader(f))


def compare_csv(a, b, rel_tol):
    ra, rb = read_csv(a), read_csv(b)
    if len(ra) != len(rb) or not ra or ra[0] != rb[0]:
        return ["%s: different layout" % os.path.basename(a)]
    cols = len(ra[0])
    scale = [0.0] * cols
    for row in ra[1:]:
        for j, x in enumerate(row):
            try:
                scale[j] = max(scale[j], abs(float(x)))
            except ValueError:
                pass
    errors = []
    for i in range(1, len(ra)):
        for j in range(cols):
            x, y = ra[i][j], rb[i][j]
            if x == y:
                continue
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                errors.append("csv row %d col %s: %r != %r" % (i, ra[0][j], x, y))
                continue
            if not abs(fx - fy) <= rel_tol * max(abs(fx), abs(fy), scale[j]):
                errors.append("csv row %d col %s: %r vs %r" % (i, ra[0][j], x, y))
    return errors


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--mpiexec", required=True)
    ap.add_argument("--mpiexec-flags", default="", help="flags before -n (e.g. --oversubscribe)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--rel-tol", type=float, default=1e-9)
    ap.add_argument("--se-frac", type=float, default=1e-6)
    args = ap.parse_args(argv)

    dirs = []
    for n in args.ranks:
        out = os.path.join(args.out, "ranks_%d" % n)
        cmd = [args.mpiexec] + shlex.split(args.mpiexec_flags) + ["-n", str(n), args.solver, "--quiet",
                                                                  "--config", args.config, "--output-dir", out] + extra
        print("$ " + " ".join(cmd))
        sys.stdout.flush()
        rc = subprocess.call(cmd)
        if rc != 0:
            print("FAIL: run with %d rank(s) exited with %d" % (n, rc))
            return 1
        dirs.append((n, out))

    ref_n, ref = dirs[0]
    ra = json.load(open(os.path.join(ref, "results.json")))
    failed = False
    for n, d in dirs[1:]:
        rb = json.load(open(os.path.join(d, "results.json")))
        ranks = rb.get("diagnostics", {}).get("parallel", {}).get("ranks")
        errors = []
        if ranks != n:
            errors.append("results.json reports %r ranks, expected %d" % (ranks, n))
        errors += compare_json(ra, rb, args.rel_tol, args.se_frac)
        sa, sb = os.path.join(ref, "surface_fields.csv"), os.path.join(d, "surface_fields.csv")
        if os.path.exists(sa) != os.path.exists(sb):
            errors.append("surface_fields.csv written by only one run")
        elif os.path.exists(sa):
            errors += compare_csv(sa, sb, args.rel_tol)
        if errors:
            failed = True
            print("FAIL: %d rank(s) vs %d rank(s): %d difference(s)" % (n, ref_n, len(errors)))
            for e in errors[:40]:
                print("  " + e)
        else:
            print("ok: %d rank(s) agree with %d rank(s)" % (n, ref_n))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
