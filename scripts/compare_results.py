#!/usr/bin/env python3
"""Compares two vleo results.json files (e.g. CPU vs. GPU backend).

For every main quantity it prints both values, the relative difference and the
z-score |a - b| / sqrt(SE_a^2 + SE_b^2). Exit code 0 if all quantities agree:

  * numerically:   |a - b| <= rel_tol * max(|a|, |b|)          (default 1e-9), or
  * statistically: z <= z_max                                    (default 5)

Usage: compare_results.py A.json B.json [--rel-tol 1e-9] [--z-max 5]
"""
import argparse
import json
import math
import sys

QUANTITIES = ["C_D", "C_L", "C_Y", "heat_rate_W", "heat_rate_positive_W", "heat_rate_negative_W"]
VECTORS = ["force_N", "moment_Nm", "C_M"]


def flatten(block):
    out = {}
    for q in QUANTITIES:
        out[q] = block[q]
    for v in VECTORS:
        for i, c in enumerate("xyz"):
            out[f"{v}[{c}]"] = block[v][i]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rel-tol", type=float, default=1e-9)
    ap.add_argument("--z-max", type=float, default=5.0)
    args = ap.parse_args()
    ra, rb = (json.load(open(f)) for f in (args.a, args.b))
    va, vb = flatten(ra["results"]), flatten(rb["results"])
    sa, sb = flatten(ra["std_error"]), flatten(rb["std_error"])
    for tag, r in (("A", ra), ("B", rb)):
        st = r["statistics"]
        size = (f"{st['histories']} histories" if "histories" in st
                else f"{st['sample_steps']} sampled steps")
        print(f"{tag}: {r['simulation']['mode']} on {r['simulation']['backend']} "
              f"({r['provenance']['execution_space']}), {size}")
    print(f"{'quantity':<22}{'A':>24}{'B':>24}{'rel diff':>11}{'z':>8}  verdict")
    ok_all = True
    for k in va:
        a, b = va[k], vb[k]
        scale = max(abs(a), abs(b))
        rel = abs(a - b) / scale if scale > 0 else 0.0
        se = math.hypot(sa[k], sb[k])
        z = abs(a - b) / se if se > 0 else (0.0 if a == b else math.inf)
        if rel <= args.rel_tol:
            verdict = "numerically equal"
        elif z <= args.z_max:
            verdict = "statistically equal"
        else:
            verdict = "DIFFERENT"
            ok_all = False
        print(f"{k:<22}{a:>24.15e}{b:>24.15e}{rel:>11.1e}{z:>8.2f}  {verdict}")
    print("agreement:", "yes" if ok_all else "NO")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
