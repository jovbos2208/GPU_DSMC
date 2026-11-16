#!/usr/bin/env python3
"""Attitude sweep (M6.8): one configuration, several flow directions.

Runs vleo_solver once per (alpha, beta) with --attitude (flight direction
d = (cos a cos b, sin b, sin a cos b) in the body frame, speed from the
configuration) and collects the aerodynamic coefficients with their standard
errors into one table (sweep.json, sweep.csv, Markdown on stdout).

  attitude_sweep.py --solver vleo_solver --config sim.yaml --out DIR
                    --alpha -10 0 10 20 [--beta 0 10]   (all combinations)
                    [--mpiexec mpiexec --ranks 2] [-- extra solver arguments]
Python 3.6 compatible.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys

COLUMNS = [("C_D", ("C_D",)), ("C_L", ("C_L",)), ("C_Y", ("C_Y",)), ("C_Mx", ("C_M", 0)), ("C_My", ("C_M", 1)),
           ("C_Mz", ("C_M", 2)), ("F_x_N", ("force_N", 0)), ("F_y_N", ("force_N", 1)), ("F_z_N", ("force_N", 2)),
           ("Q_W", ("heat_rate_W",))]


def pick(block, key):
    v = block[key[0]]
    return v[key[1]] if len(key) > 1 else v


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0])
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0])
    ap.add_argument("--mpiexec")
    ap.add_argument("--mpiexec-flags", default="")
    ap.add_argument("--ranks", type=int, default=1)
    args = ap.parse_args(argv)

    launch = []
    if args.mpiexec or args.ranks > 1:
        launch = [args.mpiexec or "mpiexec"] + shlex.split(args.mpiexec_flags) + ["-n", str(args.ranks)]
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for b in args.beta:
        for a in args.alpha:
            d = os.path.join(args.out, "a%+g_b%+g" % (a, b))
            cmd = launch + [args.solver, "-q", "-c", args.config, "--attitude", repr(a), repr(b), "-o", d] + extra
            print("$ " + " ".join(cmd), flush=True)
            if subprocess.call(cmd) != 0:
                print("FAIL: alpha %g beta %g" % (a, b))
                return 1
            r = json.load(open(os.path.join(d, "results.json")))
            row = {"alpha_deg": a, "beta_deg": b, "mode": r["simulation"]["mode"]}
            for name, key in COLUMNS:
                row[name] = pick(r["results"], key)
                row[name + "_se"] = pick(r["std_error"], key)
            rows.append(row)

    with open(os.path.join(args.out, "sweep.json"), "w") as f:
        json.dump({"config": os.path.abspath(args.config), "convention":
                   "flight direction d = (cos a cos b, sin b, sin a cos b), gas velocity -|V| d", "rows": rows},
                  f, indent=2)
        f.write("\n")
    with open(os.path.join(args.out, "sweep.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| alpha | beta | C_D | C_L | C_Y | C_Mx | C_My | C_Mz |")
    print("| ----- | ---- | --- | --- | --- | ---- | ---- | ---- |")
    for r in rows:
        cells = ["%.4f ± %.1g" % (r[c], r[c + "_se"]) for c in ("C_D", "C_L", "C_Y", "C_Mx", "C_My", "C_Mz")]
        print("| %g | %g | %s |" % (r["alpha_deg"], r["beta_deg"], " | ".join(cells)))
    print("wrote %s" % os.path.join(args.out, "sweep.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
