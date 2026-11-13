#!/usr/bin/env python3
"""Automated check of the shipped examples with fixed seeds (M5.10).

Runs every configuration in configs/simulations/ with its fixed random_seed and
a shortened run length, then compares all main quantities with the stored
reference values in tests/golden/<name>.json:

  * identical build/backend as the reference: values must agree to 1e-9
    relative (seeded runs are bitwise reproducible, docs/conventions.md);
  * any other backend/compiler: statistical agreement, z <= 5 with
    z = |a - b| / sqrt(SE_a^2 + SE_b^2) (floating-point differences between
    backends change the individual random histories, not the estimate).

  check_examples.py <vleo_solver> [--only NAME ...] [--update]
    --update  rewrites the reference files from this run (review the diff!)
Exit code 1 if any example disagrees.
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
CONFIGS = ROOT / "configs" / "simulations"
QUANTITIES = ["C_D", "C_L", "C_Y", "heat_rate_W"]
VECTORS = ["force_N", "moment_Nm"]
# shortened run lengths: a few seconds per example on a CPU
OVERRIDES = {"tpmc": ["--histories", "400000"], "dsmc": ["--warmup-steps", "60", "--sample-steps", "120"]}


def flatten(block):
    out = {q: block[q] for q in QUANTITIES}
    for v in VECTORS:
        for i, c in enumerate("xyz"):
            out[f"{v}[{c}]"] = block[v][i]
    return out


def mode_of(cfg):
    for line in cfg.read_text().splitlines():
        s = line.split("#")[0].strip()
        if s.startswith("mode:"):
            return s.split(":", 1)[1].strip()
    raise SystemExit(f"{cfg}: no simulation mode")


def run(solver, cfg, out):
    cmd = [solver, "-q", "-c", str(cfg), "-o", str(out), *OVERRIDES[mode_of(cfg)]]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return json.loads((out / "results.json").read_text())


def signature(out):
    """Everything that may change the floating-point sequence of a seeded run."""
    m = json.loads((out / "run_metadata.json").read_text())
    b, rt = m["build"], m["runtime"]
    return {"compiler": b["compiler"]["detail"], "flags": b["compiler"]["flags"], "build_type": b["build_type"],
            "kokkos": b["kokkos"]["version"], "arch": b["kokkos"]["arch"], "execution_space": rt["execution_space"],
            "concurrency": rt["concurrency"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solver")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--z-max", type=float, default=5.0)
    args = ap.parse_args()
    names = args.only or sorted(p.stem for p in CONFIGS.glob("*.yaml"))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vleo_examples_"))
    failed = []
    for name in names:
        r = run(args.solver, CONFIGS / f"{name}.yaml", tmp / name)
        val, se = flatten(r["results"]), flatten(r["std_error"])
        ref_path = GOLDEN / f"{name}.json"
        if args.update:
            GOLDEN.mkdir(parents=True, exist_ok=True)
            ref = {"example": name, "run_length": OVERRIDES[r["simulation"]["mode"]],
                   "seed": r["simulation"]["random_seed"], "build": signature(tmp / name), "results": val, "std_error": se}
            ref_path.write_text(json.dumps(ref, indent=2) + "\n")
            print(f"[{name}] reference written")
            continue
        ref = json.loads(ref_path.read_text())
        same_build = ref["build"] == signature(tmp / name)
        worst, bad = 0.0, []
        for k, a in val.items():
            b, s = ref["results"][k], math.hypot(se[k], ref["std_error"][k])
            scale = max(abs(a), abs(b))
            rel = abs(a - b) / scale if scale > 0 else 0.0
            z = abs(a - b) / s if s > 0 else (0.0 if a == b else math.inf)
            ok = rel <= 1e-9 if same_build else (rel <= 1e-9 or z <= args.z_max)
            worst = max(worst, rel if same_build else z)
            if not ok:
                bad.append(f"{k}: {a:.10g} vs reference {b:.10g} (rel {rel:.1e}, z {z:.1f})")
        kind = "bitwise (same build)" if same_build else f"statistical (max z = {worst:.2f})"
        print(f"[{name}] {'ok' if not bad else 'FAILED'}: {kind}, C_D = {val['C_D']:.6f} ± {se['C_D']:.2g}")
        for b in bad:
            print("    " + b)
        if bad:
            failed.append(name)
    if failed:
        print("examples disagreeing with the reference:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
