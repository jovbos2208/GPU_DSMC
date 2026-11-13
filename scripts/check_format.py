#!/usr/bin/env python3
"""Formatting check run by CI (M5.6).

Enforced rules for all tracked project files (third_party/ excluded):
  * UTF-8, LF line endings, newline at end of file
  * no tab characters in C++/CMake/Python/YAML sources
  * no trailing whitespace
  * C++ lines at most 150 characters (the style target is 120, see .clang-format)

Usage: check_format.py [--list]   exit code 1 on violations.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CPP = {".cpp", ".hpp", ".h", ".in"}
NO_TABS = CPP | {".cmake", ".txt", ".py", ".yaml", ".yml", ".json"}
CHECKED = NO_TABS | {".md", ".sh", ".stl"}
MAX_CPP = 150


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for name in out.splitlines():
        p = pathlib.Path(name)
        if p.parts and p.parts[0] in ("third_party", "IMPLEMENTATION_PLAN.md"):
            continue
        if p.suffix in CHECKED or p.name == "CMakeLists.txt":
            yield p


def main():
    issues = []
    files = list(tracked())
    for rel in files:
        data = (ROOT / rel).read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"{rel}: not valid UTF-8")
            continue
        if not text:
            continue
        if "\r" in text:
            issues.append(f"{rel}: CR line endings")
        if not text.endswith("\n"):
            issues.append(f"{rel}: no newline at end of file")
        for no, line in enumerate(text.split("\n"), 1):
            if line.rstrip() != line:
                issues.append(f"{rel}:{no}: trailing whitespace")
            if "\t" in line and (rel.suffix in NO_TABS or rel.name == "CMakeLists.txt"):
                issues.append(f"{rel}:{no}: tab character")
            if rel.suffix in CPP and len(line) > MAX_CPP:
                issues.append(f"{rel}:{no}: line longer than {MAX_CPP} characters ({len(line)})")
    for i in issues:
        print(i)
    print(f"checked {len(files)} files: {len(issues)} issue(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
