#!/usr/bin/env bash
# Configure, build and test one preset: scripts/build_and_test.sh <cpu-debug|cpu-release|cuda|hip|hip-gfx1100>
set -euo pipefail
preset="${1:?usage: $0 <cpu-debug|cpu-release|cuda|hip|hip-gfx1100>}"
cd "$(dirname "$0")/.."
cmake --preset "$preset"
cmake --build --preset "$preset" --parallel
ctest --preset "$preset"
