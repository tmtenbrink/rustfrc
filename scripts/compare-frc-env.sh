#!/usr/bin/env bash
# Makes .venv-compare for scripts/compare_frc.py: Python 3.12 (frc supports up to 3.12) with
# rustfrc built from this checkout, DIPlib and SciPy, and frc without its dependencies, so that
# it uses this rustfrc rather than one from PyPI.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf .venv-compare
uv venv -q -p 3.12 .venv-compare
uv pip install -q -p .venv-compare . diplib scipy numpy
uv pip install -q -p .venv-compare --no-deps frc==0.1.3
