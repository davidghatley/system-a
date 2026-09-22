#!/usr/bin/env python3
"""Manifest-driven entry point for iteration 3 inexpensive baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Enforce the workstream CPU-thread ceiling before any optional numeric runtime can load.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

from .core import run_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    result = run_manifest(manifest.resolve(), root)
    print(json.dumps({"result": result["artifacts"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
