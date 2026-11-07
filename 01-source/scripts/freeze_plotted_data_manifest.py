#!/usr/bin/env python3
"""Freeze SHA-256 values for every required Figure 1--3 plotted-data table."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.io import atomic_json, sha256_file
from protein_sae_pipeline.plotted_data import REQUIRED_PLOTTED_DATA_PATHS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing frozen manifest.",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = args.manifest.resolve()
    if manifest.exists() and not args.replace:
        parser.error("--manifest already exists; pass --replace only after scientific review")
    files: dict[str, str] = {}
    for relative in sorted(REQUIRED_PLOTTED_DATA_PATHS):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Required plotted-data path escapes output root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Missing required plotted-data table: {path}")
        files[relative] = sha256_file(path)
    atomic_json(manifest, {"schema_version": 1, "files": files})
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
