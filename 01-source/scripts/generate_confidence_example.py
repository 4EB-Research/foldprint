#!/usr/bin/env python3
"""Create clearly fictional pTM/IUPred fixtures for the bundled candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from protein_sae_pipeline.figures.common import write_frame
from protein_sae_pipeline.io import atomic_json


def generate(data_root: Path) -> None:
    expected = pd.read_csv(data_root / "inputs/candidate_expectations.csv")
    keys = ["reference_id", "sequence_sha256"]
    index = expected.manifest_row.to_numpy()
    analogue = expected.intended_structural_class.eq("analogue").to_numpy()
    confidence = expected[keys].copy()
    confidence["predicted_ptm"] = np.where(analogue, .55 + .04 * (index % 5),
        np.where(index % 4 == 0, .4, .57 + .04 * (index % 3)))
    confidence["fictional"] = True
    disorder = expected[keys].copy()
    disorder["iupred_long_mean"] = .2 + .009 * (index % 13) + .04 * analogue
    disorder["smoothing"] = "medium"
    disorder["fictional"] = True
    write_frame(data_root / "inputs/fictional_confidence.csv", confidence)
    write_frame(data_root / "inputs/fictional_iupred.csv", disorder)
    atomic_json(data_root / "inputs/confidence_example_manifest.json", {
        "fictional": True, "rows": len(expected),
        "description": "Invented pTM and whole-sequence disorder means for software testing; no predictor was run",
        "generator": "scripts/generate_confidence_example.py",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("example"))
    args = parser.parse_args()
    generate(args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
