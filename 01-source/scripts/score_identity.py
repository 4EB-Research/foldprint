#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.validation import load_and_validate_inputs
from protein_sae_pipeline.workflows import score_identity


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate direct positional identity to WT.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    path = args.output_root / "scores" / "identity_scores.csv.gz"
    score_identity(inputs, path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
