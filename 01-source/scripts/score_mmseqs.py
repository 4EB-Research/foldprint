#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.validation import load_and_validate_inputs
from protein_sae_pipeline.workflows import score_mmseqs


def main() -> int:
    parser = argparse.ArgumentParser(description="Score candidates against their single cognate WT with MMseqs2.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--executable", default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    path = args.output_root / "scores" / "mmseqs_scores.csv.gz"
    score_mmseqs(inputs, config, path, executable=args.executable, threads=args.threads)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
