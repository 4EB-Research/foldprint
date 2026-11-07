#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.validation import load_and_validate_inputs
from protein_sae_pipeline.workflows import load_pfam_censor_floors, score_pfam


def main() -> int:
    parser = argparse.ArgumentParser(description="Score candidates against frozen cognate Pfam profiles.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--executable", default=None)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument(
        "--censor-floor-table",
        type=Path,
        help=(
            "Frozen input containing the complete landscape's "
            "full-sequence and best-domain floors for every reference. Omit only when "
            "the input is the complete landscape."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    censor_floors = None
    censor_floor_table_sha256 = None
    if args.censor_floor_table is not None:
        censor_floors, censor_floor_table_sha256 = load_pfam_censor_floors(
            args.censor_floor_table,
            set(inputs.references),
        )
    path = args.output_root / "scores" / "pfam_scores.csv.gz"
    score_pfam(
        inputs,
        config,
        path,
        executable=args.executable,
        cpus=args.cpus,
        censor_floors=censor_floors,
        censor_floor_table_sha256=censor_floor_table_sha256,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
