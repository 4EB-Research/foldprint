#!/usr/bin/env python3
"""Apply the frozen identity/TM thresholds and assert the reported cohort counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from protein_sae_pipeline.cohorts import construct_cohorts, write_cohorts
from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.master import validate_master_against_inputs
from protein_sae_pipeline.validation import load_and_validate_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    master_path = args.output_root / "tables" / "master.csv.gz"
    master = pd.read_csv(master_path, low_memory=False)
    validate_master_against_inputs(master, inputs, config)
    cohorts = construct_cohorts(master, config)
    audit = write_cohorts(cohorts, args.output_root / "cohorts", config)
    print(args.output_root / "cohorts")
    print(audit["global"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
