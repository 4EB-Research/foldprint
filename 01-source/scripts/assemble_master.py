#!/usr/bin/env python3
"""Assemble score outputs into the master table."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.io import atomic_csv, atomic_json
from protein_sae_pipeline.master import assemble_master, score_table_coverage_report
from protein_sae_pipeline.validation import load_and_validate_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    score_root = args.output_root / "scores"
    sources = {
        "identity": score_root / "identity_scores.csv.gz",
        "structure": score_root / "structure_scores.csv.gz",
        "pfam": score_root / "pfam_scores.csv.gz",
        "mmseqs": score_root / "mmseqs_scores.csv.gz",
        "representations": score_root / "representation_scores.csv.gz",
    }
    master = assemble_master(inputs, config, **sources)
    output = args.output_root / "tables" / "master.csv.gz"
    atomic_csv(output, master.to_dict("records"), fieldnames=list(master.columns))
    audit = score_table_coverage_report(
        master,
        config,
        {name: str(path.relative_to(args.output_root)) for name, path in sources.items()},
    )
    atomic_json(args.output_root / "tables" / "master_coverage.json", audit)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
