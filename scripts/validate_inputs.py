#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.io import atomic_json
from protein_sae_pipeline.validation import load_and_validate_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all pipeline input manifests.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    counts = {
        reference_id: sum(item.reference_id == reference_id for item in inputs.candidates)
        for reference_id in sorted(inputs.references)
    }
    report = {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "complete": True,
        "candidates": len(inputs.candidates),
        "references": len(inputs.references),
        "structures": len(inputs.structures),
        "counts_by_reference": counts,
    }
    atomic_json(args.output_root / "validation" / "input_validation.json", report)
    print(f"Validated {len(inputs.candidates)} candidates across {len(inputs.references)} references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
