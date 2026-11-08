#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.io import atomic_csv, atomic_json, sha256_file
from protein_sae_pipeline.representations.store import (
    load_indexed_representations,
    representation_score_frame,
)
from protein_sae_pipeline.validation import load_and_validate_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Score stored ESM-C and SAE representations against same-run WTs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    chunk_path = args.data_root / config.paths["representations"]
    index_path = chunk_path.with_name("representation_index.csv")
    esmc = config.model("esmc")
    sae = config.model("sae")
    store = load_indexed_representations(
        chunk_path,
        index_path,
        inputs,
        config=config,
    )
    scores = representation_score_frame(store, inputs)
    columns = [
        "reference_id", "sequence_sha256", "esmc_cosine_to_wt", "esmc_zero_norm",
        "sae_cosine_to_wt", "sae_profile_norm", "sae_profile_nnz", "sae_zero_norm",
    ]
    output = args.output_root / "scores" / "representation_scores.csv.gz"
    atomic_csv(output, scores.to_dict("records"), fieldnames=columns)
    atomic_json(
        output.with_suffix(output.suffix + ".json"),
        {
            "schema_version": 1,
            "dataset_id": config.dataset_id,
            "complete": len(scores) == len(inputs.candidates),
            "rows": len(scores),
            "chunk_sha256": sha256_file(chunk_path),
            "index_sha256": sha256_file(index_path),
            "esmc": esmc,
            "sae": sae,
            "score_table_sha256": sha256_file(output),
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
