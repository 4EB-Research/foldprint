#!/usr/bin/env python3
"""Optional live Biohub retrieval of layer-60 ESM-C and raw SAE activations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.io import sha256_file
from protein_sae_pipeline.representations.biohub import BiohubClient, BiohubConfig
from protein_sae_pipeline.representations.sae import load_sae_normalization
from protein_sae_pipeline.representations.sparse import build_representation_chunk, save_representation_chunk
from protein_sae_pipeline.representations.store import representation_contract_sha256
from protein_sae_pipeline.validation import load_and_validate_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--normalization-asset",
        type=Path,
        required=True,
        help="Private SAE normalization NPZ; its sibling JSON supplies checksums.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root)
    sae_config = config.model("sae")
    esmc_config = config.model("esmc")
    normalization_metadata = json.loads(args.normalization_asset.with_suffix(".json").read_text())
    normalization = load_sae_normalization(
        args.normalization_asset,
        expected_sha256=normalization_metadata["output_sha256"],
        expected_codebook_size=int(sae_config["codebook_size"]),
        expected_maximum_sha256=str(normalization_metadata["arrays"]["maximum"]["bytes_sha256"]),
        expected_idf_sha256=str(normalization_metadata["arrays"]["idf"]["bytes_sha256"]),
    )
    client = BiohubClient(
        BiohubConfig(
            esmc_model=str(esmc_config["api_alias"]),
            sae_model=str(sae_config["api_alias"]),
            hidden_layer=int(esmc_config["hidden_state_index"]),
            embedding_dimension=int(esmc_config["embedding_dimension"]),
            codebook_size=int(sae_config["codebook_size"]),
            topk=int(sae_config["top_k"]),
        )
    )

    rows: list[tuple[int, str, str, str, str]] = []
    for offset, reference_id in enumerate(sorted(inputs.references), start=1):
        reference = inputs.references[reference_id]
        rows.append((-offset, reference_id, "wt_reference", reference.sequence_sha256, reference.sequence))
    for candidate in inputs.candidates:
        rows.append((candidate.manifest_row, candidate.reference_id, "candidate", candidate.sequence_sha256, candidate.sequence))

    embeddings = []
    residue_matrices = []
    for index, (_manifest_row, reference_id, role, sequence_hash, sequence) in enumerate(rows, start=1):
        encoded = client.encode(sequence)
        layer = client.infer_layer60(sequence, encoded=encoded)
        sae = client.infer_sae(sequence, normalization=normalization, encoded=encoded)
        embeddings.append(layer.mean_embedding)
        residue_matrices.append(sae.residue_activations)
        print(f"[{index}/{len(rows)}] {reference_id} {role} {sequence_hash[:12]}", flush=True)

    chunk = build_representation_chunk(
        manifest_rows=[row[0] for row in rows],
        sequence_hashes=[row[3] for row in rows],
        raw_mean=embeddings,
        residue_matrices=residue_matrices,
    )
    output_dir = args.output_root / "representations"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / "landscape_representations.npz"
    save_representation_chunk(
        chunk_path,
        chunk,
        expected_embedding_dimension=int(esmc_config["embedding_dimension"]),
        expected_topk=int(sae_config["top_k"]),
        codebook_size=int(sae_config["codebook_size"]),
        compressed=True,
    )
    with (output_dir / "representation_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chunk_row", "manifest_row", "reference_id", "role", "sequence_sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        for chunk_row, row in enumerate(rows):
            writer.writerow(
                {
                    "chunk_row": chunk_row,
                    "manifest_row": row[0],
                    "reference_id": row[1],
                    "role": row[2],
                    "sequence_sha256": row[3],
                }
            )
    metadata = {
        "schema_version": 1,
        "storage_schema": "protein_sae_sparse_representations_v1",
        "dataset_id": config.dataset_id,
        "origin": "live_biohub",
        "fictional": False,
        "rows_including_wt": len(rows),
        "candidate_rows": len(inputs.candidates),
        "embedding_dimension": int(esmc_config["embedding_dimension"]),
        "codebook_size": int(sae_config["codebook_size"]),
        "top_k": int(sae_config["top_k"]),
        "chunk_sha256": sha256_file(chunk_path),
        "index_sha256": sha256_file(output_dir / "representation_index.csv"),
        "esmc_contract_sha256": representation_contract_sha256(esmc_config),
        "sae_contract_sha256": representation_contract_sha256(sae_config),
        "normalization_source_sha256": normalization_metadata["source_sha256"],
        "notice": "Live Biohub output bound to the configured model and normalization contracts.",
    }
    (output_dir / "landscape_representations.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
