from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from protein_sae_pipeline.config import PipelineConfig, load_config
from protein_sae_pipeline.representations.store import (
    IndexedRepresentations,
    load_indexed_representations,
    representation_score_frame,
)
from protein_sae_pipeline.schemas import ValidatedInputs
from protein_sae_pipeline.scoring.identity import calculate_positional_identity
from protein_sae_pipeline.validation import load_and_validate_inputs


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def fictional_context() -> tuple[
    PipelineConfig,
    ValidatedInputs,
    IndexedRepresentations,
    pd.DataFrame,
]:
    config = load_config(ROOT / "configs" / "example.yaml")
    inputs = load_and_validate_inputs(config, ROOT / "example")
    chunk_path = ROOT / "example" / config.paths["representations"]
    store = load_indexed_representations(
        chunk_path,
        chunk_path.with_name("representation_index.csv"),
        inputs,
        config=config,
    )
    representations = representation_score_frame(store, inputs).set_index(
        ["reference_id", "sequence_sha256"]
    )
    expectations = pd.read_csv(
        ROOT / "example" / "inputs" / "candidate_expectations.csv",
        dtype={"sequence_sha256": str},
    ).set_index(["reference_id", "sequence_sha256"])
    rows: list[dict[str, object]] = []
    for candidate in inputs.candidates:
        reference = inputs.references[candidate.reference_id]
        identity = calculate_positional_identity(reference.sequence, candidate.sequence)
        key = (candidate.reference_id, candidate.sequence_sha256)
        intended = expectations.loc[key]
        analogue = str(intended["intended_structural_class"]) == "analogue"
        tm_score = 0.72 + 0.01 * (candidate.manifest_row % 5) if analogue else 0.18 + 0.01 * (
            candidate.manifest_row % 5
        )
        representation = representations.loc[key]
        rows.append(
            {
                "dataset_id": config.dataset_id,
                "manifest_row": candidate.manifest_row,
                "sequence_id": candidate.sequence_id,
                "reference_id": candidate.reference_id,
                "generation_method": candidate.generation_method,
                "sequence": candidate.sequence,
                "sequence_sha256": candidate.sequence_sha256,
                "sequence_length": candidate.sequence_length,
                "positional_matches": identity.matches,
                "positional_mismatches": identity.mismatches,
                "positional_identity_fraction": identity.fraction,
                "positional_identity_percent": identity.percent,
                "usalign_tm_score_wt_norm": tm_score,
                "sae_cosine_to_wt": float(representation["sae_cosine_to_wt"]),
                "esmc_cosine_to_wt": float(representation["esmc_cosine_to_wt"]),
                "pfam_full_sequence_bit_score": 5.0 + 50.0 * tm_score + 10.0 * identity.fraction,
                "mmseqs2_bit_score": 2.0 + 25.0 * identity.fraction + 3.0 * tm_score,
                "is_remote": identity.fraction < config.identity_threshold,
                "is_structural_analogue": analogue,
                "structural_class": "analogue" if analogue else "negative",
            }
        )
    master = pd.DataFrame(rows).sort_values("manifest_row").reset_index(drop=True)
    return config, inputs, store, master
