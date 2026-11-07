"""Join the canonical representation chunk to candidate and WT identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..keys import CandidateKey
from ..config import PipelineConfig
from ..io import sha256_file
from ..schemas import ValidatedInputs
from .pooling import cosine_similarity
from .sparse import RepresentationChunk, SparseResidueMatrix, SparseVector, load_representation_chunk, sparse_cosine


INDEX_COLUMNS = ["chunk_row", "manifest_row", "reference_id", "role", "sequence_sha256"]
METADATA_FIELDS = {
    "schema_version",
    "storage_schema",
    "dataset_id",
    "origin",
    "fictional",
    "rows_including_wt",
    "candidate_rows",
    "embedding_dimension",
    "codebook_size",
    "top_k",
    "chunk_sha256",
    "index_sha256",
    "esmc_contract_sha256",
    "sae_contract_sha256",
    "notice",
}


def representation_contract_sha256(payload: object) -> str:
    """Return the canonical digest used to bind stored tensors to a config."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_representation_metadata(
    chunk_path: Path,
    index_path: Path,
    inputs: ValidatedInputs,
    config: PipelineConfig,
) -> dict[str, object]:
    metadata_path = chunk_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Representation provenance metadata is missing: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != METADATA_FIELDS:
        observed = set(payload) if isinstance(payload, dict) else set()
        raise ValueError(
            "Representation metadata fields differ: "
            f"missing={sorted(METADATA_FIELDS - observed)}, "
            f"extra={sorted(observed - METADATA_FIELDS)}"
        )
    esmc = config.model("esmc")
    sae = config.model("sae")
    expected_origin = "synthetic_offline_fixture" if config.is_example else "live_biohub"
    expected = {
        "schema_version": 1,
        "storage_schema": "protein_sae_sparse_representations_v1",
        "dataset_id": config.dataset_id,
        "origin": expected_origin,
        "fictional": config.is_example,
        "rows_including_wt": len(inputs.candidates) + len(inputs.references),
        "candidate_rows": len(inputs.candidates),
        "embedding_dimension": int(esmc["embedding_dimension"]),
        "codebook_size": int(sae["codebook_size"]),
        "top_k": int(sae["top_k"]),
        "chunk_sha256": sha256_file(chunk_path),
        "index_sha256": sha256_file(index_path),
        "esmc_contract_sha256": representation_contract_sha256(esmc),
        "sae_contract_sha256": representation_contract_sha256(sae),
    }
    differences = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if differences:
        raise ValueError(
            "Representation provenance differs from the validated dataset/configuration: "
            + ", ".join(sorted(differences))
        )
    notice = payload.get("notice")
    if not isinstance(notice, str) or not notice.strip():
        raise ValueError("Representation metadata notice must be a nonempty string")
    return payload


@dataclass(frozen=True)
class IndexedRepresentations:
    chunk: RepresentationChunk
    index: pd.DataFrame
    candidate_rows: dict[CandidateKey, int]
    wt_rows: dict[str, int]

    def candidate_profile(self, key: CandidateKey) -> SparseVector:
        return self.chunk.profile(self.candidate_rows[key])

    def candidate_residues(self, key: CandidateKey) -> SparseResidueMatrix:
        return self.chunk.residue_matrix(self.candidate_rows[key])

    def wt_profile(self, reference_id: str) -> SparseVector:
        return self.chunk.profile(self.wt_rows[reference_id])

    def wt_residues(self, reference_id: str) -> SparseResidueMatrix:
        return self.chunk.residue_matrix(self.wt_rows[reference_id])

    def wt_raw_mean(self, reference_id: str) -> np.ndarray:
        return self.chunk.raw_mean[self.wt_rows[reference_id]]


def load_indexed_representations(
    chunk_path: Path,
    index_path: Path,
    inputs: ValidatedInputs,
    *,
    config: PipelineConfig,
    verify_max_pool: bool = True,
) -> IndexedRepresentations:
    _validate_representation_metadata(chunk_path, index_path, inputs, config)
    esmc = config.model("esmc")
    sae = config.model("sae")
    embedding_dimension = int(esmc["embedding_dimension"])
    topk = int(sae["top_k"])
    codebook_size = int(sae["codebook_size"])
    chunk = load_representation_chunk(
        chunk_path,
        expected_embedding_dimension=embedding_dimension,
        expected_topk=topk,
        codebook_size=codebook_size,
        verify_max_pool=verify_max_pool,
    )
    index = pd.read_csv(index_path, dtype={"sequence_sha256": str})
    missing = set(INDEX_COLUMNS) - set(index.columns)
    extra = set(index.columns) - set(INDEX_COLUMNS)
    if missing or extra:
        raise ValueError(f"Representation index fields differ: missing={sorted(missing)}, extra={sorted(extra)}")
    if len(index) != chunk.n_sequences:
        raise ValueError("Representation index row count differs from chunk")
    if index["chunk_row"].tolist() != list(range(chunk.n_sequences)):
        raise ValueError("Representation index chunk_row must be contiguous and ordered")
    if index["sequence_sha256"].tolist() != list(chunk.hashes()):
        raise ValueError("Representation index hash order differs from chunk")
    if not bool(index["role"].isin(["candidate", "wt_reference"]).all()):
        raise ValueError("Representation index contains an unknown role")
    if index.duplicated(["reference_id", "role", "sequence_sha256"]).any():
        raise ValueError("Representation index contains duplicate identity rows")

    candidate_rows: dict[CandidateKey, int] = {}
    wt_rows: dict[str, int] = {}
    expected_candidates = {candidate.key: candidate for candidate in inputs.candidates}
    for row in index.itertuples(index=False):
        reference_id = str(row.reference_id)
        if reference_id not in inputs.references:
            raise ValueError(f"Representation index has unknown reference {reference_id}")
        chunk_row = int(row.chunk_row)
        if row.role == "wt_reference":
            if reference_id in wt_rows:
                raise ValueError(f"Representation index has duplicate WT for {reference_id}")
            reference = inputs.references[reference_id]
            if str(row.sequence_sha256) != reference.sequence_sha256 or int(row.manifest_row) >= 0:
                raise ValueError(f"Representation WT identity differs for {reference_id}")
            wt_rows[reference_id] = chunk_row
        else:
            key = CandidateKey(reference_id, str(row.sequence_sha256))
            candidate = expected_candidates.get(key)
            if candidate is None or int(row.manifest_row) != candidate.manifest_row:
                raise ValueError(f"Representation candidate identity differs for {key.display()}")
            candidate_rows[key] = chunk_row
    if set(candidate_rows) != set(expected_candidates):
        raise ValueError("Representation chunk does not exactly cover candidate keys")
    if set(wt_rows) != set(inputs.references):
        raise ValueError("Representation chunk does not contain exactly one WT per reference")
    return IndexedRepresentations(chunk, index, candidate_rows, wt_rows)


def representation_score_frame(
    store: IndexedRepresentations,
    inputs: ValidatedInputs,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in inputs.candidates:
        key = candidate.key
        row = store.candidate_rows[key]
        wt_row = store.wt_rows[candidate.reference_id]
        candidate_raw = store.chunk.raw_mean[row]
        wt_raw = store.chunk.raw_mean[wt_row]
        candidate_profile = store.chunk.profile(row)
        wt_profile = store.chunk.profile(wt_row)
        raw_zero = float(np.linalg.norm(candidate_raw.astype(np.float64))) == 0.0 or float(
            np.linalg.norm(wt_raw.astype(np.float64))
        ) == 0.0
        sae_zero = candidate_profile.norm == 0.0 or wt_profile.norm == 0.0
        rows.append(
            {
                "reference_id": candidate.reference_id,
                "sequence_sha256": candidate.sequence_sha256,
                "esmc_cosine_to_wt": cosine_similarity(candidate_raw, wt_raw),
                "esmc_zero_norm": raw_zero,
                "sae_cosine_to_wt": sparse_cosine(candidate_profile, wt_profile),
                "sae_profile_norm": candidate_profile.norm,
                "sae_profile_nnz": len(candidate_profile.indices),
                "sae_zero_norm": sae_zero,
            }
        )
    return pd.DataFrame(rows)
