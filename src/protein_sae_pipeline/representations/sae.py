"""SAE normalization, TopK extraction, max pooling, and WT scoring.

The pipeline uses the layer-60 K=64, 16,384-feature SAE. Biohub's pinned
normalization is applied in this exact order, in float32::

    normalized = (feature_magnitude / max_per_feature) * idf_per_feature

The division and multiplication are intentionally separate operations.  Each
feature is then max pooled over residue positions before float64 cosine scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .sparse import SparseResidueMatrix, SparseVector, sparse_cosine


DEFAULT_CODEBOOK_SIZE = 16_384
DEFAULT_TOPK = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class SAENormalization:
    """Pinned per-feature maximum and IDF arrays."""

    max_per_feature: np.ndarray
    idf_per_feature: np.ndarray
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        maximum = np.asarray(self.max_per_feature)
        idf = np.asarray(self.idf_per_feature)
        if maximum.ndim != 1 or idf.shape != maximum.shape or maximum.size == 0:
            raise ValueError("max_per_feature and idf_per_feature must be matching vectors")
        if maximum.dtype != np.float32 or idf.dtype != np.float32:
            raise ValueError("SAE normalization arrays must be float32")
        if not np.isfinite(maximum).all() or np.any(maximum <= 0):
            raise ValueError("max_per_feature must be finite and strictly positive")
        if not np.isfinite(idf).all() or np.any(idf < 0):
            raise ValueError("idf_per_feature must be finite and non-negative")
        if self.source_sha256 is not None:
            if len(self.source_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.source_sha256
            ):
                raise ValueError("source_sha256 must be a lowercase SHA-256 digest")

    @property
    def codebook_size(self) -> int:
        return int(np.asarray(self.max_per_feature).size)


def _normalization_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            valid_keys = (
                {"max_per_feature", "idf_per_feature"},
                {"maximum", "idf"},
            )
            if set(archive.files) not in valid_keys:
                raise ValueError(
                    "normalization NPZ must contain max_per_feature/idf_per_feature "
                    "or maximum/idf"
                )
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("normalization JSON must be an object")
        return payload
    raise ValueError("normalization assets must be .npz or .json")


def load_sae_normalization(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_codebook_size: int | None = DEFAULT_CODEBOOK_SIZE,
    expected_maximum_sha256: str | None = None,
    expected_idf_sha256: str | None = None,
) -> SAENormalization:
    """Load and checksum a local normalization asset without network access."""

    observed_sha256 = _sha256_file(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"SAE normalization checksum mismatch: {observed_sha256} != {expected_sha256}"
        )
    payload = _normalization_payload(path)
    source_keys = set(payload)
    if source_keys == {"max_per_feature", "idf_per_feature"}:
        maximum_key, idf_key = "max_per_feature", "idf_per_feature"
    elif source_keys == {"maximum", "idf"}:
        maximum_key, idf_key = "maximum", "idf"
    else:
        raise ValueError(
            "normalization asset keys must be max_per_feature/idf_per_feature "
            "or maximum/idf"
        )
    maximum = np.asarray(payload[maximum_key], dtype=np.float32)
    idf = np.asarray(payload[idf_key], dtype=np.float32)
    asset = SAENormalization(maximum, idf, observed_sha256)
    if expected_codebook_size is not None and asset.codebook_size != expected_codebook_size:
        raise ValueError(
            f"normalization codebook has {asset.codebook_size} features, "
            f"expected {expected_codebook_size}"
        )
    array_digests = {
        "maximum": hashlib.sha256(
            maximum.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest(),
        "idf": hashlib.sha256(
            idf.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest(),
    }
    if (
        expected_maximum_sha256 is not None
        and array_digests["maximum"] != expected_maximum_sha256
    ):
        raise ValueError("SAE maximum array checksum differs from the frozen asset")
    if expected_idf_sha256 is not None and array_digests["idf"] != expected_idf_sha256:
        raise ValueError("SAE IDF array checksum differs from the frozen asset")
    return asset


def normalize_sae_activations(
    feature_magnitudes: Any,
    normalization: SAENormalization,
) -> np.ndarray:
    """Apply exact max normalization followed by IDF multiplication in float32."""

    raw = _numpy_float32(feature_magnitudes)
    if raw.ndim < 1 or raw.shape[-1] != normalization.codebook_size:
        raise ValueError(
            "SAE feature dimension does not match normalization assets: "
            f"shape={raw.shape}, codebook={normalization.codebook_size}"
        )
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("SAE feature magnitudes must be finite and non-negative")

    # Preserve operation order because float32 rounding affects reproducibility.
    normalized = np.empty(raw.shape, dtype=np.float32)
    np.divide(raw, normalization.max_per_feature, out=normalized)
    np.multiply(normalized, normalization.idf_per_feature, out=normalized)
    if not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise ValueError("normalized SAE activations are invalid")
    return normalized


def extract_topk_residue_activations(
    normalized_activations: Any,
    *,
    topk: int = DEFAULT_TOPK,
) -> SparseResidueMatrix:
    """Extract per-residue TopK and store selected coordinates by feature ID.

    The trained SAE already emits K-sparse magnitudes.  For defensive use with
    denser arrays, activation ties are resolved by smaller feature ID.  Output
    rows are then sorted by feature ID, matching the canonical NPZ schema.
    """

    matrix = _numpy_float32(normalized_activations)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise ValueError("normalized residue activations must have shape (residues, features)")
    if topk <= 0 or topk > matrix.shape[1]:
        raise ValueError("topk must be between one and the codebook size")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("normalized residue activations are empty or invalid")

    indices = np.empty((matrix.shape[0], topk), dtype=np.uint16)
    values = np.empty((matrix.shape[0], topk), dtype=np.float32)
    feature_ids = np.arange(matrix.shape[1], dtype=np.int64)
    for residue, row in enumerate(matrix):
        # Descending activation, then ascending feature ID, gives deterministic ties.
        selected = np.lexsort((feature_ids, -row))[:topk]
        selected.sort()
        selected_values = row[selected]
        if np.any(selected_values <= 0):
            raise ValueError(
                f"residue {residue} has fewer than {topk} strictly positive normalized features"
            )
        indices[residue] = selected.astype(np.uint16)
        values[residue] = selected_values.astype(np.float32, copy=False)
    return SparseResidueMatrix(indices, values, matrix.shape[1])


def max_pool_dense(normalized_activations: Any) -> np.ndarray:
    """Max pool a dense ``(residues, features)`` activation matrix."""

    matrix = _numpy_float32(normalized_activations)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("normalized activations must be a non-empty residue-by-feature matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("normalized activations must be finite and non-negative")
    return np.max(matrix, axis=0).astype(np.float32, copy=False)


def max_pool_sparse(residue_activations: SparseResidueMatrix) -> SparseVector:
    """Max pool canonical sparse residue activations over positions."""

    return residue_activations.max_pool()


def dense_profile_to_sparse(profile: Any) -> SparseVector:
    values = _numpy_float32(profile)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("a pooled SAE profile must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("pooled SAE profile must be finite and non-negative")
    active = np.flatnonzero(values > 0).astype(np.uint16)
    return SparseVector(active, values[active.astype(np.int64)], values.size)


def sae_cosine_to_wt(candidate: SparseVector, wt: SparseVector) -> float:
    """Cosine between normalized max-pooled candidate and WT SAE profiles."""

    return sparse_cosine(candidate, wt)
