"""Validated sparse storage for residue activations and max-pooled profiles.

The canonical chunk schema is intentionally simple NumPy rather than a custom
binary format.  One ``.npz`` contains these arrays::

    manifest_rows          int32   [N]
    sequence_sha256        S64     [N]
    sequence_lengths       uint16  [N]
    residue_offsets        uint64  [N + 1]
    raw_mean               float32 [N, 2560]
    sae_residue_indices    uint16  [sum(sequence_lengths), 64]
    sae_residue_values     float32 [sum(sequence_lengths), 64]
    sae_profile_offsets    uint64  [N + 1]
    sae_profile_indices    uint16  [profile_nnz]
    sae_profile_values     float32 [profile_nnz]
    sae_profile_norms      float32 [N]

Residue rows and max-pooled profile entries are sorted by feature ID.  Sparse
values are strictly positive; absence means zero.  Offsets make the schema
valid for mixed sequence lengths without storing sequences themselves.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


CHUNK_SCHEMA = "landscape_esmc_sae_chunk_v1"
REQUIRED_ARRAYS = (
    "manifest_rows",
    "sequence_sha256",
    "sequence_lengths",
    "residue_offsets",
    "raw_mean",
    "sae_residue_indices",
    "sae_residue_values",
    "sae_profile_offsets",
    "sae_profile_indices",
    "sae_profile_values",
    "sae_profile_norms",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _strict_offsets(offsets: np.ndarray, *, size: int, final: int, label: str) -> None:
    if offsets.dtype != np.uint64 or offsets.shape != (size + 1,):
        raise ValueError(f"{label} must be uint64 with shape ({size + 1},)")
    if int(offsets[0]) != 0 or int(offsets[-1]) != final:
        raise ValueError(f"{label} must start at zero and end at {final}")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{label} must be nondecreasing")


@dataclass(frozen=True)
class SparseVector:
    """One sorted sparse feature vector; absent coordinates are exactly zero."""

    indices: np.ndarray
    values: np.ndarray
    size: int = 16_384

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        values = np.asarray(self.values)
        if indices.ndim != 1 or values.ndim != 1 or indices.shape != values.shape:
            raise ValueError("sparse indices and values must be matching one-dimensional arrays")
        if indices.dtype.kind not in "iu":
            raise ValueError("sparse feature indices must be integers")
        if values.dtype.kind != "f":
            raise ValueError("sparse feature values must be floating point")
        if self.size <= 0:
            raise ValueError("sparse vector size must be positive")
        indices64 = indices.astype(np.int64, copy=False)
        if len(indices64) and (
            int(indices64[0]) < 0
            or int(indices64[-1]) >= self.size
            or np.any(indices64[1:] <= indices64[:-1])
        ):
            raise ValueError("sparse indices must be unique, increasing, and in range")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("stored sparse values must be finite and strictly positive")

    @property
    def norm(self) -> float:
        """Float64 L2 calculation rounded to the stored float32 norm contract.

        Norms are calculated from float32 profile values in float64 and stored
        in ``sae_profile_norms`` as float32. The final cast is required for
        reproducible score calculations.
        """

        calculated = np.linalg.norm(np.asarray(self.values, dtype=np.float64))
        return float(np.float32(calculated))

    @property
    def unrounded_norm(self) -> float:
        """Diagnostic float64 norm before the canonical storage rounding."""

        return float(np.linalg.norm(np.asarray(self.values, dtype=np.float64)))

    def values_at(self, feature_ids: Any) -> np.ndarray:
        """Return values for arbitrary feature IDs, filling absent IDs with zero."""

        requested = np.asarray(feature_ids)
        if requested.dtype.kind not in "iu":
            raise ValueError("requested feature IDs must be integers")
        flat = requested.astype(np.int64, copy=False).ravel()
        if np.any(flat < 0) or np.any(flat >= self.size):
            raise ValueError("requested feature ID is outside the codebook")
        result = np.zeros(flat.shape, dtype=np.float32)
        indices = np.asarray(self.indices, dtype=np.int64)
        if len(indices) and len(flat):
            locations = np.searchsorted(indices, flat)
            valid = locations < len(indices)
            matched = np.zeros_like(valid)
            matched[valid] = indices[locations[valid]] == flat[valid]
            result[matched] = np.asarray(self.values, dtype=np.float32)[locations[matched]]
        return result.reshape(requested.shape)

    def to_dense(self, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
        dense = np.zeros(self.size, dtype=dtype)
        dense[np.asarray(self.indices, dtype=np.int64)] = np.asarray(self.values, dtype=dtype)
        return dense


@dataclass(frozen=True)
class SparseResidueMatrix:
    """Fixed-TopK sparse activations for every residue of one protein."""

    indices: np.ndarray
    values: np.ndarray
    size: int = 16_384

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        values = np.asarray(self.values)
        if indices.ndim != 2 or values.shape != indices.shape:
            raise ValueError("residue indices and values must be matching 2D arrays")
        if indices.dtype.kind not in "iu" or values.dtype.kind != "f":
            raise ValueError("residue indices must be integer and values floating point")
        if indices.shape[0] == 0 or indices.shape[1] == 0:
            raise ValueError("a residue activation matrix cannot be empty")
        indices64 = indices.astype(np.int64, copy=False)
        if np.any(indices64 < 0) or np.any(indices64 >= self.size):
            raise ValueError("residue feature index is outside the codebook")
        if np.any(indices64[:, 1:] <= indices64[:, :-1]):
            raise ValueError("feature IDs must be unique and increasing within each residue")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("stored residue activations must be finite and strictly positive")

    @property
    def sequence_length(self) -> int:
        return int(np.asarray(self.indices).shape[0])

    @property
    def topk(self) -> int:
        return int(np.asarray(self.indices).shape[1])

    def feature_matrix(self, feature_ids: Any) -> np.ndarray:
        """Materialize only selected feature columns as float32 ``(L, F)``."""

        requested = np.asarray(feature_ids)
        if requested.ndim != 1 or requested.dtype.kind not in "iu":
            raise ValueError("feature_ids must be a one-dimensional integer array")
        ids = requested.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= self.size):
            raise ValueError("requested feature ID is outside the codebook")
        result = np.zeros((self.sequence_length, len(ids)), dtype=np.float32)
        rank = {int(feature_id): column for column, feature_id in enumerate(ids)}
        stored_ids = np.asarray(self.indices, dtype=np.int64)
        stored_values = np.asarray(self.values, dtype=np.float32)
        for residue in range(self.sequence_length):
            for source_column, feature_id in enumerate(stored_ids[residue]):
                target_column = rank.get(int(feature_id))
                if target_column is not None:
                    result[residue, target_column] = stored_values[residue, source_column]
        return result

    def max_pool(self) -> SparseVector:
        """Max pool each feature over residues without densifying the codebook."""

        maxima: dict[int, float] = {}
        for feature_id, value in zip(
            np.asarray(self.indices, dtype=np.int64).ravel(),
            np.asarray(self.values, dtype=np.float32).ravel(),
            strict=True,
        ):
            key = int(feature_id)
            maxima[key] = max(maxima.get(key, 0.0), float(value))
        ids = np.asarray(sorted(maxima), dtype=np.uint16)
        values = np.asarray([maxima[int(key)] for key in ids], dtype=np.float32)
        return SparseVector(ids, values, self.size)


def sparse_dot(left: SparseVector, right: SparseVector) -> float:
    """Intersection dot product accumulated in float64."""

    if left.size != right.size:
        raise ValueError("sparse vector sizes differ")
    left_ids = np.asarray(left.indices, dtype=np.int64)
    right_ids = np.asarray(right.indices, dtype=np.int64)
    left_values = np.asarray(left.values, dtype=np.float64)
    right_values = np.asarray(right.values, dtype=np.float64)
    i = j = 0
    total = 0.0
    while i < len(left_ids) and j < len(right_ids):
        if left_ids[i] == right_ids[j]:
            total += float(left_values[i] * right_values[j])
            i += 1
            j += 1
        elif left_ids[i] < right_ids[j]:
            i += 1
        else:
            j += 1
    return float(total)


def sparse_cosine(left: SparseVector, right: SparseVector) -> float:
    if left.size != right.size:
        raise ValueError("sparse vector sizes differ")
    denominator = left.norm * right.norm
    return sparse_dot(left, right) / denominator if denominator else 0.0


@dataclass(frozen=True)
class RepresentationChunk:
    """In-memory view of one canonical representation chunk."""

    manifest_rows: np.ndarray
    sequence_sha256: np.ndarray
    sequence_lengths: np.ndarray
    residue_offsets: np.ndarray
    raw_mean: np.ndarray
    sae_residue_indices: np.ndarray
    sae_residue_values: np.ndarray
    sae_profile_offsets: np.ndarray
    sae_profile_indices: np.ndarray
    sae_profile_values: np.ndarray
    sae_profile_norms: np.ndarray

    @property
    def n_sequences(self) -> int:
        return int(np.asarray(self.manifest_rows).shape[0])

    @property
    def embedding_dimension(self) -> int:
        raw = np.asarray(self.raw_mean)
        return int(raw.shape[1]) if raw.ndim == 2 else -1

    @property
    def topk(self) -> int:
        indices = np.asarray(self.sae_residue_indices)
        return int(indices.shape[1]) if indices.ndim == 2 else -1

    def hashes(self) -> tuple[str, ...]:
        values = np.asarray(self.sequence_sha256)
        return tuple(
            value.decode("ascii") if isinstance(value, bytes) else str(value)
            for value in values.tolist()
        )

    def residue_matrix(self, row: int, *, codebook_size: int = 16_384) -> SparseResidueMatrix:
        if row < 0 or row >= self.n_sequences:
            raise IndexError(row)
        start = int(self.residue_offsets[row])
        stop = int(self.residue_offsets[row + 1])
        return SparseResidueMatrix(
            np.asarray(self.sae_residue_indices[start:stop]),
            np.asarray(self.sae_residue_values[start:stop]),
            codebook_size,
        )

    def profile(self, row: int, *, codebook_size: int = 16_384) -> SparseVector:
        if row < 0 or row >= self.n_sequences:
            raise IndexError(row)
        start = int(self.sae_profile_offsets[row])
        stop = int(self.sae_profile_offsets[row + 1])
        return SparseVector(
            np.asarray(self.sae_profile_indices[start:stop]),
            np.asarray(self.sae_profile_values[start:stop]),
            codebook_size,
        )

    def iter_profiles(self, *, codebook_size: int = 16_384) -> Iterator[SparseVector]:
        for row in range(self.n_sequences):
            yield self.profile(row, codebook_size=codebook_size)

    def validate(
        self,
        *,
        expected_embedding_dimension: int | None = None,
        expected_topk: int | None = None,
        codebook_size: int = 16_384,
        verify_max_pool: bool = True,
    ) -> None:
        """Validate dtypes, offsets, hashes, norms, and residue/profile agreement."""

        n = self.n_sequences
        if self.manifest_rows.dtype != np.int32 or self.manifest_rows.shape != (n,):
            raise ValueError("manifest_rows must be int32[N]")
        if len(np.unique(self.manifest_rows)) != n:
            raise ValueError("manifest_rows must be unique within a chunk")
        if self.sequence_sha256.dtype != np.dtype("S64") or self.sequence_sha256.shape != (n,):
            raise ValueError("sequence_sha256 must be S64[N]")
        hashes = self.hashes()
        # A sequence digest is not the complete candidate key: the same
        # sequence may be evaluated against multiple references. Reference
        # identity lives in the mandatory companion index, where the combined
        # identity is validated.
        if any(_SHA256_RE.fullmatch(value) is None for value in hashes):
            raise ValueError("sequence_sha256 values must be lowercase hex digests")
        if self.sequence_lengths.dtype != np.uint16 or self.sequence_lengths.shape != (n,):
            raise ValueError("sequence_lengths must be uint16[N]")
        if np.any(self.sequence_lengths == 0):
            raise ValueError("sequence lengths must be positive")
        residues = int(np.asarray(self.sequence_lengths, dtype=np.uint64).sum())
        _strict_offsets(self.residue_offsets, size=n, final=residues, label="residue_offsets")
        if not np.array_equal(
            np.diff(self.residue_offsets), np.asarray(self.sequence_lengths, dtype=np.uint64)
        ):
            raise ValueError("residue offsets do not agree with sequence_lengths")

        if (
            self.raw_mean.dtype != np.float32
            or self.raw_mean.ndim != 2
            or self.raw_mean.shape[0] != n
        ):
            raise ValueError("raw_mean must be float32[N, embedding_dimension]")
        if (
            expected_embedding_dimension is not None
            and self.raw_mean.shape[1] != expected_embedding_dimension
        ):
            raise ValueError("raw_mean embedding dimension does not match configuration")
        if not np.isfinite(self.raw_mean).all():
            raise ValueError("raw_mean contains NaN or infinity")

        if self.sae_residue_indices.dtype != np.uint16 or self.sae_residue_indices.ndim != 2:
            raise ValueError("sae_residue_indices must be uint16[sumL, TopK]")
        if self.sae_residue_indices.shape[0] != residues:
            raise ValueError("sae_residue_indices does not contain one row per residue")
        if expected_topk is not None and self.sae_residue_indices.shape[1] != expected_topk:
            raise ValueError("residue TopK does not match configuration")
        if (
            self.sae_residue_values.dtype != np.float32
            or self.sae_residue_values.shape != self.sae_residue_indices.shape
        ):
            raise ValueError("sae_residue_values must be float32 and match residue indices")
        residue_ids = self.sae_residue_indices.astype(np.int64, copy=False)
        if np.any(residue_ids >= codebook_size) or np.any(
            residue_ids[:, 1:] <= residue_ids[:, :-1]
        ):
            raise ValueError("residue feature IDs must be in range, unique, and increasing")
        if not np.isfinite(self.sae_residue_values).all() or np.any(self.sae_residue_values <= 0):
            raise ValueError("residue feature values must be finite and positive")

        profile_nnz = int(np.asarray(self.sae_profile_indices).shape[0])
        _strict_offsets(
            self.sae_profile_offsets, size=n, final=profile_nnz, label="sae_profile_offsets"
        )
        if self.sae_profile_indices.dtype != np.uint16 or self.sae_profile_indices.ndim != 1:
            raise ValueError("sae_profile_indices must be uint16[profile_nnz]")
        if (
            self.sae_profile_values.dtype != np.float32
            or self.sae_profile_values.shape != self.sae_profile_indices.shape
        ):
            raise ValueError("sae_profile_values must be float32 and match profile indices")
        if self.sae_profile_norms.dtype != np.float32 or self.sae_profile_norms.shape != (n,):
            raise ValueError("sae_profile_norms must be float32[N]")
        if not np.isfinite(self.sae_profile_values).all() or np.any(self.sae_profile_values <= 0):
            raise ValueError("profile values must be finite and positive")
        if not np.isfinite(self.sae_profile_norms).all() or np.any(self.sae_profile_norms <= 0):
            raise ValueError("profile norms must be finite and positive")

        if verify_max_pool:
            for row in range(n):
                profile = self.profile(row, codebook_size=codebook_size)
                observed_norm = profile.unrounded_norm
                if not np.isclose(
                    observed_norm,
                    float(self.sae_profile_norms[row]),
                    rtol=2e-6,
                    atol=1e-8,
                ):
                    raise ValueError(f"stored SAE profile norm is inconsistent at row {row}")
                recomputed = self.residue_matrix(row, codebook_size=codebook_size).max_pool()
                if not np.array_equal(recomputed.indices, profile.indices) or not np.array_equal(
                    recomputed.values, profile.values
                ):
                    raise ValueError(f"stored SAE profile is not the residue max pool at row {row}")

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in REQUIRED_ARRAYS}


def representation_chunk_from_arrays(arrays: Mapping[str, Any]) -> RepresentationChunk:
    missing = set(REQUIRED_ARRAYS) - set(arrays)
    extra = set(arrays) - set(REQUIRED_ARRAYS)
    if missing or extra:
        raise ValueError(
            "chunk arrays differ from schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return RepresentationChunk(**{name: np.asarray(arrays[name]) for name in REQUIRED_ARRAYS})


def load_representation_chunk(
    path: Path,
    *,
    expected_embedding_dimension: int | None = None,
    expected_topk: int | None = None,
    codebook_size: int = 16_384,
    verify_max_pool: bool = True,
) -> RepresentationChunk:
    """Load a chunk with ``allow_pickle=False`` and validate the complete schema."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(REQUIRED_ARRAYS):
            raise ValueError("NPZ fields do not match the canonical representation chunk schema")
        arrays = {name: np.array(archive[name], copy=True) for name in REQUIRED_ARRAYS}
    chunk = representation_chunk_from_arrays(arrays)
    chunk.validate(
        expected_embedding_dimension=expected_embedding_dimension,
        expected_topk=expected_topk,
        codebook_size=codebook_size,
        verify_max_pool=verify_max_pool,
    )
    return chunk


def save_representation_chunk(
    path: Path,
    chunk: RepresentationChunk,
    *,
    expected_embedding_dimension: int | None = None,
    expected_topk: int | None = None,
    codebook_size: int = 16_384,
    compressed: bool = False,
) -> None:
    """Validate and atomically save a chunk without embedding Python objects."""

    chunk.validate(
        expected_embedding_dimension=expected_embedding_dimension,
        expected_topk=expected_topk,
        codebook_size=codebook_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    writer = np.savez_compressed if compressed else np.savez
    writer(temporary, **chunk.arrays())
    temporary.replace(path)


def build_representation_chunk(
    *,
    manifest_rows: Sequence[int],
    sequence_hashes: Sequence[str],
    raw_mean: Any,
    residue_matrices: Sequence[SparseResidueMatrix],
) -> RepresentationChunk:
    """Build the canonical arrays from validated per-sequence residue matrices."""

    if len(manifest_rows) != len(sequence_hashes) or len(sequence_hashes) != len(residue_matrices):
        raise ValueError("manifest rows, hashes, and residue matrices must have equal length")
    if not residue_matrices:
        raise ValueError("cannot build an empty representation chunk")
    topk = residue_matrices[0].topk
    codebook_size = residue_matrices[0].size
    if any(matrix.topk != topk or matrix.size != codebook_size for matrix in residue_matrices):
        raise ValueError("all residue matrices must share TopK and codebook size")
    sequence_lengths = np.asarray(
        [matrix.sequence_length for matrix in residue_matrices], dtype=np.uint16
    )
    residue_offsets = np.zeros(len(residue_matrices) + 1, dtype=np.uint64)
    residue_offsets[1:] = np.cumsum(sequence_lengths, dtype=np.uint64)
    profiles = [matrix.max_pool() for matrix in residue_matrices]
    profile_offsets = np.zeros(len(profiles) + 1, dtype=np.uint64)
    profile_offsets[1:] = np.cumsum([len(profile.indices) for profile in profiles], dtype=np.uint64)
    chunk = RepresentationChunk(
        manifest_rows=np.asarray(manifest_rows, dtype=np.int32),
        sequence_sha256=np.asarray(
            [value.encode("ascii") for value in sequence_hashes], dtype="S64"
        ),
        sequence_lengths=sequence_lengths,
        residue_offsets=residue_offsets,
        raw_mean=np.asarray(raw_mean, dtype=np.float32),
        sae_residue_indices=np.concatenate([matrix.indices for matrix in residue_matrices]).astype(
            np.uint16, copy=False
        ),
        sae_residue_values=np.concatenate([matrix.values for matrix in residue_matrices]).astype(
            np.float32, copy=False
        ),
        sae_profile_offsets=profile_offsets,
        sae_profile_indices=np.concatenate([profile.indices for profile in profiles]).astype(
            np.uint16, copy=False
        ),
        sae_profile_values=np.concatenate([profile.values for profile in profiles]).astype(
            np.float32, copy=False
        ),
        sae_profile_norms=np.asarray([profile.norm for profile in profiles], dtype=np.float32),
    )
    chunk.validate(expected_topk=topk, codebook_size=codebook_size)
    return chunk
