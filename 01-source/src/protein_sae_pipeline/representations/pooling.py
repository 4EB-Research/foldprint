"""Pooling and cosine kernels for the raw ESM-C score.

The representation is ``hidden_states[60]``: the raw residual stream
after transformer block 59.  Encoded proteins contain one BOS token, the
residues, and one EOS token.  Only residue rows are pooled, and their mean is
formed in float32.  Cosines are deliberately evaluated in float64.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _numpy_float32(value: Any) -> np.ndarray:
    """Convert NumPy/torch-like values without sending bfloat16 to NumPy."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def validate_bos_eos_layout(
    input_ids: Any,
    attention_mask: Any,
    sequence_lengths: Sequence[int],
    *,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int | None = None,
) -> None:
    """Validate the exact ``BOS + residues + EOS + right padding`` layout.

    This check belongs next to inference because hidden states alone cannot
    prove which rows are special tokens.  ``sequence_lengths`` are amino-acid
    lengths, not token counts.
    """

    tokens = np.asarray(input_ids)
    mask = np.asarray(attention_mask)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    if mask.ndim == 1:
        mask = mask[None, :]
    lengths = np.asarray(sequence_lengths, dtype=np.int64)
    if tokens.ndim != 2 or mask.shape != tokens.shape:
        raise ValueError("input_ids and attention_mask must be matching 1D or 2D arrays")
    if lengths.shape != (tokens.shape[0],):
        raise ValueError("sequence_lengths does not match the token batch")
    if np.any(lengths <= 0):
        raise ValueError("sequence lengths must be positive")

    active_mask = mask.astype(bool)
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        expected_tokens = length + 2
        if expected_tokens > tokens.shape[1]:
            raise ValueError(f"row {row} is shorter than BOS + {length} residues + EOS")
        if int(active_mask[row].sum()) != expected_tokens:
            raise ValueError(f"row {row} does not contain exactly sequence_length + 2 tokens")
        if int(tokens[row, 0]) != int(bos_token_id):
            raise ValueError(f"row {row} does not begin with BOS")
        if int(tokens[row, length + 1]) != int(eos_token_id):
            raise ValueError(f"row {row} does not place EOS at sequence_length + 1")
        if not bool(active_mask[row, :expected_tokens].all()):
            raise ValueError(f"row {row} contains padding before EOS")
        if bool(active_mask[row, expected_tokens:].any()):
            raise ValueError(f"row {row} contains active tokens after EOS")
        if pad_token_id is not None and tokens.shape[1] > expected_tokens:
            if not bool((tokens[row, expected_tokens:] == int(pad_token_id)).all()):
                raise ValueError(f"row {row} contains non-padding token IDs after EOS")


def residue_hidden_states(
    hidden_states: Any,
    *,
    sequence_length: int,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Return float32 residue rows after removing exactly one BOS and one EOS.

    ``hidden_states`` may be ``(L + 2, D)`` or the singleton-batch equivalent
    ``(1, L + 2, D)``.  Inputs that have already lost special tokens are
    rejected so the convention cannot silently drift between callers.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    states = _numpy_float32(hidden_states)
    if states.ndim == 3 and states.shape[0] == 1:
        states = states[0]
    if states.ndim != 2:
        raise ValueError(f"expected a 2D token-by-channel tensor, received {states.shape}")
    expected_tokens = sequence_length + 2
    if states.shape[0] != expected_tokens:
        raise ValueError(
            "layer-60 token count must be BOS + residues + EOS: "
            f"observed={states.shape[0]}, expected={expected_tokens}"
        )
    if expected_dimension is not None and states.shape[1] != expected_dimension:
        raise ValueError(
            f"layer-60 width is {states.shape[1]}, expected {expected_dimension}"
        )
    if not np.isfinite(states).all():
        raise ValueError("layer-60 hidden states contain NaN or infinity")
    return np.ascontiguousarray(states[1 : sequence_length + 1], dtype=np.float32)


def mean_pool_layer60(
    hidden_states: Any,
    *,
    sequence_length: int,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Mean-pool ESM-C layer 60 over residues only, accumulating in float32."""

    residues = residue_hidden_states(
        hidden_states,
        sequence_length=sequence_length,
        expected_dimension=expected_dimension,
    )
    # The dtype argument is intentional: it mirrors torch's residues.float().mean().
    pooled = np.mean(residues, axis=0, dtype=np.float32)
    return np.asarray(pooled, dtype=np.float32)


def mean_pool_layer60_batch(
    hidden_states: Any,
    sequence_lengths: Sequence[int],
    *,
    attention_mask: Any | None = None,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Pool a right-padded batch using the same residue-only convention.

    An attention mask is mandatory when token rows extend beyond ``L + 2``.
    The special-token identities should separately be checked with
    :func:`validate_bos_eos_layout` at the tokenizer boundary.
    """

    states = _numpy_float32(hidden_states)
    lengths = np.asarray(sequence_lengths, dtype=np.int64)
    if states.ndim != 3 or lengths.shape != (states.shape[0],):
        raise ValueError("expected hidden_states=(batch,tokens,channels) and one length per row")
    if np.any(lengths <= 0):
        raise ValueError("sequence lengths must be positive")
    if expected_dimension is not None and states.shape[2] != expected_dimension:
        raise ValueError(f"hidden-state width is {states.shape[2]}, expected {expected_dimension}")
    mask = None if attention_mask is None else np.asarray(attention_mask).astype(bool)
    if mask is not None and mask.shape != states.shape[:2]:
        raise ValueError("attention_mask does not match hidden-state token dimensions")

    pooled: list[np.ndarray] = []
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        expected_tokens = length + 2
        if expected_tokens > states.shape[1]:
            raise ValueError(f"row {row} is shorter than sequence_length + 2")
        if states.shape[1] != expected_tokens and mask is None:
            raise ValueError("attention_mask is required for a padded layer-60 batch")
        if mask is not None:
            if int(mask[row].sum()) != expected_tokens:
                raise ValueError(f"row {row} attention count is not sequence_length + 2")
            if not bool(mask[row, :expected_tokens].all()) or bool(
                mask[row, expected_tokens:].any()
            ):
                raise ValueError(f"row {row} is not right padded after EOS")
        residues = states[row, 1 : length + 1]
        if not np.isfinite(residues).all():
            raise ValueError(f"row {row} contains non-finite residue states")
        pooled.append(np.mean(residues, axis=0, dtype=np.float32))
    return np.asarray(pooled, dtype=np.float32)


def cosine_similarity(left: Any, right: Any) -> float:
    """Return cosine similarity with dot products and norms in float64.

    The original scoring code defined a zero-norm comparison as ``0.0``; this
    behavior is retained even though valid embeddings are nonzero.
    """

    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.ndim != 1 or right64.ndim != 1 or left64.shape != right64.shape:
        raise ValueError("cosine operands must be one-dimensional arrays of equal shape")
    if not np.isfinite(left64).all() or not np.isfinite(right64).all():
        raise ValueError("cosine operands contain NaN or infinity")
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    return float(np.dot(left64, right64) / denominator) if denominator else 0.0


def cosine_to_wt(candidate_embeddings: Any, wt_embedding: Any) -> np.ndarray:
    """Score one or more pooled candidate embeddings against a WT embedding."""

    candidates = np.asarray(candidate_embeddings)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    if candidates.ndim != 2:
        raise ValueError("candidate_embeddings must be a vector or a matrix")
    wt = np.asarray(wt_embedding)
    if wt.ndim != 1 or candidates.shape[1] != wt.shape[0]:
        raise ValueError("candidate and WT embedding dimensions do not match")
    return np.asarray([cosine_similarity(row, wt) for row in candidates], dtype=np.float64)
