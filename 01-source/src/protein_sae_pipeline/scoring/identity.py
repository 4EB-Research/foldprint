"""Equal-length positional sequence identity.

This is deliberately not an alignment routine.  Candidate and WT sequences
must have the same length, and residues are compared at the same sequence
position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..keys import normalize_sequence as _normalize_sequence


@dataclass(frozen=True)
class PositionalIdentity:
    """A complete positional-identity result for one candidate/WT pair."""

    aligned_length: int
    matches: int
    mismatches: int
    fraction: float
    percent: float


def normalize_sequence(value: Any) -> str:
    """Return the canonical sequence spelling used before identity scoring.

    Whitespace is removed and letters are upper-cased. Stop markers, gaps,
    ambiguous residues, and empty values are rejected exactly as they are by
    the input-manifest validator.
    """

    if not isinstance(value, str):
        raise TypeError("sequence must be a string")
    return _normalize_sequence(value)


def calculate_positional_identity(reference: Any, candidate: Any) -> PositionalIdentity:
    """Compare an equal-length candidate to its WT reference position by position."""

    normalized_reference = normalize_sequence(reference)
    normalized_candidate = normalize_sequence(candidate)
    if len(normalized_reference) != len(normalized_candidate):
        raise ValueError(
            "Positional identity requires equal lengths; got "
            f"{len(normalized_reference)} and {len(normalized_candidate)}."
        )

    length = len(normalized_reference)
    matches = sum(
        reference_residue == candidate_residue
        for reference_residue, candidate_residue in zip(
            normalized_reference, normalized_candidate, strict=True
        )
    )
    fraction = matches / length
    return PositionalIdentity(
        aligned_length=length,
        matches=matches,
        mismatches=length - matches,
        fraction=fraction,
        percent=100.0 * fraction,
    )


def positional_sequence_identity(reference: Any, candidate: Any) -> float:
    """Return the equal-length positional identity as a fraction in ``[0, 1]``."""

    return calculate_positional_identity(reference, candidate).fraction
