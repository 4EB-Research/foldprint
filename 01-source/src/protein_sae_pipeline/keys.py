"""Canonical sequence normalization and composite keys."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def normalize_sequence(value: str, *, label: str = "sequence") -> str:
    """Return an uppercase, whitespace-free canonical protein sequence."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise ValueError(f"{label} is empty")
    invalid = sorted(set(sequence) - CANONICAL_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"{label} contains noncanonical residues: {invalid}")
    return sequence


def sequence_sha256(sequence: str) -> str:
    normalized = normalize_sequence(sequence)
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


@dataclass(frozen=True, order=True)
class CandidateKey:
    reference_id: str
    sequence_sha256: str

    def __post_init__(self) -> None:
        if not self.reference_id or not self.reference_id.strip():
            raise ValueError("reference_id must be nonempty")
        if SHA256_PATTERN.fullmatch(self.sequence_sha256) is None:
            raise ValueError("sequence_sha256 must be 64 lowercase hexadecimal characters")

    def as_tuple(self) -> tuple[str, str]:
        return self.reference_id, self.sequence_sha256

    def display(self) -> str:
        return f"{self.reference_id}:{self.sequence_sha256}"
