"""Typed input records used by validation and scoring stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .keys import CandidateKey


@dataclass(frozen=True)
class Candidate:
    manifest_row: int
    sequence_id: str
    reference_id: str
    generation_method: str
    sequence: str
    sequence_sha256: str
    sequence_length: int

    @property
    def key(self) -> CandidateKey:
        return CandidateKey(self.reference_id, self.sequence_sha256)


@dataclass(frozen=True)
class Reference:
    reference_id: str
    reference_sequence_id: str
    sequence: str
    sequence_sha256: str
    sequence_length: int
    structure_path: Path
    chain_id: str
    pfam_accession: str
    pfam_profile_sha256: str
    pfam_profile_path: Path
    figure_focal_position: int | None


@dataclass(frozen=True)
class ReferencePosition:
    reference_id: str
    sequence_position: int
    residue: str
    observed: bool
    chain_id: str | None
    author_residue_number: int | None
    insertion_code: str


@dataclass(frozen=True)
class CandidateStructure:
    key: CandidateKey
    structure_path: Path
    chain_id: str


@dataclass(frozen=True)
class ValidatedInputs:
    dataset_id: str
    data_root: Path
    candidates: tuple[Candidate, ...]
    references: dict[str, Reference]
    reference_positions: dict[str, tuple[ReferencePosition, ...]]
    structures: dict[CandidateKey, CandidateStructure]

    @property
    def candidate_keys(self) -> tuple[CandidateKey, ...]:
        return tuple(candidate.key for candidate in self.candidates)
