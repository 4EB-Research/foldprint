"""Conventional US-align execution and strict labeled-output parsing.

The primary endpoint is the alignment-optimized TM-score normalized by the WT
coordinate length.  Candidate is always ``Structure_1`` and WT is always
``Structure_2``.  Both inputs are rewritten as deterministic, sequential,
C-alpha-only PDBs before running exactly::

    USalign candidate_ca.pdb wt_ca.pdb -mol prot
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Literal

from protein_sae_pipeline.structures.pdb import (
    PDBValidationError,
    ParsedStructure,
    parse_pdb,
    write_sequential_ca_pdb,
)


DEFAULT_TM_THRESHOLD = 0.5
ANALOGUE_CLASS = "analogue"
NEGATIVE_CLASS = "negative"

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_SUMMARY_RE = re.compile(
    rf"Aligned\s+length\s*=\s*(\d+)\s*,\s*RMSD\s*=\s*({_FLOAT})\s*,\s*"
    rf"Seq_ID=n_identical/n_aligned\s*=\s*({_FLOAT})",
    re.IGNORECASE,
)
_TM_SCORE_RE = re.compile(
    rf"TM-score\s*=\s*({_FLOAT})[^\n\r]*normalized\s+by\s+(?:the\s+)?"
    r"length\s+of\s+Structure[_ ]?([12])\b",
    re.IGNORECASE,
)
_LENGTH_RE = re.compile(
    r"Length\s+of\s+Structure[_ ]?([12])\s*:\s*(\d+)\s+residues?\b",
    re.IGNORECASE,
)
_VERSION_RES = (
    re.compile(r"US-align\s*\(Version\s+([^\)]+)\)", re.IGNORECASE),
    re.compile(r"US-align\s+Version\s*[:=]?\s*([^\s\)]+)", re.IGNORECASE),
)


class USAlignError(RuntimeError):
    """Raised when US-align execution or output violates the scoring contract."""


@dataclass(frozen=True)
class USAlignResult:
    """Alignment-optimized metrics with both normalization directions retained."""

    candidate_coordinate_length: int
    wt_coordinate_length: int
    tm_score_candidate_normalized: float
    tm_score_wt_normalized: float
    aligned_length: int
    rmsd_angstrom: float
    aligned_sequence_identity: float
    version: str | None = None
    command: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        candidate_length = self.candidate_coordinate_length
        wt_length = self.wt_coordinate_length
        aligned_length = self.aligned_length
        values = (
            self.tm_score_candidate_normalized,
            self.tm_score_wt_normalized,
            self.rmsd_angstrom,
            self.aligned_sequence_identity,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (
            candidate_length,
            wt_length,
            aligned_length,
        )):
            raise USAlignError("US-align lengths must be integers.")
        if candidate_length <= 0 or wt_length <= 0:
            raise USAlignError("US-align coordinate lengths must be positive.")
        if aligned_length <= 0 or aligned_length > min(candidate_length, wt_length):
            raise USAlignError(
                "US-align aligned length must be positive and no greater than either "
                "coordinate length."
            )
        if not all(math.isfinite(float(value)) for value in values):
            raise USAlignError("US-align returned a non-finite metric.")
        if not 0.0 <= self.tm_score_candidate_normalized <= 1.0:
            raise USAlignError("Candidate-normalized TM-score is outside [0, 1].")
        if not 0.0 <= self.tm_score_wt_normalized <= 1.0:
            raise USAlignError("WT-normalized TM-score is outside [0, 1].")
        if self.rmsd_angstrom < 0.0:
            raise USAlignError("US-align RMSD cannot be negative.")
        if not 0.0 <= self.aligned_sequence_identity <= 1.0:
            raise USAlignError("US-align aligned sequence identity is outside [0, 1].")
        if self.command is not None:
            object.__setattr__(self, "command", tuple(str(value) for value in self.command))

    @property
    def primary_tm_score(self) -> float:
        """Structure_2/WT-coordinate-normalized TM-score."""

        return self.tm_score_wt_normalized

    @property
    def aligned_fraction_candidate(self) -> float:
        return self.aligned_length / self.candidate_coordinate_length

    @property
    def aligned_fraction_wt(self) -> float:
        return self.aligned_length / self.wt_coordinate_length

    def structural_class(self, threshold: float = DEFAULT_TM_THRESHOLD) -> str:
        return classify_tm_score(self.primary_tm_score, threshold=threshold)


def _unique_labeled_values(
    pairs: list[tuple[str, str]],
    *,
    label: str,
    cast: type[float] | type[int],
) -> dict[int, float] | dict[int, int]:
    collected: dict[int, set[float | int]] = {1: set(), 2: set()}
    for raw_value, raw_structure in pairs:
        collected[int(raw_structure)].add(cast(raw_value))
    if any(not collected[index] for index in (1, 2)):
        raise USAlignError(f"US-align output lacks labeled {label} values for both structures.")
    if any(len(collected[index]) != 1 for index in (1, 2)):
        raise USAlignError(f"US-align output contains conflicting labeled {label} values.")
    return {index: next(iter(collected[index])) for index in (1, 2)}


def _parse_version(stdout: str) -> str | None:
    for expression in _VERSION_RES:
        match = expression.search(stdout)
        if match is not None:
            return match.group(1).strip()
    return None


def parse_usalign_output(stdout: str) -> USAlignResult:
    """Parse a complete US-align report without relying on score line order.

    Both TM-scores and both coordinate lengths must explicitly identify
    ``Structure_1`` or ``Structure_2``.  An unlabeled or one-score report is
    rejected rather than guessed.
    """

    if not isinstance(stdout, str) or not stdout.strip():
        raise USAlignError("US-align output is empty.")
    summary_matches = _SUMMARY_RE.findall(stdout)
    if len(summary_matches) != 1:
        raise USAlignError(
            "US-align output must contain exactly one aligned length/RMSD/identity summary."
        )
    raw_aligned_length, raw_rmsd, raw_identity = summary_matches[0]
    tm_scores = _unique_labeled_values(
        _TM_SCORE_RE.findall(stdout), label="TM-score", cast=float
    )
    lengths = _unique_labeled_values(
        # Reorder regex output from (structure, length) to (length, structure).
        [(length, structure) for structure, length in _LENGTH_RE.findall(stdout)],
        label="coordinate length",
        cast=int,
    )
    return USAlignResult(
        candidate_coordinate_length=int(lengths[1]),
        wt_coordinate_length=int(lengths[2]),
        tm_score_candidate_normalized=float(tm_scores[1]),
        tm_score_wt_normalized=float(tm_scores[2]),
        aligned_length=int(raw_aligned_length),
        rmsd_angstrom=float(raw_rmsd),
        aligned_sequence_identity=float(raw_identity),
        version=_parse_version(stdout),
    )


def classify_tm_score(
    tm_score_wt_normalized: float,
    *,
    threshold: float = DEFAULT_TM_THRESHOLD,
) -> Literal["analogue", "negative"]:
    """Classify a WT-normalized TM-score using the inclusive configured threshold."""

    score = float(tm_score_wt_normalized)
    cutoff = float(threshold)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("WT-normalized TM-score must be finite and within [0, 1].")
    if not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise ValueError("TM-score threshold must be finite and within [0, 1].")
    return ANALOGUE_CLASS if score >= cutoff else NEGATIVE_CLASS


def _as_structure(
    value: ParsedStructure | str | os.PathLike[str],
    *,
    chain_id: str | None,
) -> ParsedStructure:
    if isinstance(value, ParsedStructure):
        if chain_id is not None and value.chain_id != str(chain_id).strip():
            raise PDBValidationError(
                f"Parsed structure chain {value.chain_id!r} disagrees with requested "
                f"chain {chain_id!r}."
            )
        return value
    return parse_pdb(value, chain_id=chain_id)


def run_usalign(
    candidate: ParsedStructure | str | os.PathLike[str],
    wt: ParsedStructure | str | os.PathLike[str],
    *,
    executable: str | os.PathLike[str] = "USalign",
    candidate_chain_id: str | None = None,
    wt_chain_id: str | None = None,
    temp_root: str | os.PathLike[str] | None = None,
    timeout_seconds: float | None = None,
) -> USAlignResult:
    """Run alignment-optimized US-align with candidate as 1 and WT as 2."""

    candidate_structure = _as_structure(candidate, chain_id=candidate_chain_id)
    wt_structure = _as_structure(wt, chain_id=wt_chain_id)
    root: Path | None
    if temp_root is None:
        root = None
    else:
        root = Path(temp_root)
        root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="protein_sae_usalign_", dir=root) as temporary:
        temporary_directory = Path(temporary)
        candidate_ca = write_sequential_ca_pdb(
            candidate_structure, temporary_directory / "candidate_ca.pdb"
        )
        wt_ca = write_sequential_ca_pdb(wt_structure, temporary_directory / "wt_ca.pdb")
        command = (
            os.fspath(executable),
            os.fspath(candidate_ca),
            os.fspath(wt_ca),
            "-mol",
            "prot",
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise USAlignError(f"Could not execute US-align: {exc}.") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise USAlignError(
                f"US-align exited with status {completed.returncode}: {detail[:1000]}"
            )
        parsed = parse_usalign_output(completed.stdout)
        if parsed.candidate_coordinate_length != len(candidate_structure.residues):
            raise USAlignError(
                "US-align Structure_1 length disagrees with the strict candidate PDB parser."
            )
        if parsed.wt_coordinate_length != len(wt_structure.residues):
            raise USAlignError(
                "US-align Structure_2 length disagrees with the strict WT PDB parser."
            )
        return replace(parsed, command=command)
