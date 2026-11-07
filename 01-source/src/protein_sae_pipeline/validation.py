"""Input-manifest validation and canonicalization."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Iterable

from .config import PipelineConfig
from .io import safe_relative_path, safe_unresolved_relative_path, sha256_file
from .keys import CandidateKey, normalize_sequence, sequence_sha256
from .schemas import (
    Candidate,
    CandidateStructure,
    Reference,
    ReferencePosition,
    ValidatedInputs,
)


def _rows(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        extra = fields - required
        if missing or extra:
            raise ValueError(
                f"{label} columns differ: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return list(reader)


def _parse_bool(value: str, *, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be true or false")


def _hmm_accession(path: Path, *, reference_id: str) -> str:
    """Read the single HMMER ``ACC`` record without reserializing the profile."""

    accession_lines = [
        line for line in path.read_bytes().splitlines() if line.startswith(b"ACC   ")
    ]
    if len(accession_lines) != 1:
        raise ValueError(
            f"Pfam profile for {reference_id} must contain exactly one ACC record; "
            f"found {len(accession_lines)}"
        )
    try:
        return accession_lines[0].split(None, 1)[1].strip().decode("ascii")
    except (IndexError, UnicodeDecodeError) as error:
        raise ValueError(f"Pfam profile for {reference_id} has an invalid ACC record") from error


def load_and_validate_inputs(
    config: PipelineConfig,
    data_root: str | Path,
    *,
    validate_candidate_structure_files: bool = True,
) -> ValidatedInputs:
    root = Path(data_root).resolve()
    paths = config.paths

    reference_rows = _rows(
        safe_relative_path(root, paths["references"], label="references path"),
        {
            "reference_id",
            "reference_sequence_id",
            "sequence",
            "structure_path",
            "chain_id",
            "pfam_accession",
            "pfam_profile_path",
            "figure_focal_position",
        },
        "reference manifest",
    )
    references: dict[str, Reference] = {}
    reference_sequence_ids: set[str] = set()
    for line, row in enumerate(reference_rows, start=2):
        reference_id = row["reference_id"].strip()
        sequence_id = row["reference_sequence_id"].strip()
        if not reference_id or reference_id in references:
            raise ValueError(f"Duplicate/empty reference_id at references row {line}")
        if not sequence_id or sequence_id in reference_sequence_ids:
            raise ValueError(f"Duplicate/empty reference_sequence_id at references row {line}")
        if reference_id not in config.raw["references"]:
            raise ValueError(f"Unknown reference_id at references row {line}: {reference_id!r}")
        reference_config = config.raw["references"][reference_id]
        expected_accession = str(reference_config["pfam_accession"])
        manifest_accession = row["pfam_accession"].strip()
        if manifest_accession != expected_accession:
            raise ValueError(
                f"reference {reference_id} Pfam accession differs from config: "
                f"{manifest_accession!r} != {expected_accession!r}"
            )
        sequence = normalize_sequence(row["sequence"], label=f"reference {reference_id}")
        chain = row["chain_id"].strip()
        if len(chain) != 1:
            raise ValueError(f"reference {reference_id} chain_id must be one character")
        focal_raw = row["figure_focal_position"].strip()
        focal = int(focal_raw) if focal_raw else None
        if focal is not None and not 1 <= focal <= len(sequence):
            raise ValueError(f"reference {reference_id} focal position is out of range")
        structure_path = safe_relative_path(
            root, row["structure_path"], label=f"reference {reference_id} structure_path"
        )
        profile_path = safe_relative_path(
            root, row["pfam_profile_path"], label=f"reference {reference_id} pfam_profile_path"
        )
        if not structure_path.is_file():
            raise FileNotFoundError(structure_path)
        if not profile_path.is_file():
            raise FileNotFoundError(profile_path)
        observed_profile_sha256 = sha256_file(profile_path)
        profile_accession = _hmm_accession(profile_path, reference_id=reference_id)
        if profile_accession != expected_accession:
            raise ValueError(
                f"reference {reference_id} HMM ACC differs from config: "
                f"{profile_accession!r} != {expected_accession!r}"
            )
        references[reference_id] = Reference(
            reference_id=reference_id,
            reference_sequence_id=sequence_id,
            sequence=sequence,
            sequence_sha256=sequence_sha256(sequence),
            sequence_length=len(sequence),
            structure_path=structure_path,
            chain_id=chain,
            pfam_accession=expected_accession,
            pfam_profile_sha256=observed_profile_sha256,
            pfam_profile_path=profile_path,
            figure_focal_position=focal,
        )
        reference_sequence_ids.add(sequence_id)

    if set(references) != set(config.raw["references"]):
        raise ValueError(
            "Reference IDs differ from config: "
            f"manifest={sorted(references)}, config={sorted(config.raw['references'])}"
        )

    candidate_rows = _rows(
        safe_relative_path(root, paths["candidates"], label="candidates path"),
        {"sequence_id", "reference_id", "generation_method", "sequence"},
        "candidate manifest",
    )
    candidates: list[Candidate] = []
    seen_ids: set[tuple[str, str]] = set()
    seen_keys: set[CandidateKey] = set()
    for manifest_row, row in enumerate(candidate_rows):
        line = manifest_row + 2
        sequence_id = row["sequence_id"].strip()
        reference_id = row["reference_id"].strip()
        method = row["generation_method"].strip()
        if reference_id not in references:
            raise ValueError(f"Unknown reference_id at candidates row {line}: {reference_id!r}")
        if not sequence_id:
            raise ValueError(f"Empty sequence_id at candidates row {line}")
        id_key = (reference_id, sequence_id)
        if id_key in seen_ids:
            raise ValueError(f"Duplicate sequence_id within reference: {id_key}")
        if method not in config.generation_methods:
            raise ValueError(f"Unknown generation_method at candidates row {line}: {method!r}")
        sequence = normalize_sequence(row["sequence"], label=f"candidate row {line}")
        if len(sequence) != references[reference_id].sequence_length:
            raise ValueError(
                f"candidate row {line} length {len(sequence)} differs from WT length "
                f"{references[reference_id].sequence_length}"
            )
        digest = sequence_sha256(sequence)
        key = CandidateKey(reference_id, digest)
        if key in seen_keys:
            raise ValueError(f"Duplicate candidate sequence for reference: {key.display()}")
        candidates.append(
            Candidate(
                manifest_row,
                sequence_id,
                reference_id,
                method,
                sequence,
                digest,
                len(sequence),
            )
        )
        seen_ids.add(id_key)
        seen_keys.add(key)
    if not candidates:
        raise ValueError("Candidate manifest is empty")

    position_rows = _rows(
        safe_relative_path(root, paths["reference_positions"], label="positions path"),
        {
            "reference_id",
            "sequence_position",
            "residue",
            "observed",
            "chain_id",
            "author_residue_number",
            "insertion_code",
        },
        "reference-position map",
    )
    positions: dict[str, list[ReferencePosition]] = defaultdict(list)
    seen_positions: set[tuple[str, int]] = set()
    for line, row in enumerate(position_rows, start=2):
        reference_id = row["reference_id"].strip()
        if reference_id not in references:
            raise ValueError(f"Unknown reference in position row {line}: {reference_id!r}")
        position = int(row["sequence_position"])
        key = (reference_id, position)
        if key in seen_positions:
            raise ValueError(f"Duplicate position row: {key}")
        reference = references[reference_id]
        if not 1 <= position <= reference.sequence_length:
            raise ValueError(f"Out-of-range position row: {key}")
        residue = normalize_sequence(row["residue"], label=f"position row {line} residue")
        if len(residue) != 1 or residue != reference.sequence[position - 1]:
            raise ValueError(f"Position row {key} residue disagrees with reference sequence")
        observed = _parse_bool(row["observed"], label=f"position row {line} observed")
        chain = row["chain_id"].strip() or None
        number_raw = row["author_residue_number"].strip()
        number = int(number_raw) if number_raw else None
        insertion = row["insertion_code"].strip()
        if observed:
            if chain != reference.chain_id or number is None:
                raise ValueError(f"Observed position row {key} lacks the declared chain/residue")
        elif chain is not None or number is not None or insertion:
            raise ValueError(f"Unobserved position row {key} must not claim coordinates")
        positions[reference_id].append(
            ReferencePosition(reference_id, position, residue, observed, chain, number, insertion)
        )
        seen_positions.add(key)
    for reference_id, reference in references.items():
        observed_positions = sorted(item.sequence_position for item in positions[reference_id])
        expected_positions = list(range(1, reference.sequence_length + 1))
        if observed_positions != expected_positions:
            raise ValueError(f"Position map is not complete for reference {reference_id}")
        positions[reference_id].sort(key=lambda item: item.sequence_position)

    structure_rows = _rows(
        safe_relative_path(root, paths["structures"], label="structures path"),
        {"reference_id", "sequence_sha256", "structure_path", "chain_id"},
        "candidate structure manifest",
    )
    structures: dict[CandidateKey, CandidateStructure] = {}
    for line, row in enumerate(structure_rows, start=2):
        key = CandidateKey(row["reference_id"].strip(), row["sequence_sha256"].strip())
        if key not in seen_keys:
            raise ValueError(f"Unexpected candidate structure key at row {line}: {key.display()}")
        if key in structures:
            raise ValueError(f"Duplicate candidate structure key: {key.display()}")
        chain = row["chain_id"].strip()
        if len(chain) != 1:
            raise ValueError(f"Candidate structure row {line} chain_id must be one character")
        if validate_candidate_structure_files:
            path = safe_relative_path(
                root, row["structure_path"], label=f"structure row {line} path"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
        else:
            path = safe_unresolved_relative_path(
                root, row["structure_path"], label=f"structure row {line} path"
            )
        structures[key] = CandidateStructure(key, path, chain)
    if set(structures) != seen_keys:
        missing = sorted(key.display() for key in seen_keys - set(structures))
        raise ValueError(f"Candidate structure coverage is incomplete: {missing[:10]}")

    expected_total = config.expected_counts.get("full_total")
    if expected_total is not None and int(expected_total) != len(candidates):
        raise ValueError(
            f"Candidate count differs from config: {len(candidates)} != {int(expected_total)}"
        )
    for reference_id, reference_config in config.raw["references"].items():
        expected = int(reference_config["expected_full"])
        observed = sum(candidate.reference_id == reference_id for candidate in candidates)
        if observed != expected:
            raise ValueError(
                f"Candidate count differs for {reference_id}: {observed} != {expected}"
            )

    return ValidatedInputs(
        config.dataset_id,
        root,
        tuple(candidates),
        references,
        {key: tuple(value) for key, value in positions.items()},
        structures,
    )


def canonical_candidate_rows(inputs: ValidatedInputs) -> Iterable[dict[str, object]]:
    for candidate in inputs.candidates:
        yield {
            "manifest_row": candidate.manifest_row,
            "sequence_id": candidate.sequence_id,
            "reference_id": candidate.reference_id,
            "generation_method": candidate.generation_method,
            "sequence": candidate.sequence,
            "sequence_sha256": candidate.sequence_sha256,
            "sequence_length": candidate.sequence_length,
        }
