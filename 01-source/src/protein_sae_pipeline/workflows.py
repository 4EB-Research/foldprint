"""File-oriented workflow helpers used by the small public scripts."""

from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

import pandas as pd

from .config import PipelineConfig
from .io import atomic_csv, atomic_json, sha256_file
from .schemas import ValidatedInputs
from .scoring.identity import calculate_positional_identity
from .scoring.mmseqs import run_mmseqs_easy_search
from .scoring.pfam import run_hmmsearch
from .scoring.usalign import run_usalign
from .structures.pdb import ResidueID, parse_pdb


KEYS = ["reference_id", "sequence_sha256"]
PFAM_CENSOR_FLOOR_COLUMNS = [
    "reference_id",
    "pfam_full_sequence_censor_floor",
    "pfam_best_domain_censor_floor",
]


def load_pfam_censor_floors(
    path: Path, expected_reference_ids: set[str]
) -> tuple[dict[str, tuple[float, float]], str]:
    """Load byte-bound full-landscape floors for representative recomputation."""

    frame = pd.read_csv(path)
    if list(frame.columns) != PFAM_CENSOR_FLOOR_COLUMNS:
        raise ValueError(
            "Pfam censor-floor columns must be exactly "
            f"{PFAM_CENSOR_FLOOR_COLUMNS} in that order"
        )
    if frame["reference_id"].duplicated().any():
        raise ValueError("Pfam censor-floor table contains duplicate references")
    observed = set(frame["reference_id"].astype(str))
    if observed != expected_reference_ids:
        raise ValueError(
            "Pfam censor-floor references differ from the validated references"
        )
    result: dict[str, tuple[float, float]] = {}
    for row in frame.itertuples(index=False):
        full_floor = float(row.pfam_full_sequence_censor_floor)
        domain_floor = float(row.pfam_best_domain_censor_floor)
        if not math.isfinite(full_floor) or not math.isfinite(domain_floor):
            raise ValueError("Pfam censor-floor values must be finite")
        result[str(row.reference_id)] = (full_floor, domain_floor)
    return result, sha256_file(path)


def _fasta(path: Path, records: list[tuple[str, str]], *, prefix: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f">{prefix}{identifier}\n{sequence}\n" for identifier, sequence in records),
        encoding="ascii",
    )


def _score_metadata(
    path: Path,
    *,
    inputs: ValidatedInputs,
    method: str,
    rows: int,
    parameters: dict[str, Any],
) -> None:
    atomic_json(
        path.with_suffix(path.suffix + ".json"),
        {
            "schema_version": 1,
            "dataset_id": inputs.dataset_id,
            "method": method,
            "rows": rows,
            "complete": rows == len(inputs.candidates),
            "parameters": parameters,
            "score_table_sha256": sha256_file(path),
        },
    )


def score_identity(inputs: ValidatedInputs, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in inputs.candidates:
        reference = inputs.references[candidate.reference_id]
        score = calculate_positional_identity(reference.sequence, candidate.sequence)
        rows.append(
            {
                "reference_id": candidate.reference_id,
                "sequence_sha256": candidate.sequence_sha256,
                "positional_matches": score.matches,
                "positional_mismatches": score.mismatches,
                "positional_identity_fraction": score.fraction,
                "positional_identity_percent": score.percent,
            }
        )
    columns = KEYS + [
        "positional_matches",
        "positional_mismatches",
        "positional_identity_fraction",
        "positional_identity_percent",
    ]
    atomic_csv(output_path, rows, fieldnames=columns)
    _score_metadata(
        output_path,
        inputs=inputs,
        method="positional_identity",
        rows=len(rows),
        parameters={
            "correspondence": "same sequence position",
            "gap_policy": "unequal lengths rejected",
        },
    )
    return pd.DataFrame(rows, columns=columns)

def score_pfam(
    inputs: ValidatedInputs,
    config: PipelineConfig,
    output_path: Path,
    *,
    executable: str | None = None,
    cpus: int = 1,
    censor_floors: Mapping[str, tuple[float, float]] | None = None,
    censor_floor_table_sha256: str | None = None,
) -> pd.DataFrame:
    tool = config.native_tool("hmmer")
    executable = executable or str(tool["executable"])
    if censor_floors is not None and set(censor_floors) != set(inputs.references):
        raise ValueError(
            "Frozen Pfam censor-floor references differ from the validated references"
        )
    if (censor_floors is None) != (censor_floor_table_sha256 is None):
        raise ValueError(
            "Pfam censor floors and their source-table checksum must be supplied together"
        )
    rows: list[dict[str, Any]] = []
    run_root = output_path.parent / "pfam_work"
    for reference_id in sorted(inputs.references):
        reference = inputs.references[reference_id]
        candidates = [item for item in inputs.candidates if item.reference_id == reference_id]
        target_root = run_root / reference_id
        target_root.mkdir(parents=True, exist_ok=True)
        fasta = target_root / "candidates.fasta"
        tblout = target_root / "scores.tblout"
        _fasta(
            fasta,
            [(candidate.sequence_sha256, candidate.sequence) for candidate in candidates],
            prefix="seq_",
        )
        frozen_floor = censor_floors[reference_id] if censor_floors is not None else None
        result = run_hmmsearch(
            profile_path=reference.pfam_profile_path,
            candidate_fasta_path=fasta,
            tblout_path=tblout,
            expected_sequence_ids=[candidate.sequence_sha256 for candidate in candidates],
            executable=executable,
            cpus=cpus,
            censor_floor=frozen_floor,
        )
        finite_hit_scores = [
            score.pfam_full_sequence_bit_score
            for score in result.scores
            if not score.pfam_score_left_censored
        ]
        if not finite_hit_scores and frozen_floor is None:
            raise RuntimeError(f"Reference {reference_id} has no finite Pfam hit")
        floor = (
            float(frozen_floor[0])
            if frozen_floor is not None
            else min(finite_hit_scores) - 1.0
        )
        for score in result.scores:
            rows.append(
                {
                    "reference_id": reference_id,
                    "sequence_sha256": score.sequence_id,
                    "pfam_accession": reference.pfam_accession,
                    "pfam_full_sequence_evalue": score.pfam_full_sequence_evalue,
                    "pfam_full_sequence_bit_score": score.pfam_full_sequence_bit_score,
                    "pfam_best_domain_evalue": score.pfam_best_domain_evalue,
                    "pfam_best_domain_bit_score": score.pfam_best_domain_bit_score,
                    "pfam_score_left_censored": score.pfam_score_left_censored,
                    "pfam_censor_floor": floor,
                }
            )
    columns = KEYS + [
        "pfam_accession",
        "pfam_full_sequence_evalue",
        "pfam_full_sequence_bit_score",
        "pfam_best_domain_evalue",
        "pfam_best_domain_bit_score",
        "pfam_score_left_censored",
        "pfam_censor_floor",
    ]
    if len(rows) != len(inputs.candidates):
        raise RuntimeError("Pfam workflow did not return exact candidate coverage")
    atomic_csv(output_path, rows, fieldnames=columns)
    _score_metadata(
        output_path,
        inputs=inputs,
        method="pfam_hmmsearch",
        rows=len(rows),
        parameters={
            "program": "hmmsearch",
            "version": "3.4",
            "arguments": list(tool["arguments"]),
            "score": tool["score"],
            "no_hit": tool["no_hit"],
            "censor_floor_source": (
                "frozen_full_landscape_table"
                if censor_floors is not None
                else "complete_input_landscape"
            ),
            "censor_floor_table_sha256": censor_floor_table_sha256,
            "profile_contracts": {
                reference_id: {
                    "accession": reference.pfam_accession,
                    "sha256": reference.pfam_profile_sha256,
                }
                for reference_id, reference in sorted(inputs.references.items())
            },
        },
    )
    return pd.DataFrame(rows, columns=columns)


def score_mmseqs(
    inputs: ValidatedInputs,
    config: PipelineConfig,
    output_path: Path,
    *,
    executable: str | None = None,
    threads: int = 1,
) -> pd.DataFrame:
    tool = config.native_tool("mmseqs2")
    executable = executable or str(tool["executable"])
    rows: list[dict[str, Any]] = []
    run_root = output_path.parent / "mmseqs_work"
    for reference_id in sorted(inputs.references):
        reference = inputs.references[reference_id]
        candidates = [item for item in inputs.candidates if item.reference_id == reference_id]
        target_root = run_root / reference_id
        target_root.mkdir(parents=True, exist_ok=True)
        query_fasta = target_root / "candidates.fasta"
        wt_fasta = target_root / "wt.fasta"
        raw_results = target_root / "results.tsv"
        temporary_directory = target_root / "tmp"
        target_id = f"WT::{reference_id}"
        _fasta(query_fasta, [(item.sequence_sha256, item.sequence) for item in candidates])
        _fasta(wt_fasta, [(target_id, reference.sequence)])
        result = run_mmseqs_easy_search(
            query_fasta_path=query_fasta,
            target_wt_fasta_path=wt_fasta,
            raw_output_path=raw_results,
            temporary_directory=temporary_directory,
            expected_query_ids=[item.sequence_sha256 for item in candidates],
            target_id=target_id,
            executable=executable,
            threads=threads,
        )
        for score in result.scores:
            rows.append(
                {
                    "reference_id": reference_id,
                    "sequence_sha256": score.sequence_id,
                    "mmseqs2_bit_score": score.mmseqs2_bit_score,
                    "mmseqs2_evalue": score.mmseqs2_evalue,
                    "mmseqs2_percent_identity": score.mmseqs2_percent_identity,
                    "mmseqs2_alignment_length": score.mmseqs2_alignment_length,
                    "mmseqs2_query_coverage": score.mmseqs2_query_coverage,
                    "mmseqs2_target_coverage": score.mmseqs2_target_coverage,
                    "mmseqs2_hit": score.mmseqs2_hit,
                }
            )
    columns = KEYS + [
        "mmseqs2_bit_score",
        "mmseqs2_evalue",
        "mmseqs2_percent_identity",
        "mmseqs2_alignment_length",
        "mmseqs2_query_coverage",
        "mmseqs2_target_coverage",
        "mmseqs2_hit",
    ]
    if len(rows) != len(inputs.candidates):
        raise RuntimeError("MMseqs2 workflow did not return exact candidate coverage")
    atomic_csv(output_path, rows, fieldnames=columns)
    _score_metadata(
        output_path,
        inputs=inputs,
        method="mmseqs2_candidate_to_cognate_wt",
        rows=len(rows),
        parameters={
            "release": tool["release"],
            "commit": tool["commit"],
            "evalue": tool["evalue"],
            "sensitivity": tool["sensitivity"],
            "max_sequences": tool["max_sequences"],
            "format": tool["format"],
        },
    )
    return pd.DataFrame(rows, columns=columns)


def _require_usalign_version(executable: str, expected: str) -> tuple[str, str]:
    resolved = shutil.which(executable) if "/" not in executable else executable
    if resolved is None or not Path(resolved).is_file():
        raise FileNotFoundError(f"US-align executable was not found: {executable}")
    completed = subprocess.run([resolved], capture_output=True, text=True, check=False)
    banner = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    match = re.search(r"US-align\s*\(Version\s+([^\)]+)\)", banner, re.IGNORECASE)
    if match is None:
        raise RuntimeError("Could not parse the US-align version banner")
    observed = match.group(1).strip()
    if observed != expected:
        raise RuntimeError(f"Expected US-align {expected}; observed {observed}")
    return str(Path(resolved).resolve()), observed


def _validate_wt_structure(inputs: ValidatedInputs, reference_id: str):  # type: ignore[no-untyped-def]
    reference = inputs.references[reference_id]
    structure = parse_pdb(reference.structure_path, chain_id=reference.chain_id)
    observed_positions = [
        position for position in inputs.reference_positions[reference_id] if position.observed
    ]
    expected_ids = [
        ResidueID(
            str(position.chain_id),
            int(position.author_residue_number),
            position.insertion_code,
        )
        for position in observed_positions
    ]
    observed_ids = [residue.author_id for residue in structure.residues]
    if observed_ids != expected_ids:
        raise ValueError(f"WT structure residues disagree with the position map for {reference_id}")
    expected_sequence = "".join(position.residue for position in observed_positions)
    if structure.sequence != expected_sequence:
        raise ValueError(f"WT structure sequence disagrees with observed WT positions for {reference_id}")
    return structure


def score_usalign(
    inputs: ValidatedInputs,
    config: PipelineConfig,
    output_path: Path,
    *,
    executable: str | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    tool = config.native_tool("usalign")
    requested = executable or str(tool["executable"])
    resolved, version = _require_usalign_version(requested, str(tool["version"]))
    if workers <= 0:
        raise ValueError("US-align workers must be positive")
    wt_structures = {
        reference_id: _validate_wt_structure(inputs, reference_id)
        for reference_id in inputs.references
    }
    temp_root = output_path.parent / "usalign_work"
    temp_root.mkdir(parents=True, exist_ok=True)

    def score_one(candidate):  # type: ignore[no-untyped-def]
        candidate_structure = inputs.structures[candidate.key]
        parsed_candidate = parse_pdb(
            candidate_structure.structure_path,
            chain_id=candidate_structure.chain_id,
            expected_sequence=candidate.sequence,
        )
        result = run_usalign(
            parsed_candidate,
            wt_structures[candidate.reference_id],
            executable=resolved,
            temp_root=temp_root,
        )
        return candidate.manifest_row, {
            "reference_id": candidate.reference_id,
            "sequence_sha256": candidate.sequence_sha256,
            "candidate_coordinate_length": result.candidate_coordinate_length,
            "wt_coordinate_length": result.wt_coordinate_length,
            "usalign_tm_score_candidate_norm": result.tm_score_candidate_normalized,
            "usalign_tm_score_wt_norm": result.tm_score_wt_normalized,
            "usalign_aligned_length": result.aligned_length,
            "usalign_rmsd_angstrom": result.rmsd_angstrom,
            "usalign_aligned_sequence_identity": result.aligned_sequence_identity,
        }

    if workers == 1:
        scored = [score_one(candidate) for candidate in inputs.candidates]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scored = list(executor.map(score_one, inputs.candidates))
    rows = [row for _index, row in sorted(scored, key=lambda item: item[0])]
    columns = KEYS + [
        "candidate_coordinate_length",
        "wt_coordinate_length",
        "usalign_tm_score_candidate_norm",
        "usalign_tm_score_wt_norm",
        "usalign_aligned_length",
        "usalign_rmsd_angstrom",
        "usalign_aligned_sequence_identity",
    ]
    if len(rows) != len(inputs.candidates):
        raise RuntimeError("US-align workflow did not return exact candidate coverage")
    atomic_csv(output_path, rows, fieldnames=columns)
    _score_metadata(
        output_path,
        inputs=inputs,
        method="usalign_alignment_optimized_wt_normalized",
        rows=len(rows),
        parameters={
            "version": version,
            "source_commit": tool["source_commit"],
            "arguments": tool["arguments"],
            "structure_1": "candidate",
            "structure_2": "wild_type",
            "primary_normalization": tool["primary_normalization"],
            "fixed_position_flag_used": False,
        },
    )
    return pd.DataFrame(rows, columns=columns)
