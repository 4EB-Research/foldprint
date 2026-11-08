"""Candidate-to-cognate-WT MMseqs2 scoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Mapping, Sequence


EXPECTED_MMSEQS_RELEASE = "18-8cc5c"
EXPECTED_MMSEQS_COMMIT = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
MMSEQS_RAW_FORMAT = "query,target,evalue,bits,pident,alnlen,qcov,tcov"
MMSEQS_EVALUE = 1.0
MMSEQS_SENSITIVITY = 7.5
MMSEQS_MAX_SEQS = 2000


@dataclass(frozen=True)
class MMseqsHit:
    target_id: str
    evalue: float
    bit_score: float
    percent_identity: float
    alignment_length: int
    query_coverage: float
    target_coverage: float


@dataclass(frozen=True)
class MMseqsScore:
    """A complete candidate score; alignment fields are ``None`` for no hits."""

    sequence_id: str
    wt_sequence_id: str
    mmseqs2_bit_score: float
    mmseqs2_evalue: float | None
    mmseqs2_percent_identity: float | None
    mmseqs2_alignment_length: int | None
    mmseqs2_query_coverage: float | None
    mmseqs2_target_coverage: float | None
    mmseqs2_hit: bool


@dataclass(frozen=True)
class MMseqsVersion:
    executable: str
    executable_sha256: str
    banner: str
    normalized_release: str = EXPECTED_MMSEQS_RELEASE


@dataclass(frozen=True)
class MMseqsRun:
    command: tuple[str, ...]
    version: MMseqsVersion | None
    stdout: str
    stderr: str
    scores: tuple[MMseqsScore, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_unique_ids(sequence_ids: Iterable[str]) -> tuple[str, ...]:
    ordered = tuple(str(sequence_id).strip() for sequence_id in sequence_ids)
    if not ordered or any(not sequence_id for sequence_id in ordered):
        raise ValueError("Expected query IDs must be non-empty.")
    if len(set(ordered)) != len(ordered):
        raise ValueError("Expected query IDs contain duplicates.")
    return ordered


def parse_raw_results(
    path: Path, expected_query_ids: set[str], target_id: str
) -> dict[str, MMseqsHit]:
    """Parse the pinned eight-column MMseqs2 output and retain each best hit.

    Duplicate alignments are ranked by highest bit score, then lowest E-value.
    Missing or empty output is a valid all-no-hit result.
    """

    expected = {str(value) for value in expected_query_ids}
    best: dict[str, MMseqsHit] = {}
    if not path.is_file() or path.stat().st_size == 0:
        return best
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                raise ValueError(
                    f"Unexpected raw MMseqs2 field count at {path}:{line_number}: "
                    f"expected 8, observed {len(fields)}"
                )
            query, observed_target = fields[:2]
            if query not in expected:
                raise ValueError(f"MMseqs2 returned unexpected query: {query}")
            if observed_target != target_id:
                raise ValueError(
                    f"MMseqs2 returned non-cognate target for {query}: "
                    f"{observed_target!r}"
                )
            try:
                hit = MMseqsHit(
                    target_id=observed_target,
                    evalue=float(fields[2]),
                    bit_score=float(fields[3]),
                    percent_identity=float(fields[4]),
                    alignment_length=int(fields[5]),
                    query_coverage=float(fields[6]),
                    target_coverage=float(fields[7]),
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric MMseqs2 result at {path}:{line_number}"
                ) from error
            numeric = (
                hit.evalue,
                hit.bit_score,
                hit.percent_identity,
                hit.query_coverage,
                hit.target_coverage,
            )
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"Non-finite MMseqs2 result at {path}:{line_number}")
            if (
                hit.evalue < 0
                or hit.alignment_length <= 0
                or not 0.0 <= hit.percent_identity <= 100.0
                or not 0.0 <= hit.query_coverage <= 1.0
                or not 0.0 <= hit.target_coverage <= 1.0
            ):
                raise ValueError(f"Out-of-range MMseqs2 result at {path}:{line_number}")

            previous = best.get(query)
            if previous is None or (hit.bit_score, -hit.evalue) > (
                previous.bit_score,
                -previous.evalue,
            ):
                best[query] = hit
    return best


def complete_mmseqs_scores(
    sequence_ids: Iterable[str],
    *,
    wt_sequence_id: str,
    hits: Mapping[str, MMseqsHit],
) -> tuple[MMseqsScore, ...]:
    """Return exact query coverage, assigning the configured no-hit score."""

    ordered_ids = _ordered_unique_ids(sequence_ids)
    wt_sequence_id = str(wt_sequence_id).strip()
    if not wt_sequence_id:
        raise ValueError("WT sequence ID must be non-empty.")
    extra = set(hits) - set(ordered_ids)
    if extra:
        raise ValueError(f"MMseqs2 hits contain unexpected query IDs: {sorted(extra)}")

    completed: list[MMseqsScore] = []
    for sequence_id in ordered_ids:
        hit = hits.get(sequence_id)
        if hit is None:
            completed.append(
                MMseqsScore(
                    sequence_id=sequence_id,
                    wt_sequence_id=wt_sequence_id,
                    mmseqs2_bit_score=0.0,
                    mmseqs2_evalue=None,
                    mmseqs2_percent_identity=None,
                    mmseqs2_alignment_length=None,
                    mmseqs2_query_coverage=None,
                    mmseqs2_target_coverage=None,
                    mmseqs2_hit=False,
                )
            )
            continue
        if hit.target_id != wt_sequence_id:
            raise ValueError(
                f"MMseqs2 hit for {sequence_id} targets {hit.target_id!r}, "
                f"not cognate WT {wt_sequence_id!r}."
            )
        completed.append(
            MMseqsScore(
                sequence_id=sequence_id,
                wt_sequence_id=wt_sequence_id,
                mmseqs2_bit_score=hit.bit_score,
                mmseqs2_evalue=hit.evalue,
                mmseqs2_percent_identity=hit.percent_identity,
                mmseqs2_alignment_length=hit.alignment_length,
                mmseqs2_query_coverage=hit.query_coverage,
                mmseqs2_target_coverage=hit.target_coverage,
                mmseqs2_hit=True,
            )
        )
    return tuple(completed)


def require_mmseqs_18_8cc5c(executable: str = "mmseqs") -> MMseqsVersion:
    """Resolve MMseqs2 and require the configured release."""

    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"MMseqs2 is not on PATH: {executable}")
    resolved_path = Path(resolved).resolve()
    completed = subprocess.run(
        [str(resolved_path), "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    banner = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    release_label = re.search(
        r"(?<![0-9])18[-.]8cc5c(?![0-9a-f])", banner, re.IGNORECASE
    )
    if banner != EXPECTED_MMSEQS_COMMIT and release_label is None:
        raise RuntimeError(
            f"Expected MMseqs2 release {EXPECTED_MMSEQS_RELEASE}; "
            f"observed: {banner!r}"
        )
    return MMseqsVersion(
        executable=str(resolved_path),
        executable_sha256=_sha256_file(resolved_path),
        banner=banner,
    )


def build_mmseqs_easy_search_command(
    *,
    executable: str,
    query_fasta_path: Path,
    target_wt_fasta_path: Path,
    raw_output_path: Path,
    temporary_directory: Path,
    threads: int,
) -> tuple[str, ...]:
    """Build the exact candidate-to-cognate-WT ``easy-search`` command."""

    if threads <= 0:
        raise ValueError("MMseqs2 thread count must be positive.")
    return (
        executable,
        "easy-search",
        str(query_fasta_path),
        str(target_wt_fasta_path),
        str(raw_output_path),
        str(temporary_directory),
        "-e",
        str(MMSEQS_EVALUE),
        "-s",
        str(MMSEQS_SENSITIVITY),
        "--max-seqs",
        str(MMSEQS_MAX_SEQS),
        "--threads",
        str(threads),
        "--format-output",
        MMSEQS_RAW_FORMAT,
    )


def run_mmseqs_easy_search(
    *,
    query_fasta_path: Path,
    target_wt_fasta_path: Path,
    raw_output_path: Path,
    temporary_directory: Path,
    expected_query_ids: Sequence[str],
    target_id: str,
    executable: str = "mmseqs",
    threads: int = 1,
    check_version: bool = True,
) -> MMseqsRun:
    """Execute, parse, and complete a candidate-to-cognate-WT search."""

    version = require_mmseqs_18_8cc5c(executable) if check_version else None
    resolved_executable = version.executable if version is not None else executable
    command = build_mmseqs_easy_search_command(
        executable=resolved_executable,
        query_fasta_path=query_fasta_path,
        target_wt_fasta_path=target_wt_fasta_path,
        raw_output_path=raw_output_path,
        temporary_directory=temporary_directory,
        threads=threads,
    )
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"MMseqs2 failed ({completed.returncode}): {completed.stderr[-3000:]}"
        )
    if not raw_output_path.exists():
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text("", encoding="utf-8")
    ordered_ids = _ordered_unique_ids(expected_query_ids)
    hits = parse_raw_results(raw_output_path, set(ordered_ids), target_id)
    scores = complete_mmseqs_scores(
        ordered_ids, wt_sequence_id=target_id, hits=hits
    )
    return MMseqsRun(
        command=command,
        version=version,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        scores=scores,
    )
