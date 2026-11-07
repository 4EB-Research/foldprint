"""Target-Pfam scoring with the configured HMMER ``hmmsearch`` protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Mapping, Sequence


EXPECTED_HMMER_VERSION = "3.4"
HMMSEARCH_PROTOCOL = (
    "--max",
    "--noali",
    "-T",
    "-1000",
    "--domT",
    "-1000",
    "--incT",
    "-1000",
    "--incdomT",
    "-1000",
)


@dataclass(frozen=True)
class PfamHit:
    """The full-sequence and best-domain values in one HMMER ``tblout`` row."""

    full_sequence_evalue: float
    full_sequence_bit_score: float
    best_domain_evalue: float
    best_domain_bit_score: float


@dataclass(frozen=True)
class PfamScore:
    """A complete target-profile score, including explicit no-hit censoring."""

    sequence_id: str
    pfam_full_sequence_evalue: float
    pfam_full_sequence_bit_score: float
    pfam_best_domain_evalue: float
    pfam_best_domain_bit_score: float
    pfam_score_left_censored: bool


@dataclass(frozen=True)
class HmmerVersion:
    executable: str
    banner: str


@dataclass(frozen=True)
class HmmsearchRun:
    command: tuple[str, ...]
    version: HmmerVersion | None
    stderr: str
    scores: tuple[PfamScore, ...]


def _ordered_unique_ids(sequence_ids: Iterable[str]) -> tuple[str, ...]:
    ordered = tuple(str(sequence_id).strip() for sequence_id in sequence_ids)
    if not ordered or any(not sequence_id for sequence_id in ordered):
        raise ValueError("Expected sequence IDs must be non-empty.")
    if len(set(ordered)) != len(ordered):
        raise ValueError("Expected sequence IDs contain duplicates.")
    return ordered


def parse_tblout(
    path: Path,
    expected_sequence_ids: set[str],
    *,
    fasta_id_prefix: str = "seq_",
) -> dict[str, PfamHit]:
    """Parse HMMER 3.4 ``--tblout`` output for one cognate Pfam profile.

    HMMER is run as ``hmmsearch PROFILE CANDIDATES``.  Therefore the first
    ``tblout`` field is the candidate FASTA identifier.  The default prefix
    matches the private pipeline's ``seq_<sequence hash>`` FASTA records.
    """

    expected = {str(value) for value in expected_sequence_ids}
    hits: dict[str, PfamHit] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split(maxsplit=18)
        if len(fields) != 19:
            raise ValueError(
                f"Unexpected hmmsearch row at {path}:{line_number}: {line[:240]}"
            )
        raw_sequence_id = fields[0]
        if fasta_id_prefix:
            if not raw_sequence_id.startswith(fasta_id_prefix):
                raise ValueError(
                    f"Unexpected hmmsearch target ID at {path}:{line_number}: "
                    f"{raw_sequence_id!r}"
                )
            sequence_id = raw_sequence_id.removeprefix(fasta_id_prefix)
        else:
            sequence_id = raw_sequence_id
        if sequence_id not in expected:
            raise ValueError(f"hmmsearch returned unexpected sequence ID: {sequence_id}")
        try:
            hit = PfamHit(
                full_sequence_evalue=float(fields[4]),
                full_sequence_bit_score=float(fields[5]),
                best_domain_evalue=float(fields[7]),
                best_domain_bit_score=float(fields[8]),
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid numeric hmmsearch result at {path}:{line_number}"
            ) from error
        numeric = (
            hit.full_sequence_evalue,
            hit.full_sequence_bit_score,
            hit.best_domain_evalue,
            hit.best_domain_bit_score,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Non-finite hmmsearch result at {path}:{line_number}")
        if hit.full_sequence_evalue < 0 or hit.best_domain_evalue < 0:
            raise ValueError(f"Negative HMMER E-value at {path}:{line_number}")

        previous = hits.get(sequence_id)
        if (
            previous is None
            or hit.full_sequence_bit_score > previous.full_sequence_bit_score
        ):
            hits[sequence_id] = hit
    return hits


def complete_pfam_scores(
    sequence_ids: Iterable[str],
    hits: Mapping[str, PfamHit],
    *,
    censor_floor: tuple[float, float] | None = None,
) -> tuple[PfamScore, ...]:
    """Return one score per expected candidate using the configured no-hit rule.

    Missing HMMER rows are left-censored at one bit below the minimum finite
    full-sequence (and domain) score for the *entire target input*.  Call this
    once after combining every shard for a target; per-shard floors are not
    scientifically equivalent.  A representative recomputation must instead
    supply the two frozen full-landscape floors through ``censor_floor``.
    Negative finite bit scores are valid.
    """

    ordered_ids = _ordered_unique_ids(sequence_ids)
    expected = set(ordered_ids)
    extra = set(hits) - expected
    if extra:
        raise ValueError(f"Pfam hits contain unexpected sequence IDs: {sorted(extra)}")
    if not hits and censor_floor is None:
        raise ValueError("Cannot define the Pfam no-hit floor without a finite hit.")

    for sequence_id, hit in hits.items():
        numeric = (
            hit.full_sequence_evalue,
            hit.full_sequence_bit_score,
            hit.best_domain_evalue,
            hit.best_domain_bit_score,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Pfam hit is non-finite for {sequence_id}.")
        if hit.full_sequence_evalue < 0 or hit.best_domain_evalue < 0:
            raise ValueError(f"Pfam hit has a negative E-value for {sequence_id}.")

    if censor_floor is None:
        full_floor = min(hit.full_sequence_bit_score for hit in hits.values()) - 1.0
        domain_floor = min(hit.best_domain_bit_score for hit in hits.values()) - 1.0
    else:
        if len(censor_floor) != 2:
            raise ValueError("Pfam censor_floor must contain full-sequence and domain floors.")
        full_floor, domain_floor = (float(value) for value in censor_floor)
        if not math.isfinite(full_floor) or not math.isfinite(domain_floor):
            raise ValueError("Pfam censor floors must be finite.")
        if any(hit.full_sequence_bit_score <= full_floor for hit in hits.values()):
            raise ValueError("A finite Pfam full-sequence hit is not above the frozen floor.")
        if any(hit.best_domain_bit_score <= domain_floor for hit in hits.values()):
            raise ValueError("A finite Pfam domain hit is not above the frozen floor.")
    completed: list[PfamScore] = []
    for sequence_id in ordered_ids:
        hit = hits.get(sequence_id)
        if hit is None:
            completed.append(
                PfamScore(
                    sequence_id=sequence_id,
                    pfam_full_sequence_evalue=math.inf,
                    pfam_full_sequence_bit_score=full_floor,
                    pfam_best_domain_evalue=math.inf,
                    pfam_best_domain_bit_score=domain_floor,
                    pfam_score_left_censored=True,
                )
            )
        else:
            completed.append(
                PfamScore(
                    sequence_id=sequence_id,
                    pfam_full_sequence_evalue=hit.full_sequence_evalue,
                    pfam_full_sequence_bit_score=hit.full_sequence_bit_score,
                    pfam_best_domain_evalue=hit.best_domain_evalue,
                    pfam_best_domain_bit_score=hit.best_domain_bit_score,
                    pfam_score_left_censored=False,
                )
            )
    return tuple(completed)


def require_hmmer_3_4(executable: str = "hmmsearch") -> HmmerVersion:
    """Resolve ``hmmsearch`` and fail unless its help banner reports HMMER 3.4."""

    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"hmmsearch is not on PATH: {executable}")
    completed = subprocess.run(
        [resolved, "-h"], capture_output=True, text=True, check=True
    )
    banner = "\n".join(completed.stdout.splitlines()[:4])
    if f"HMMER {EXPECTED_HMMER_VERSION}" not in banner:
        raise RuntimeError(
            f"Expected HMMER {EXPECTED_HMMER_VERSION}; observed:\n{banner}"
        )
    return HmmerVersion(executable=str(Path(resolved).resolve()), banner=banner)


def build_hmmsearch_command(
    *,
    executable: str,
    profile_path: Path,
    candidate_fasta_path: Path,
    tblout_path: Path,
    cpus: int,
) -> tuple[str, ...]:
    """Build the configured target-profile command."""

    if cpus <= 0:
        raise ValueError("HMMER CPU count must be positive.")
    return (
        executable,
        "--cpu",
        str(cpus),
        *HMMSEARCH_PROTOCOL,
        "--tblout",
        str(tblout_path),
        str(profile_path),
        str(candidate_fasta_path),
    )


def run_hmmsearch(
    *,
    profile_path: Path,
    candidate_fasta_path: Path,
    tblout_path: Path,
    expected_sequence_ids: Sequence[str],
    executable: str = "hmmsearch",
    cpus: int = 1,
    fasta_id_prefix: str = "seq_",
    check_version: bool = True,
    censor_floor: tuple[float, float] | None = None,
) -> HmmsearchRun:
    """Execute, parse, and complete one full target-profile scoring job."""

    version = require_hmmer_3_4(executable) if check_version else None
    resolved_executable = version.executable if version is not None else executable
    command = build_hmmsearch_command(
        executable=resolved_executable,
        profile_path=profile_path,
        candidate_fasta_path=candidate_fasta_path,
        tblout_path=tblout_path,
        cpus=cpus,
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"hmmsearch failed ({completed.returncode}): {completed.stderr[-3000:]}"
        )
    if not tblout_path.is_file():
        raise RuntimeError("hmmsearch completed without creating its tblout file.")
    ordered_ids = _ordered_unique_ids(expected_sequence_ids)
    hits = parse_tblout(
        tblout_path, set(ordered_ids), fasta_id_prefix=fasta_id_prefix
    )
    scores = complete_pfam_scores(ordered_ids, hits, censor_floor=censor_floor)
    return HmmsearchRun(
        command=command,
        version=version,
        stderr=completed.stderr.strip(),
        scores=scores,
    )
