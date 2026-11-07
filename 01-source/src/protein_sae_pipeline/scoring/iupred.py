"""Adapter for a user-installed, unmodified official IUPred3 distribution.

Adapted from mutavault/sieve/audit/iupred3_adapter.py. The licensed predictor
and its data files are deliberately external to this repository.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


def package_provenance(package: Path) -> dict:
    names = ["LICENSE", "README", "P53_HUMAN.seq", "__init__.py", "iupred3.py", "iupred3_lib.py"]
    names += [str(p.relative_to(package)) for p in sorted((package / "data").iterdir()) if p.is_file()]
    hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}
    return {
        "name": "IUPred3", "version": "Official standalone distribution; no numeric version declared",
        "file_sha256": hashes,
        "package_fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }


def load_library(package: Path):
    path = package.resolve() / "iupred3_lib.py"
    if not path.is_file():
        raise FileNotFoundError(f"Official IUPred3 package is missing: {path}")
    spec = importlib.util.spec_from_file_location("protein_sae_external_iupred3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(library, sequence: str, mode: str = "long", smoothing: str = "medium") -> dict:
    if not sequence or set(sequence) - CANONICAL:
        raise ValueError("Expected a nonempty uppercase canonical amino-acid sequence")
    if mode not in {"long", "short"} or smoothing not in {"no", "medium", "strong"}:
        raise ValueError("Unsupported disorder mode or smoothing setting")
    if smoothing == "medium" and len(sequence) < 19:
        raise ValueError("Official medium smoothing requires at least 19 residues")
    scores = [float(v) for v in library.iupred(sequence, mode=mode, smoothing=smoothing)[0]]
    if len(scores) != len(sequence) or not all(math.isfinite(v) for v in scores):
        raise ValueError("Invalid predictor output length or non-finite values")
    longest = current = count = 0
    for value in scores:
        current = current + 1 if value > 0.5 else 0
        count += int(value > 0.5)
        longest = max(longest, current)
    return {
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "sequence_length": len(sequence), "mode": mode, "smoothing": smoothing,
        "scores": scores,
        "summary": {
            "mean": sum(scores) / len(scores), "fraction_disordered": count / len(scores),
            "longest_disordered_run": longest,
            "longest_disordered_run_fraction": longest / len(scores),
            "minimum": min(scores), "maximum": max(scores),
            "residues_outside_unit_interval": sum(v < 0 or v > 1 for v in scores),
        },
    }


def official_cli(package: Path, sequence: str, mode: str, smoothing: str) -> str:
    with tempfile.TemporaryDirectory(prefix="iupred3-single-") as directory:
        fasta = Path(directory) / "sequence.fasta"
        fasta.write_text(">sequence\n" + sequence + "\n")
        return subprocess.run(
            [sys.executable, str(package.resolve() / "iupred3.py"), str(fasta), mode, "-s", smoothing],
            check=True, capture_output=True, text=True,
        ).stdout
