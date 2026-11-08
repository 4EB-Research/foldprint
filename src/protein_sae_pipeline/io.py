"""Small deterministic I/O helpers shared by scripts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    """Write deterministic CSV or gzip CSV, including an empty table header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as raw:
        temporary = Path(raw.name)
        if path.suffix == ".gz":
            binary: Any = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        else:
            binary = raw
        text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        text.flush()
        if path.suffix == ".gz":
            text.detach().close()
        else:
            text.detach()
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)


def safe_relative_path(data_root: Path, raw_path: str, *, label: str) -> Path:
    """Resolve a required relative path and reject lexical or symlink escapes."""

    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to data-root, not absolute: {raw_path}")
    root = data_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes data-root: {raw_path}") from error
    return resolved


def safe_unresolved_relative_path(data_root: Path, raw_path: str, *, label: str) -> Path:
    """Validate relative-path syntax without touching the referenced filesystem.

    This is only for manifests whose files will deliberately not be accessed.
    Avoiding ``Path.resolve`` is important for very large manifests on remote
    filesystems, where resolving every absent candidate path can issue a stat.
    """

    if not raw_path.strip():
        raise ValueError(f"{label} must be a nonempty relative path")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to data-root, not absolute: {raw_path}")
    if ".." in candidate.parts:
        raise ValueError(f"{label} escapes data-root: {raw_path}")
    return data_root / candidate
