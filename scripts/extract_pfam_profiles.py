#!/usr/bin/env python3
"""Extract and checksum-verify frozen profile blocks from a flat Pfam library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import yaml


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hmm_blocks(library: Path) -> Iterator[tuple[str, bytes]]:
    """Yield versioned accessions and byte-exact HMM blocks in library order."""

    block: list[bytes] = []
    with library.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith(b"HMMER3/"):
                if block:
                    raise ValueError(
                        f"Profile starting before previous terminator at line {line_number}."
                    )
                block = [line]
                continue
            if not block:
                continue
            block.append(line)
            if line.strip() != b"//":
                continue
            accession_lines = [item for item in block if item.startswith(b"ACC   ")]
            if len(accession_lines) != 1:
                raise ValueError(
                    f"HMM block ending at line {line_number} has "
                    f"{len(accession_lines)} ACC records."
                )
            try:
                accession = accession_lines[0].split(None, 1)[1].strip().decode("ascii")
            except (IndexError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"Invalid HMM accession ending at line {line_number}."
                ) from exc
            yield accession, b"".join(block)
            block = []
    if block:
        raise ValueError(f"Pfam library ends inside an unterminated HMM block: {library}")


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Pfam profile manifest must have schema_version: 1.")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("Pfam profile manifest contains no profiles.")
    profiles: dict[str, dict[str, str]] = {}
    seen_accessions: set[str] = set()
    for reference_id, raw_record in raw_profiles.items():
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError("Pfam manifest reference IDs must be non-empty strings.")
        if not isinstance(raw_record, dict):
            raise ValueError(f"Invalid profile record for {reference_id}.")
        record = {str(key): str(value) for key, value in raw_record.items()}
        required = {"accession", "target_pfam", "sha256"}
        missing = required - set(record)
        if missing:
            raise ValueError(f"Profile {reference_id} lacks fields: {sorted(missing)}")
        if record["accession"].split(".", 1)[0] != record["target_pfam"]:
            raise ValueError(f"Profile accession and target disagree for {reference_id}.")
        if record["accession"] in seen_accessions:
            raise ValueError(f"Duplicate profile accession: {record['accession']}")
        if len(record["sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in record["sha256"]
        ):
            raise ValueError(f"Invalid profile SHA-256 for {reference_id}.")
        seen_accessions.add(record["accession"])
        profiles[reference_id] = record
    return profiles


def verify_outputs(
    profiles: dict[str, dict[str, str]], output_directory: Path
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for reference_id, expected in sorted(profiles.items()):
        path = output_directory / f"{reference_id}.hmm"
        if not path.is_file():
            raise FileNotFoundError(f"Extracted Pfam profile is missing: {path}")
        payload = path.read_bytes()
        observed_sha256 = sha256_bytes(payload)
        if observed_sha256 != expected["sha256"]:
            raise ValueError(
                f"Profile checksum mismatch for {reference_id}: "
                f"{observed_sha256} != {expected['sha256']}"
            )
        accessions = [
            line.split(None, 1)[1].strip().decode("ascii")
            for line in payload.splitlines()
            if line.startswith(b"ACC   ")
        ]
        if accessions != [expected["accession"]]:
            raise ValueError(
                f"Profile accession mismatch for {reference_id}: {accessions!r}"
            )
        records.append(
            {
                "reference_id": reference_id,
                "filename": path.name,
                "accession": expected["accession"],
                "target_pfam": expected["target_pfam"],
                "sha256": observed_sha256,
            }
        )
    return records


def extract_profiles(
    library: Path,
    profiles: dict[str, dict[str, str]],
    output_directory: Path,
) -> list[dict[str, str]]:
    wanted = {record["accession"] for record in profiles.values()}
    blocks: dict[str, bytes] = {}
    for accession, payload in hmm_blocks(library):
        if accession not in wanted:
            continue
        if accession in blocks:
            raise ValueError(f"Pfam library contains duplicate accession {accession}.")
        blocks[accession] = payload
    missing = wanted - set(blocks)
    if missing:
        raise ValueError(f"Pfam library lacks frozen accession(s): {sorted(missing)}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for reference_id, expected in sorted(profiles.items()):
        payload = blocks[expected["accession"]]
        observed_sha256 = sha256_bytes(payload)
        if observed_sha256 != expected["sha256"]:
            raise ValueError(
                f"Source profile checksum mismatch for {reference_id}: "
                f"{observed_sha256} != {expected['sha256']}"
            )
        destination = output_directory / f"{reference_id}.hmm"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return verify_outputs(profiles, output_directory)


def write_extraction_manifest(
    output_directory: Path,
    records: list[dict[str, str]],
    *,
    source_library_sha256: str | None,
) -> None:
    payload = {
        "schema_version": 1,
        "extraction": "byte_exact_flat_hmm_block",
        "source_library_sha256": source_library_sha256,
        "profiles": records,
    }
    destination = output_directory / "profile_manifest.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing output directory without reading Pfam-A.",
    )
    args = parser.parse_args()

    profiles = load_manifest(args.manifest)
    if args.verify_only:
        records = verify_outputs(profiles, args.output_dir)
        library_sha256 = None
    else:
        if args.library is None or not args.library.is_file():
            raise FileNotFoundError("--library must name the frozen flat Pfam-A.hmm file.")
        records = extract_profiles(args.library, profiles, args.output_dir)
        library_sha256 = sha256_file(args.library)
    write_extraction_manifest(
        args.output_dir,
        records,
        source_library_sha256=library_sha256,
    )
    print(f"Verified {len(records)} byte-exact profiles in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
