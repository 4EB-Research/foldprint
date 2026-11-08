#!/usr/bin/env python3
"""Regenerate the small fictional, non-biological public example dataset.

All sequences are pseudorandom and all structures/representations are
synthetic. They exercise the production schemas and scientific process but
shouldn't be interpreted as model predictions or biological results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import yaml


AMINO_ACIDS = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
THREE_LETTER = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
REFERENCE_IDS = ("fictional_alpha", "fictional_beta")
METHODS = ("fictional_method_a", "fictional_method_b")
LENGTH = 40
CODEBOOK_SIZE = 16384
TOP_K = 64
EMBEDDING_DIMENSION = 2560


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def random_sequence(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(AMINO_ACIDS, size=length).tolist())


def sequence_with_matches(
    wt: str, matches: int, rng: np.random.Generator, *, salt: int
) -> str:
    positions = np.arange(len(wt))
    local = np.random.default_rng(int(rng.integers(0, 2**32 - 1)) ^ salt)
    match_positions = set(local.choice(positions, size=matches, replace=False).tolist())
    result: list[str] = []
    for position, residue in enumerate(wt):
        if position in match_positions:
            result.append(residue)
            continue
        choices = AMINO_ACIDS[AMINO_ACIDS != residue]
        result.append(str(local.choice(choices)))
    return "".join(result)


def wt_coordinates(reference_index: int) -> np.ndarray:
    i = np.arange(LENGTH, dtype=np.float64)
    phase = 0.21 * reference_index
    return np.column_stack(
        (
            7.5 * np.cos(0.73 * i + phase) + 1.3 * np.cos(0.17 * i),
            7.5 * np.sin(0.73 * i + phase) + 1.1 * np.sin(0.23 * i),
            0.62 * i + 2.2 * np.sin(0.31 * i + phase),
        )
    )


def candidate_coordinates(
    wt: np.ndarray,
    *,
    analogue: bool,
    rng: np.random.Generator,
    index: int,
) -> np.ndarray:
    if analogue:
        angle = 0.25 + 0.013 * index
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return wt @ rotation.T + np.asarray([3.0, -2.0, 1.0]) + rng.normal(0, 0.35, wt.shape)
    steps = rng.normal(size=wt.shape)
    steps[:, 2] *= 0.22
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    steps *= 3.78
    coordinates = np.cumsum(steps, axis=0)
    coordinates += np.column_stack(
        (0.025 * np.arange(LENGTH) ** 2, np.zeros(LENGTH), np.zeros(LENGTH))
    )
    return coordinates


def pdb_text(sequence: str, ca: np.ndarray, *, chain: str = "A") -> str:
    records: list[str] = ["REMARK Fictional synthetic structure; no biological source\n"]
    serial = 1
    atom_offsets = {
        "N": np.asarray([-1.22, 0.18, -0.08]),
        "CA": np.zeros(3),
        "C": np.asarray([1.25, -0.12, 0.10]),
        "O": np.asarray([1.82, -0.78, 0.18]),
    }
    for number, (residue, center) in enumerate(zip(sequence, ca, strict=True), start=1):
        for atom, offset in atom_offsets.items():
            x, y, z = center + offset
            element = atom[0]
            records.append(
                f"ATOM  {serial:5d} {atom:^4s} {THREE_LETTER[residue]:>3s} {chain}{number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{90.00:6.2f}          {element:>2s}\n"
            )
            serial += 1
    records.extend([f"TER   {serial:5d}      {THREE_LETTER[sequence[-1]]:>3s} {chain}{LENGTH:4d}\n", "END\n"])
    return "".join(records)


def build_profile(
    root: Path,
    reference_id: str,
    wt: str,
    nearby: list[str],
    *,
    hmmbuild: str,
) -> None:
    alignment = root / "profiles" / f"{reference_id}.sto"
    profile = root / "profiles" / f"{reference_id}.hmm"
    lines = ["# STOCKHOLM 1.0", f"#=GF ID {reference_id}", f"#=GF AC FICTIONAL.{1}"]
    lines.append(f"wt       {wt}")
    for index, sequence in enumerate(nearby[:4], start=1):
        lines.append(f"near{index:<4d} {sequence}")
    lines.extend(["//", ""])
    alignment.write_text("\n".join(lines), encoding="ascii")
    subprocess.run(
        [hmmbuild, "--amino", "--informat", "stockholm", str(profile), str(alignment)],
        check=True,
        capture_output=True,
        text=True,
    )


def make_representations(
    root: Path,
    candidates: list[dict[str, object]],
    references: dict[str, str],
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed + 991)
    rows: list[dict[str, object]] = []
    for reference_index, reference_id in enumerate(REFERENCE_IDS):
        wt = references[reference_id]
        rows.append(
            {
                "manifest_row": -(reference_index + 1),
                "reference_id": reference_id,
                "sequence_sha256": digest(wt),
                "sequence_length": len(wt),
                "role": "wt_reference",
                "analogue": True,
            }
        )
        rows.extend(
            {
                "manifest_row": int(candidate["manifest_row"]),
                "reference_id": reference_id,
                "sequence_sha256": str(candidate["sequence_sha256"]),
                "sequence_length": len(str(candidate["sequence"])),
                "role": "candidate",
                "analogue": bool(candidate["intended_structural_class"] == "analogue"),
            }
            for candidate in candidates
            if candidate["reference_id"] == reference_id
        )

    n = len(rows)
    raw_mean = np.empty((n, EMBEDDING_DIMENSION), dtype=np.float32)
    residue_indices = np.empty((n * LENGTH, TOP_K), dtype=np.uint16)
    residue_values = np.empty((n * LENGTH, TOP_K), dtype=np.float32)
    reference_feature_bases: dict[str, np.ndarray] = {}
    reference_embeddings: dict[str, np.ndarray] = {}
    for reference_index, reference_id in enumerate(REFERENCE_IDS):
        base = rng.normal(size=EMBEDDING_DIMENSION).astype(np.float32)
        base /= np.linalg.norm(base.astype(np.float64))
        reference_embeddings[reference_id] = base
        reference_feature_bases[reference_id] = (
            500 + reference_index * 4000 + np.arange(TOP_K, dtype=np.uint16)
        )

    for row_index, row in enumerate(rows):
        reference_id = str(row["reference_id"])
        is_wt = row["role"] == "wt_reference"
        analogue = bool(row["analogue"])
        noise_scale = 0.0 if is_wt else (0.34 if analogue else 1.8)
        vector = reference_embeddings[reference_id] + rng.normal(
            0, noise_scale / math.sqrt(EMBEDDING_DIMENSION), EMBEDDING_DIMENSION
        ).astype(np.float32)
        raw_mean[row_index] = vector.astype(np.float32)

        base_features = reference_feature_bases[reference_id].astype(np.int64)
        for position in range(LENGTH):
            flat = row_index * LENGTH + position
            if is_wt or analogue:
                retained = 56 if analogue and not is_wt else TOP_K
            else:
                retained = 24
            features = base_features.copy()
            if retained < TOP_K:
                replacement = rng.choice(
                    np.setdiff1d(np.arange(CODEBOOK_SIZE), base_features),
                    size=TOP_K - retained,
                    replace=False,
                )
                features[retained:] = replacement
            features[0] = 12000 + (position % 17)
            if position == 19:
                features[0] = 15725
                features[1] = 1326
                features[2] = 12476
            features = np.unique(features)
            while len(features) < TOP_K:
                proposal = int(rng.integers(0, CODEBOOK_SIZE))
                features = np.unique(np.append(features, proposal))
            features = np.sort(features[:TOP_K]).astype(np.uint16)
            values = rng.uniform(0.02, 0.34, TOP_K).astype(np.float32)
            for feature, boost in ((15725, 0.98), (1326, 0.82), (12476, 0.75)):
                location = np.flatnonzero(features == feature)
                if location.size:
                    values[location[0]] = np.float32(boost if (is_wt or analogue) else 0.12)
            residue_indices[flat] = features
            residue_values[flat] = values

    profile_offsets = [0]
    profile_indices: list[np.ndarray] = []
    profile_values: list[np.ndarray] = []
    profile_norms: list[float] = []
    for row_index in range(n):
        start = row_index * LENGTH
        stop = start + LENGTH
        dense = np.zeros(CODEBOOK_SIZE, dtype=np.float32)
        for features, values in zip(
            residue_indices[start:stop], residue_values[start:stop], strict=True
        ):
            dense[features.astype(np.int64)] = np.maximum(
                dense[features.astype(np.int64)], values
            )
        active = np.flatnonzero(dense > 0).astype(np.uint16)
        values = dense[active.astype(np.int64)]
        profile_indices.append(active)
        profile_values.append(values)
        profile_offsets.append(profile_offsets[-1] + len(active))
        profile_norms.append(float(np.linalg.norm(values.astype(np.float64))))

    output = root / "representations" / "landscape_representations.npz"
    np.savez_compressed(
        output,
        manifest_rows=np.asarray([row["manifest_row"] for row in rows], dtype=np.int32),
        sequence_sha256=np.asarray([row["sequence_sha256"] for row in rows], dtype="S64"),
        sequence_lengths=np.asarray([row["sequence_length"] for row in rows], dtype=np.uint16),
        residue_offsets=np.arange(0, (n + 1) * LENGTH, LENGTH, dtype=np.uint64),
        raw_mean=raw_mean,
        sae_residue_indices=residue_indices,
        sae_residue_values=residue_values,
        sae_profile_offsets=np.asarray(profile_offsets, dtype=np.uint64),
        sae_profile_indices=np.concatenate(profile_indices).astype(np.uint16),
        sae_profile_values=np.concatenate(profile_values).astype(np.float32),
        sae_profile_norms=np.asarray(profile_norms, dtype=np.float32),
    )
    index_output = root / "representations" / "representation_index.csv"
    write_csv(
        index_output,
        ["chunk_row", "manifest_row", "reference_id", "role", "sequence_sha256"],
        [
            {
                "chunk_row": row_index,
                "manifest_row": row["manifest_row"],
                "reference_id": row["reference_id"],
                "role": row["role"],
                "sequence_sha256": row["sequence_sha256"],
            }
            for row_index, row in enumerate(rows)
        ],
    )
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "example.yaml").read_text(
            encoding="utf-8"
        )
    )
    metadata = {
        "schema_version": 1,
        "storage_schema": "protein_sae_sparse_representations_v1",
        "dataset_id": "fictional_landscape_v1",
        "origin": "synthetic_offline_fixture",
        "fictional": True,
        "notice": "Synthetic vectors for offline code-path demonstration; not model outputs.",
        "rows_including_wt": n,
        "candidate_rows": len(candidates),
        "codebook_size": CODEBOOK_SIZE,
        "top_k": TOP_K,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_sha256": file_digest(output),
        "index_sha256": file_digest(index_output),
        "esmc_contract_sha256": contract_digest(config["models"]["esmc"]),
        "sae_contract_sha256": contract_digest(config["models"]["sae"]),
    }
    (root / "representations" / "landscape_representations.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(root: Path, *, seed: int, hmmbuild: str) -> None:
    rng = np.random.default_rng(seed)
    for path in (root / "inputs", root / "structures", root / "profiles", root / "representations"):
        path.mkdir(parents=True, exist_ok=True)

    references = {reference_id: random_sequence(rng, LENGTH) for reference_id in REFERENCE_IDS}
    candidates: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []
    nearby_by_reference: dict[str, list[str]] = {key: [] for key in REFERENCE_IDS}
    used: set[tuple[str, str]] = set()
    manifest_row = 0
    for reference_index, reference_id in enumerate(REFERENCE_IDS):
        wt = references[reference_id]
        wt_ca = wt_coordinates(reference_index)
        wt_path = root / "structures" / f"{reference_id}_wt.pdb"
        wt_path.write_text(pdb_text(wt, wt_ca), encoding="ascii")
        for method_index, method in enumerate(METHODS):
            roles = (
                [("remote", "analogue", 8)] * 4
                + [("remote", "negative", 8)] * 4
                + [("near", "analogue", 30)] * 2
                + [("near", "negative", 30)] * 2
            )
            for local_index, (identity_group, structure_class, matches) in enumerate(roles):
                sequence = sequence_with_matches(
                    wt, matches, rng, salt=reference_index * 1000 + method_index * 100 + local_index
                )
                while (reference_id, digest(sequence)) in used:
                    sequence = sequence_with_matches(wt, matches, rng, salt=int(rng.integers(1, 10**9)))
                used.add((reference_id, digest(sequence)))
                sequence_id = f"{reference_id}_{method_index + 1}_{local_index + 1:02d}"
                coordinates = candidate_coordinates(
                    wt_ca,
                    analogue=structure_class == "analogue",
                    rng=np.random.default_rng(seed + manifest_row * 17 + 5),
                    index=manifest_row,
                )
                structure_path = root / "structures" / f"{sequence_id}.pdb"
                structure_path.write_text(pdb_text(sequence, coordinates), encoding="ascii")
                row = {
                    "manifest_row": manifest_row,
                    "sequence_id": sequence_id,
                    "reference_id": reference_id,
                    "generation_method": method,
                    "sequence": sequence,
                    "sequence_sha256": digest(sequence),
                    "intended_identity_group": identity_group,
                    "intended_structural_class": structure_class,
                }
                candidates.append(row)
                structure_rows.append(
                    {
                        "reference_id": reference_id,
                        "sequence_sha256": digest(sequence),
                        "structure_path": f"structures/{structure_path.name}",
                        "chain_id": "A",
                    }
                )
                if identity_group == "near":
                    nearby_by_reference[reference_id].append(sequence)
                manifest_row += 1

    write_csv(
        root / "inputs" / "candidates.csv",
        ["sequence_id", "reference_id", "generation_method", "sequence"],
        [
            {key: row[key] for key in ("sequence_id", "reference_id", "generation_method", "sequence")}
            for row in candidates
        ],
    )
    write_csv(
        root / "inputs" / "candidate_expectations.csv",
        [
            "manifest_row",
            "reference_id",
            "sequence_sha256",
            "intended_identity_group",
            "intended_structural_class",
        ],
        [
            {key: row[key] for key in (
                "manifest_row", "reference_id", "sequence_sha256", "intended_identity_group", "intended_structural_class"
            )}
            for row in candidates
        ],
    )

    reference_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    for reference_id in REFERENCE_IDS:
        wt = references[reference_id]
        build_profile(root, reference_id, wt, nearby_by_reference[reference_id], hmmbuild=hmmbuild)
        reference_rows.append(
            {
                "reference_id": reference_id,
                "reference_sequence_id": f"WT::{reference_id}",
                "sequence": wt,
                "structure_path": f"structures/{reference_id}_wt.pdb",
                "chain_id": "A",
                "pfam_accession": f"FICTIONAL_{reference_id}.1",
                "pfam_profile_path": f"profiles/{reference_id}.hmm",
                "figure_focal_position": 20 if reference_id == "fictional_alpha" else "",
            }
        )
        position_rows.extend(
            {
                "reference_id": reference_id,
                "sequence_position": position,
                "residue": residue,
                "observed": "true",
                "chain_id": "A",
                "author_residue_number": position,
                "insertion_code": "",
            }
            for position, residue in enumerate(wt, start=1)
        )
    write_csv(
        root / "inputs" / "references.csv",
        [
            "reference_id", "reference_sequence_id", "sequence", "structure_path", "chain_id",
            "pfam_accession", "pfam_profile_path", "figure_focal_position",
        ],
        reference_rows,
    )
    write_csv(
        root / "inputs" / "reference_positions.csv",
        [
            "reference_id", "sequence_position", "residue", "observed", "chain_id",
            "author_residue_number", "insertion_code",
        ],
        position_rows,
    )
    write_csv(
        root / "inputs" / "structures.csv",
        ["reference_id", "sequence_sha256", "structure_path", "chain_id"],
        structure_rows,
    )

    fixed_examples = []
    for role, structure_class in (("trace_analogue", "analogue"), ("trace_negative", "negative")):
        selected = next(
            row
            for row in candidates
            if row["reference_id"] == "fictional_alpha"
            and row["intended_identity_group"] == "remote"
            and row["intended_structural_class"] == structure_class
        )
        fixed_examples.append(
            {
                "panel_role": role,
                "reference_id": selected["reference_id"],
                "sequence_sha256": selected["sequence_sha256"],
                "required_structural_class": structure_class,
                "required_identity_regime": "remote",
            }
        )
    write_csv(
        root / "inputs" / "figure_examples.csv",
        [
            "panel_role", "reference_id", "sequence_sha256", "required_structural_class",
            "required_identity_regime",
        ],
        fixed_examples,
    )
    make_representations(root, candidates, references, seed=seed)
    from generate_confidence_example import generate as generate_confidence
    generate_confidence(root)

    allowlist = {
        "schema_version": 1,
        "fictional": True,
        "sequence_sha256": sorted(
            {digest(value) for value in references.values()}
            | {str(row["sequence_sha256"]) for row in candidates}
        ),
    }
    (root / "expected" / "fictional_sequence_allowlist.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "expected" / "fictional_sequence_allowlist.json").write_text(
        json.dumps(allowlist, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": "fictional_landscape_v1",
        "fictional": True,
        "seed": seed,
        "references": len(references),
        "candidates": len(candidates),
        "warning": "All sequences, structures, embeddings and SAE activations are synthetic.",
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("example"))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--hmmbuild", default="hmmbuild")
    args = parser.parse_args()
    generate(args.output.resolve(), seed=args.seed, hmmbuild=args.hmmbuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
