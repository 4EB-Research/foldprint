#!/usr/bin/env python3
"""Freeze per-reference Pfam no-hit floors from a complete canonical score table."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from protein_sae_pipeline.io import atomic_csv, atomic_json, sha256_file
from protein_sae_pipeline.workflows import PFAM_CENSOR_FLOOR_COLUMNS


REQUIRED_COLUMNS = {
    "reference_id",
    "sequence_sha256",
    "pfam_full_sequence_bit_score",
    "pfam_best_domain_bit_score",
    "pfam_score_left_censored",
    "pfam_censor_floor",
}


def _booleans(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError("pfam_score_left_censored contains non-boolean values")
    return mapped.astype(bool)


def freeze_floors(source: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Complete Pfam score table lacks columns: {sorted(missing)}")
    if frame.empty or frame.duplicated(["reference_id", "sequence_sha256"]).any():
        raise ValueError("Complete Pfam score table is empty or has duplicate candidate keys")
    frame = frame.copy()
    frame["pfam_score_left_censored"] = _booleans(
        frame["pfam_score_left_censored"]
    )
    rows: list[dict[str, object]] = []
    for reference_id, group in frame.groupby("reference_id", sort=True, observed=True):
        full = pd.to_numeric(group["pfam_full_sequence_bit_score"], errors="raise")
        domain = pd.to_numeric(group["pfam_best_domain_bit_score"], errors="raise")
        declared = pd.to_numeric(group["pfam_censor_floor"], errors="raise")
        censored = group["pfam_score_left_censored"]
        finite = ~censored
        if not finite.any():
            raise ValueError(f"Reference {reference_id} has no finite Pfam hit")
        if not full.map(math.isfinite).all() or not domain.map(math.isfinite).all():
            raise ValueError(f"Reference {reference_id} has non-finite Pfam bit scores")
        if declared.nunique(dropna=False) != 1 or not math.isfinite(float(declared.iloc[0])):
            raise ValueError(f"Reference {reference_id} lacks one finite declared censor floor")
        full_floor = float(full.loc[finite].min() - 1.0)
        domain_floor = float(domain.loc[finite].min() - 1.0)
        if float(declared.iloc[0]) != full_floor:
            raise ValueError(
                f"Reference {reference_id} declared full-sequence floor differs from min-1"
            )
        if censored.any():
            if not full.loc[censored].eq(full_floor).all():
                raise ValueError(f"Reference {reference_id} has inconsistent censored full scores")
            if not domain.loc[censored].eq(domain_floor).all():
                raise ValueError(f"Reference {reference_id} has inconsistent censored domain scores")
        rows.append(
            {
                "reference_id": str(reference_id),
                "pfam_full_sequence_censor_floor": full_floor,
                "pfam_best_domain_censor_floor": domain_floor,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-score-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = freeze_floors(args.full_score_table)
    atomic_csv(args.output, rows, fieldnames=PFAM_CENSOR_FLOOR_COLUMNS)
    atomic_json(
        args.output.with_suffix(args.output.suffix + ".json"),
        {
            "schema_version": 1,
            "source_score_table_sha256": sha256_file(args.full_score_table),
            "rows": len(rows),
            "floor_definition": "per_reference_complete_landscape_minimum_finite_minus_one_bit",
            "output_sha256": sha256_file(args.output),
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
