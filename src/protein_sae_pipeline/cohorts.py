"""Cohort construction from the complete master table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .io import atomic_csv, atomic_json


REQUIRED_COLUMNS = {
    "dataset_id",
    "reference_id",
    "sequence_sha256",
    "is_remote",
    "is_structural_analogue",
    "structural_class",
}


def construct_cohorts(master: pd.DataFrame, config: PipelineConfig) -> dict[str, pd.DataFrame]:
    missing = REQUIRED_COLUMNS - set(master.columns)
    if missing:
        raise ValueError(f"Master table lacks cohort columns: {sorted(missing)}")
    if master[["reference_id", "sequence_sha256"]].duplicated().any():
        raise ValueError("Master table has duplicate candidate keys")
    if set(master["dataset_id"].astype(str)) != {config.dataset_id}:
        raise ValueError("Master table dataset_id differs from configuration")
    reference_id = str(config.raw["interpretability"]["reference_id"])
    full = master.copy()
    focal_full = full.loc[full["reference_id"].eq(reference_id)].copy()
    remote = full.loc[full["is_remote"].astype(bool)].copy()
    focal_remote = remote.loc[remote["reference_id"].eq(reference_id)].copy()
    cohorts = {
        "full_landscape": full,
        "focal_reference_full": focal_full,
        "focal_reference_remote": focal_remote,
        "all_references_remote": remote,
    }
    audit = cohort_audit(cohorts, config)
    if not audit["complete"]:
        raise ValueError("Cohort counts differ from the frozen configuration: " + "; ".join(audit["errors"]))
    return cohorts


def cohort_audit(cohorts: dict[str, pd.DataFrame], config: PipelineConfig) -> dict[str, Any]:
    expected = config.expected_counts
    full = cohorts["full_landscape"]
    remote = cohorts["all_references_remote"]
    errors: list[str] = []

    observed_global = {
        "full_total": int(len(full)),
        "remote_total": int(len(remote)),
        "remote_analogues": int(remote["is_structural_analogue"].astype(bool).sum()),
        "remote_negatives": int((~remote["is_structural_analogue"].astype(bool)).sum()),
    }
    for field, observed in observed_global.items():
        if field in expected and observed != int(expected[field]):
            errors.append(f"{field}: observed {observed}, expected {int(expected[field])}")

    observed_by_reference: dict[str, dict[str, int]] = {}
    for reference_id in sorted(config.raw["references"]):
        target_full = full.loc[full["reference_id"].eq(reference_id)]
        target_remote = remote.loc[remote["reference_id"].eq(reference_id)]
        row = {
            "full": int(len(target_full)),
            "remote_analogues": int(target_remote["is_structural_analogue"].astype(bool).sum()),
            "remote_negatives": int((~target_remote["is_structural_analogue"].astype(bool)).sum()),
            "remote_total": int(len(target_remote)),
        }
        observed_by_reference[reference_id] = row
        expected_row = expected.get("by_reference", {}).get(reference_id)
        if expected_row is not None:
            for field, observed in row.items():
                if observed != int(expected_row[field]):
                    errors.append(
                        f"{reference_id}.{field}: observed {observed}, expected {int(expected_row[field])}"
                    )

    return {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "rules": {
            "remote": f"positional_identity_fraction < {config.identity_threshold}",
            "structural_analogue": f"usalign_tm_score_wt_norm >= {config.tm_threshold}",
        },
        "complete": not errors,
        "errors": errors,
        "global": observed_global,
        "by_reference": observed_by_reference,
    }


def write_cohorts(
    cohorts: dict[str, pd.DataFrame], output_root: Path, config: PipelineConfig
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in cohorts.items():
        atomic_csv(
            output_root / f"{name}.csv.gz",
            frame.to_dict("records"),
            fieldnames=list(frame.columns),
        )
    audit = cohort_audit(cohorts, config)
    atomic_json(output_root / "cohort_audit.json", audit)
    return audit
