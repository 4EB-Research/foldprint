"""Exact-coverage assembly of the canonical landscape score table."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import sha256_file
from .keys import SHA256_PATTERN
from .schemas import ValidatedInputs


KEY_COLUMNS = ["reference_id", "sequence_sha256"]
IDENTITY_COLUMNS = {
    "positional_matches",
    "positional_mismatches",
    "positional_identity_fraction",
    "positional_identity_percent",
}
STRUCTURE_COLUMNS = {
    "candidate_coordinate_length",
    "wt_coordinate_length",
    "usalign_tm_score_candidate_norm",
    "usalign_tm_score_wt_norm",
    "usalign_aligned_length",
    "usalign_rmsd_angstrom",
    "usalign_aligned_sequence_identity",
}
PFAM_COLUMNS = {
    "pfam_accession",
    "pfam_full_sequence_evalue",
    "pfam_full_sequence_bit_score",
    "pfam_best_domain_evalue",
    "pfam_best_domain_bit_score",
    "pfam_score_left_censored",
    "pfam_censor_floor",
}
MMSEQS_COLUMNS = {
    "mmseqs2_bit_score",
    "mmseqs2_evalue",
    "mmseqs2_percent_identity",
    "mmseqs2_alignment_length",
    "mmseqs2_query_coverage",
    "mmseqs2_target_coverage",
    "mmseqs2_hit",
}
REPRESENTATION_COLUMNS = {
    "esmc_cosine_to_wt",
    "esmc_zero_norm",
    "sae_cosine_to_wt",
    "sae_profile_norm",
    "sae_profile_nnz",
    "sae_zero_norm",
}


def manifest_frame(inputs: ValidatedInputs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "manifest_row": candidate.manifest_row,
                "sequence_id": candidate.sequence_id,
                "reference_id": candidate.reference_id,
                "generation_method": candidate.generation_method,
                "sequence": candidate.sequence,
                "sequence_sha256": candidate.sequence_sha256,
                "sequence_length": candidate.sequence_length,
            }
            for candidate in inputs.candidates
        ]
    )


def validate_master_against_inputs(
    master: pd.DataFrame,
    inputs: ValidatedInputs,
    config: PipelineConfig,
) -> None:
    """Bind a downstream master table to the validated candidate manifest.

    This check is repeated at figure and cohort entry points to prevent a
    same-sized score table from being combined with the wrong inputs.
    """

    manifest = manifest_frame(inputs)
    required = {
        "dataset_id",
        "manifest_row",
        "sequence_id",
        "reference_id",
        "generation_method",
        "sequence",
        "sequence_sha256",
        "sequence_length",
    }
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master table lacks manifest-binding columns: {sorted(missing)}")
    if len(master) != len(manifest):
        raise ValueError(
            f"Master row count differs from validated inputs: {len(master)} != {len(manifest)}"
        )
    if set(master["dataset_id"].astype(str)) != {config.dataset_id}:
        raise ValueError("Master dataset_id differs from the validated configuration")
    if master[KEY_COLUMNS].duplicated().any():
        raise ValueError("Master table contains duplicate candidate keys")
    expected_keys = set(map(tuple, manifest[KEY_COLUMNS].itertuples(index=False, name=None)))
    observed_keys = set(map(tuple, master[KEY_COLUMNS].itertuples(index=False, name=None)))
    if observed_keys != expected_keys:
        raise ValueError(
            "Master candidate keys differ from validated inputs: "
            f"missing={len(expected_keys - observed_keys)}, "
            f"unexpected={len(observed_keys - expected_keys)}"
        )
    binding_columns = [
        "manifest_row",
        "sequence_id",
        "generation_method",
        "sequence",
        "sequence_length",
    ]
    observed = master[KEY_COLUMNS + binding_columns].merge(
        manifest[KEY_COLUMNS + binding_columns],
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        suffixes=("_master", "_input"),
        indicator=True,
    )
    if not observed["_merge"].eq("both").all():
        raise ValueError("Master/manifest binding merge is incomplete")
    mismatches = [
        column
        for column in binding_columns
        if not observed[f"{column}_master"].eq(observed[f"{column}_input"]).all()
    ]
    if mismatches:
        raise ValueError(
            "Master manifest metadata differs from validated inputs: "
            + ", ".join(mismatches)
        )


def _validate_score_sidecar(
    path: Path,
    *,
    label: str,
    expected_dataset_id: str,
    observed_rows: int,
    config: PipelineConfig,
) -> None:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"{label} score metadata is missing: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"{label} score metadata has an unsupported schema")
    if payload.get("dataset_id") != expected_dataset_id:
        raise ValueError(f"{label} score metadata belongs to a different dataset")
    if payload.get("complete") is not True or int(payload.get("rows", -1)) != observed_rows:
        raise ValueError(f"{label} score metadata does not declare exact completion")
    if payload.get("score_table_sha256") != sha256_file(path):
        raise ValueError(f"{label} score-table checksum differs from its metadata")

    if label == "structure":
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("structure score metadata lacks parameters")
        expected = config.native_tool("usalign")
        required = {
            "version": str(expected["version"]),
            "source_commit": str(expected["source_commit"]),
            "arguments": list(expected["arguments"]),
            "structure_1": "candidate",
            "structure_2": "wild_type",
            "primary_normalization": str(expected["primary_normalization"]),
            "fixed_position_flag_used": False,
        }
        if any(parameters.get(key) != value for key, value in required.items()):
            raise ValueError("structure score metadata differs from the frozen US-align protocol")
    elif label == "pfam":
        parameters = payload.get("parameters")
        expected = config.native_tool("hmmer")
        if not isinstance(parameters, dict) or any(
            parameters.get(key) != value
            for key, value in {
                "program": "hmmsearch",
                "version": str(expected["version"]),
                "arguments": list(expected["arguments"]),
                "score": expected["score"],
                "no_hit": expected["no_hit"],
            }.items()
        ):
            raise ValueError("Pfam score metadata differs from the frozen hmmsearch protocol")
        profile_contracts = parameters.get("profile_contracts")
        expected_accessions = {
            reference_id: str(reference["pfam_accession"])
            for reference_id, reference in sorted(config.raw["references"].items())
        }
        if not isinstance(profile_contracts, dict) or set(profile_contracts) != set(
            expected_accessions
        ):
            raise ValueError("Pfam score metadata differs from the frozen profile contracts")
        for reference_id, expected_accession in expected_accessions.items():
            contract = profile_contracts[reference_id]
            if (
                not isinstance(contract, dict)
                or contract.get("accession") != expected_accession
                or SHA256_PATTERN.fullmatch(str(contract.get("sha256", ""))) is None
            ):
                raise ValueError("Pfam score metadata differs from the frozen profile contracts")
        floor_source = parameters.get("censor_floor_source")
        floor_table_sha256 = parameters.get("censor_floor_table_sha256")
        if floor_source == "complete_input_landscape":
            if floor_table_sha256 is not None:
                raise ValueError("Complete-landscape Pfam metadata claims an external floor table")
        elif floor_source == "frozen_full_landscape_table":
            if not isinstance(floor_table_sha256, str) or SHA256_PATTERN.fullmatch(
                floor_table_sha256
            ) is None:
                raise ValueError("Frozen Pfam floor-table metadata lacks its SHA-256")
        else:
            raise ValueError("Pfam score metadata has an unknown censor-floor source")
    elif label == "mmseqs":
        parameters = payload.get("parameters")
        expected = config.native_tool("mmseqs2")
        if not isinstance(parameters, dict) or any(
            parameters.get(key) != expected[key]
            for key in ("release", "commit", "evalue", "sensitivity", "max_sequences", "format")
        ):
            raise ValueError("MMseqs2 score metadata differs from the frozen protocol")
    elif label == "representations":
        if payload.get("esmc") != config.model("esmc") or payload.get("sae") != config.model("sae"):
            raise ValueError("representation metadata differs from the frozen model configuration")


def _read_score_table(
    path_or_frame: str | Path | pd.DataFrame,
    label: str,
    *,
    config: PipelineConfig,
) -> pd.DataFrame:
    if isinstance(path_or_frame, pd.DataFrame):
        frame = path_or_frame.copy()
    else:
        path = Path(path_or_frame)
        frame = pd.read_csv(path, low_memory=False)
        _validate_score_sidecar(
            path,
            label=label,
            expected_dataset_id=config.dataset_id,
            observed_rows=len(frame),
            config=config,
        )
    missing_key = set(KEY_COLUMNS) - set(frame.columns)
    if missing_key:
        raise ValueError(f"{label} lacks key columns: {sorted(missing_key)}")
    if frame.duplicated(KEY_COLUMNS).any():
        duplicates = frame.loc[frame.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"{label} has duplicate candidate keys:\n{duplicates}")
    if not frame["sequence_sha256"].map(
        lambda value: isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None
    ).all():
        raise ValueError(f"{label} has malformed sequence hashes")
    return frame


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[KEY_COLUMNS])


def _boolean_series(values: pd.Series, label: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label} must contain only true/false values")
    return normalized.eq("true")


def assemble_master(
    inputs: ValidatedInputs,
    config: PipelineConfig,
    *,
    identity: str | Path | pd.DataFrame,
    structure: str | Path | pd.DataFrame,
    pfam: str | Path | pd.DataFrame,
    mmseqs: str | Path | pd.DataFrame,
    representations: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all score tables only after proving exact candidate coverage."""

    manifest = manifest_frame(inputs)
    expected_index = _key_index(manifest)
    if expected_index.has_duplicates:
        raise ValueError("Validated manifest unexpectedly contains duplicate keys")
    tables: Sequence[tuple[str, str | Path | pd.DataFrame, set[str]]] = (
        ("identity", identity, IDENTITY_COLUMNS),
        ("structure", structure, STRUCTURE_COLUMNS),
        ("pfam", pfam, PFAM_COLUMNS),
        ("mmseqs", mmseqs, MMSEQS_COLUMNS),
        ("representations", representations, REPRESENTATION_COLUMNS),
    )
    loaded: dict[str, pd.DataFrame] = {}
    expected_set = set(expected_index.tolist())
    for label, source, required in tables:
        frame = _read_score_table(source, label, config=config)
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{label} lacks score columns: {sorted(missing_columns)}")
        observed_set = set(_key_index(frame).tolist())
        missing = expected_set - observed_set
        unexpected = observed_set - expected_set
        if missing or unexpected:
            raise ValueError(
                f"{label} key coverage differs: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        extras = [column for column in frame.columns if column not in KEY_COLUMNS]
        collisions = (set(extras) & set(manifest.columns)) - set(KEY_COLUMNS)
        if collisions:
            raise ValueError(f"{label} attempts to replace manifest columns: {sorted(collisions)}")
        loaded[label] = frame[KEY_COLUMNS + extras]

    result = manifest
    for label, _source, _required in tables:
        result = result.merge(
            loaded[label], on=KEY_COLUMNS, how="left", validate="one_to_one", sort=False
        )
    if len(result) != len(manifest) or _key_index(result).has_duplicates:
        raise RuntimeError("Master assembly changed row count or key uniqueness")

    identity_fraction = pd.to_numeric(result["positional_identity_fraction"], errors="raise")
    identity_percent = pd.to_numeric(result["positional_identity_percent"], errors="raise")
    if not identity_fraction.between(0, 1, inclusive="both").all():
        raise ValueError("positional identity lies outside [0,1]")
    if not np.allclose(identity_percent, 100.0 * identity_fraction, atol=1e-12, rtol=0):
        raise ValueError("identity percent and fraction disagree")
    if not (
        pd.to_numeric(result["positional_matches"], errors="raise")
        + pd.to_numeric(result["positional_mismatches"], errors="raise")
        == result["sequence_length"]
    ).all():
        raise ValueError("positional match/mismatch counts do not cover each sequence")

    for column in ("usalign_tm_score_candidate_norm", "usalign_tm_score_wt_norm"):
        if not pd.to_numeric(result[column], errors="raise").between(0, 1, inclusive="both").all():
            raise ValueError(f"{column} lies outside [0,1]")
    candidate_lengths = pd.to_numeric(result["candidate_coordinate_length"], errors="raise")
    wt_lengths = pd.to_numeric(result["wt_coordinate_length"], errors="raise")
    aligned_lengths = pd.to_numeric(result["usalign_aligned_length"], errors="raise")
    if not candidate_lengths.eq(result["sequence_length"]).all():
        raise ValueError("candidate coordinate lengths differ from complete candidate sequences")
    expected_wt_lengths = {
        reference_id: sum(position.observed for position in positions)
        for reference_id, positions in inputs.reference_positions.items()
    }
    if not wt_lengths.eq(result["reference_id"].map(expected_wt_lengths)).all():
        raise ValueError("WT coordinate lengths differ from reference-position maps")
    if not (
        aligned_lengths.gt(0)
        & aligned_lengths.le(np.minimum(candidate_lengths, wt_lengths))
    ).all():
        raise ValueError("US-align aligned lengths are invalid")
    if not pd.to_numeric(result["usalign_rmsd_angstrom"], errors="raise").ge(0).all():
        raise ValueError("US-align RMSD must be nonnegative")
    if not pd.to_numeric(result["usalign_aligned_sequence_identity"], errors="raise").between(
        0, 1, inclusive="both"
    ).all():
        raise ValueError("US-align aligned sequence identity lies outside [0,1]")
    if not pd.to_numeric(result["esmc_cosine_to_wt"], errors="raise").between(
        -1, 1, inclusive="both"
    ).all():
        raise ValueError("ESM-C cosine lies outside [-1,1]")
    if not pd.to_numeric(result["sae_cosine_to_wt"], errors="raise").between(
        0, 1, inclusive="both"
    ).all():
        raise ValueError("SAE cosine lies outside [0,1]")
    if not pd.to_numeric(result["pfam_full_sequence_bit_score"], errors="raise").map(
        np.isfinite
    ).all():
        raise ValueError("Pfam bit scores must be finite after censor-floor assignment")
    pfam_censored = _boolean_series(result["pfam_score_left_censored"], "Pfam censor flag")
    result["pfam_score_left_censored"] = pfam_censored
    pfam_full = pd.to_numeric(result["pfam_full_sequence_bit_score"], errors="raise")
    pfam_floor = pd.to_numeric(result["pfam_censor_floor"], errors="raise")
    if not pfam_floor.map(np.isfinite).all():
        raise ValueError("Pfam censor floors must be finite")
    if not result.groupby("reference_id", observed=True)["pfam_censor_floor"].nunique().eq(1).all():
        raise ValueError("Pfam censor floor must be target-wide, not shard-specific")
    if not np.allclose(pfam_full.loc[pfam_censored], pfam_floor.loc[pfam_censored], rtol=0, atol=0):
        raise ValueError("Pfam no-hit scores must equal the frozen target censor floor")
    if not (pfam_full.loc[~pfam_censored] > pfam_floor.loc[~pfam_censored]).all():
        raise ValueError("Finite Pfam hits must lie above the target censor floor")
    full_evalue = pd.to_numeric(result["pfam_full_sequence_evalue"], errors="raise")
    best_evalue = pd.to_numeric(result["pfam_best_domain_evalue"], errors="raise")
    if not (np.isinf(full_evalue.loc[pfam_censored]) & np.isinf(best_evalue.loc[pfam_censored])).all():
        raise ValueError("Pfam no-hit E-values must be infinite")
    if not (
        full_evalue.loc[~pfam_censored].map(np.isfinite)
        & best_evalue.loc[~pfam_censored].map(np.isfinite)
    ).all():
        raise ValueError("Finite Pfam hits must have finite E-values")
    if not pd.to_numeric(result["mmseqs2_bit_score"], errors="raise").ge(0).all():
        raise ValueError("MMseqs2 bit scores must be nonnegative")
    mmseqs_hit = _boolean_series(result["mmseqs2_hit"], "MMseqs2 hit flag")
    result["mmseqs2_hit"] = mmseqs_hit
    mmseqs_bits = pd.to_numeric(result["mmseqs2_bit_score"], errors="raise")
    mmseqs_fields = [
        "mmseqs2_evalue",
        "mmseqs2_percent_identity",
        "mmseqs2_alignment_length",
        "mmseqs2_query_coverage",
        "mmseqs2_target_coverage",
    ]
    if not mmseqs_bits.loc[~mmseqs_hit].eq(0).all() or not result.loc[
        ~mmseqs_hit, mmseqs_fields
    ].isna().all(axis=None):
        raise ValueError("MMseqs2 no-hit rows must use bit score zero and empty alignment fields")
    if result.loc[mmseqs_hit, mmseqs_fields].isna().any(axis=None):
        raise ValueError("MMseqs2 hit rows have missing alignment fields")
    if not pd.to_numeric(result.loc[mmseqs_hit, "mmseqs2_percent_identity"], errors="raise").between(
        0, 100, inclusive="both"
    ).all():
        raise ValueError("MMseqs2 percent identity lies outside [0,100]")
    for column in ("mmseqs2_query_coverage", "mmseqs2_target_coverage"):
        if not pd.to_numeric(result.loc[mmseqs_hit, column], errors="raise").between(
            0, 1, inclusive="both"
        ).all():
            raise ValueError(f"{column} lies outside [0,1]")

    for flag in ("esmc_zero_norm", "sae_zero_norm"):
        result[flag] = _boolean_series(result[flag], flag)
    if not pd.to_numeric(result["sae_profile_norm"], errors="raise").ge(0).all():
        raise ValueError("SAE profile norms must be nonnegative")
    if not pd.to_numeric(result["sae_profile_nnz"], errors="raise").ge(0).all():
        raise ValueError("SAE profile nonzero counts must be nonnegative")
    if not result.loc[result["esmc_zero_norm"], "esmc_cosine_to_wt"].eq(0).all():
        raise ValueError("zero-norm ESM-C comparisons must have cosine zero")
    if not result.loc[result["sae_zero_norm"], "sae_cosine_to_wt"].eq(0).all():
        raise ValueError("zero-norm SAE comparisons must have cosine zero")

    result["is_remote"] = identity_fraction.lt(config.identity_threshold)
    result["is_structural_analogue"] = pd.to_numeric(
        result["usalign_tm_score_wt_norm"], errors="raise"
    ).ge(config.tm_threshold)
    result["structural_class"] = np.where(
        result["is_structural_analogue"], "analogue", "negative"
    )
    result.insert(0, "dataset_id", config.dataset_id)
    return result.sort_values("manifest_row", kind="stable").reset_index(drop=True)


def score_table_coverage_report(
    master: pd.DataFrame,
    config: PipelineConfig,
    sources: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "complete": True,
        "rows": int(len(master)),
        "unique_candidate_keys": int(master[KEY_COLUMNS].drop_duplicates().shape[0]),
        "source_tables": dict(sorted(sources.items())),
        "score_columns_complete": {
            column: int(master[column].notna().sum())
            for column in sorted(
                IDENTITY_COLUMNS
                | STRUCTURE_COLUMNS
                | PFAM_COLUMNS
                | MMSEQS_COLUMNS
                | REPRESENTATION_COLUMNS
            )
        },
    }
