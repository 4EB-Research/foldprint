"""Offline WT-label concordance from complete, externally scored predictions.

No folding, job submission, rescoring, or missing-prediction substitution occurs.
Only explicitly selected aggregate columns are written by the public workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = {"esmfold": 1, "colabfold": 5, "esmfold2": 5}
ENDPOINTS = ("optimized_tm", "fixed_tm")
COHORT_COLUMNS = ["id", "family", "generator", "main", "existing", "selection_bin"]
SCORE_COLUMNS = [
    "id",
    "model",
    "sample_id",
    *ENDPOINTS,
    "prediction_coverage",
    "reference_coverage",
    "normalization_length",
]


def validate_inputs(
    cohort: pd.DataFrame, scores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require exact cohort coverage and all 1+5+5 predictions per design."""
    for frame, columns, name in (
        (cohort, COHORT_COLUMNS, "cohort"),
        (scores, SCORE_COLUMNS, "scores"),
    ):
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"{name} lacks required columns: {sorted(missing)}")
        if frame[columns].isna().any().any():
            raise ValueError(f"{name} contains missing required values")
    c, s = cohort[COHORT_COLUMNS].copy(), scores[SCORE_COLUMNS].copy()
    if c.empty or not c.id.is_unique or c.id.astype(str).str.strip().eq("").any():
        raise ValueError("Cohort must have nonempty, unique design keys")
    for col in ("family", "generator"):
        if c[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty {col}")
    for col in ("main", "existing"):
        if not c[col].isin([True, False, "True", "False"]).all():
            raise ValueError(f"{col} must be an explicit boolean")
        c[col] = c[col].map({True: True, False: False, "True": True, "False": False})
    if not c.main.any():
        raise ValueError("No main-cohort designs")
    bins = pd.to_numeric(c.selection_bin, errors="raise")
    if not bins.isin(range(5)).all():
        raise ValueError("Frozen selection_bin must be 0 through 4")
    c["selection_bin"] = bins.astype(int)
    if set(s.id) != set(c.id) or set(s.model) != set(MODELS):
        raise ValueError("Prediction keys/models must exactly cover the cohort")
    sample = pd.to_numeric(s.sample_id, errors="raise")
    if not np.isfinite(sample).all() or not sample.eq(sample.astype(int)).all():
        raise ValueError("sample_id must be an integer")
    s["sample_id"] = sample.astype(int)
    if s.duplicated(["id", "model", "sample_id"]).any():
        raise ValueError("Duplicate prediction key")
    for model, expected in MODELS.items():
        part = s[s.model.eq(model)]
        if (
            set(part.id) != set(c.id)
            or not part.groupby("id").size().eq(expected).all()
            or not part.sample_id.isin(range(1, expected + 1)).all()
        ):
            raise ValueError(f"Incomplete prediction ensemble: {model}")
    for col in [*ENDPOINTS, "prediction_coverage", "reference_coverage"]:
        s[col] = pd.to_numeric(s[col], errors="raise")
        if not np.isfinite(s[col]).all() or not s[col].between(0, 1).all():
            raise ValueError(f"{col} must be finite and in [0,1]")
    if not s.prediction_coverage.eq(1).all() or not s.reference_coverage.gt(0).all():
        raise ValueError("Full prediction coverage and positive WT coverage are required")
    lengths = pd.to_numeric(s.normalization_length, errors="raise")
    if (
        not np.isfinite(lengths).all()
        or not lengths.ge(3).all()
        or not lengths.eq(lengths.astype(int)).all()
    ):
        raise ValueError("Full-length normalization must be an integer >=3")
    s["normalization_length"] = lengths.astype(int)
    for col in ("normalization_length", "reference_coverage"):
        if not s.groupby("id")[col].nunique().eq(1).all():
            raise ValueError(f"Inconsistent WT {col} within a design")
    # When hashes are supplied, also bind every score to the frozen sequence.
    if "sequence_sha256" in cohort and "sequence_sha256" in scores:
        expected = scores.id.map(cohort.set_index("id").sequence_sha256)
        if not scores.sequence_sha256.eq(expected).all():
            raise ValueError("Score/cohort sequence hashes differ")
    return c, s


def comparisons(cohort: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Private in-memory scatter rows; these are never aggregate release data."""
    c, s = validate_inputs(cohort, scores)
    frames = []
    for endpoint in ENDPOINTS:
        stats = s.groupby(["id", "model"])[endpoint].agg(["median", "min", "max"])
        baseline = stats.xs("esmfold", level="model")["median"]
        for method in ("colabfold", "esmfold2"):
            part = c.set_index("id").copy()
            alt = stats.xs(method, level="model")
            part["x"] = baseline
            part["y"], part["y_min"], part["y_max"] = alt["median"], alt["min"], alt["max"]
            part["endpoint"], part["method"] = endpoint, method
            part["median_flip"] = part.x.ge(0.5).ne(part.y.ge(0.5))
            part["stable_flip"] = (part.x.lt(0.5) & part.y_min.ge(0.5)) | (
                part.x.ge(0.5) & part.y_max.lt(0.5)
            )
            part["crossing"] = part.y_min.lt(0.5) & part.y_max.ge(0.5)
            part["negative_to_analog"] = part.x.lt(0.5) & part.y.ge(0.5)
            part["analog_to_negative"] = part.x.ge(0.5) & part.y.lt(0.5)
            part["absolute_score_change"] = (part.x - part.y).abs()
            frames.append(part.reset_index())
    return pd.concat(frames, ignore_index=True)


def aggregate_results(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize targeted cohorts; each design counted once per comparison."""
    views = []
    for name, mask in (
        ("main", rows.main),
        ("all", np.ones(len(rows), dtype=bool)),
        ("main_shared_bins", rows.main & rows.selection_bin.ge(1)),
    ):
        views.append(rows.loc[mask].assign(view=name))
    data = pd.concat(views, ignore_index=True)
    tables = {}
    for name, keys in (
        ("predictor-summary", ["endpoint", "view", "generator", "method"]),
        ("family-summary", ["endpoint", "view", "generator", "family", "method"]),
    ):
        result = (
            data.groupby(keys, sort=True)
            .agg(
                n=("id", "size"),
                median_flips=("median_flip", "sum"),
                stable_flips=("stable_flip", "sum"),
                crossing=("crossing", "sum"),
                negative_to_analog=("negative_to_analog", "sum"),
                analog_to_negative=("analog_to_negative", "sum"),
                median_absolute_change=("absolute_score_change", "median"),
            )
            .reset_index()
        )
        result["flip_percent"] = 100 * result.median_flips / result.n
        tables[name] = result
    unique = (
        data.groupby(["endpoint", "view", "generator", "id"], sort=True)
        .agg(
            any_median_disagreement=("median_flip", "any"),
            any_within_predictor_crossing=("crossing", "any"),
        )
        .reset_index()
    )
    tables["unique-design-summary"] = (
        unique.groupby(["endpoint", "view", "generator"])
        .agg(
            n=("id", "size"),
            any_median_disagreement=("any_median_disagreement", "sum"),
            any_within_predictor_crossing=("any_within_predictor_crossing", "sum"),
        )
        .reset_index()
    )
    return tables
