"""Finite-sample retrieval operating points and empirical ROC curves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OperatingPoint:
    target_fpr: float
    threshold: float
    allowed_false_positives: int
    observed_false_positives: int
    control_count: int
    observed_fpr: float
    true_positives: int
    case_count: int
    recall: float


def empirical_operating_point(
    case_scores: Iterable[float],
    control_scores: Iterable[float],
    target_fpr: float,
) -> OperatingPoint:
    """Calibrate a conservative finite-sample threshold from controls.

    Scores are oriented so larger is more WT-like. With ``k=floor(q*N)``, the
    threshold is the next-ranked control and positives require strict ``>``.
    This makes ties conservative and guarantees no more than ``k`` false calls.
    """

    cases = np.asarray(list(case_scores), dtype=np.float64)
    controls = np.asarray(list(control_scores), dtype=np.float64)
    if cases.ndim != 1 or controls.ndim != 1 or not len(cases) or not len(controls):
        raise ValueError("Cases and controls must be nonempty one-dimensional score arrays")
    if not np.isfinite(cases).all() or not np.isfinite(controls).all():
        raise ValueError("Scores must be finite")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0,1)")
    allowed = int(math.floor(target_fpr * len(controls)))
    if allowed >= len(controls):
        raise ValueError("target_fpr leaves no next-ranked control threshold")
    sorted_controls = np.sort(controls)[::-1]
    threshold = float(sorted_controls[allowed])
    false_positives = int(np.count_nonzero(controls > threshold))
    true_positives = int(np.count_nonzero(cases > threshold))
    if false_positives > allowed:
        raise RuntimeError("Tie-conservative operating point exceeded its false-positive budget")
    return OperatingPoint(
        float(target_fpr),
        threshold,
        allowed,
        false_positives,
        int(len(controls)),
        false_positives / len(controls),
        true_positives,
        int(len(cases)),
        true_positives / len(cases),
    )


def empirical_operating_points(
    case_scores: Iterable[float],
    control_scores: Iterable[float],
    target_fprs: Iterable[float],
) -> tuple[OperatingPoint, ...]:
    """Calculate several finite-sample endpoints with one control ordering.

    This is exactly equivalent to repeated :func:`empirical_operating_point`
    calls, including strict tie handling, but avoids sorting the same controls
    once per ROC grid point or endpoint.
    """

    cases = np.asarray(list(case_scores), dtype=np.float64)
    controls = np.asarray(list(control_scores), dtype=np.float64)
    budgets = np.asarray(list(target_fprs), dtype=np.float64)
    if cases.ndim != 1 or controls.ndim != 1 or not len(cases) or not len(controls):
        raise ValueError("Cases and controls must be nonempty one-dimensional score arrays")
    if budgets.ndim != 1 or not len(budgets) or np.any((budgets < 0) | (budgets >= 1)):
        raise ValueError("target_fprs must be a nonempty vector in [0,1)")
    if not np.isfinite(cases).all() or not np.isfinite(controls).all():
        raise ValueError("Scores must be finite")
    allowed = np.floor(budgets * len(controls)).astype(np.int64)
    if np.any(allowed >= len(controls)):
        raise ValueError("a target FPR leaves no next-ranked control threshold")
    controls_ascending = np.sort(controls)
    cases_ascending = np.sort(cases)
    thresholds = controls_ascending[len(controls) - 1 - allowed]
    false_positives = len(controls) - np.searchsorted(
        controls_ascending, thresholds, side="right"
    )
    true_positives = len(cases) - np.searchsorted(cases_ascending, thresholds, side="right")
    if np.any(false_positives > allowed):
        raise RuntimeError("Tie-conservative operating point exceeded its false-positive budget")
    return tuple(
        OperatingPoint(
            float(target_fpr),
            float(threshold),
            int(allowed_count),
            int(false_positive_count),
            int(len(controls)),
            float(false_positive_count / len(controls)),
            int(true_positive_count),
            int(len(cases)),
            float(true_positive_count / len(cases)),
        )
        for target_fpr, threshold, allowed_count, false_positive_count, true_positive_count in zip(
            budgets,
            thresholds,
            allowed,
            false_positives,
            true_positives,
            strict=True,
        )
    )


def operating_points_by_reference(
    cohort: pd.DataFrame,
    score_columns: dict[str, str],
    endpoints: Iterable[float] = (0.01, 0.05),
) -> pd.DataFrame:
    required = {"reference_id", "structural_class", *score_columns.values()}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"Cohort lacks retrieval columns: {sorted(missing)}")
    endpoints_tuple = tuple(float(value) for value in endpoints)
    rows: list[dict[str, object]] = []
    for reference_id, target in cohort.groupby("reference_id", sort=True, observed=True):
        cases = target["structural_class"].eq("analogue")
        controls = target["structural_class"].eq("negative")
        if not cases.any() or not controls.any():
            raise ValueError(f"Reference {reference_id} lacks a case or control class")
        for method, column in score_columns.items():
            points = empirical_operating_points(
                target.loc[cases, column], target.loc[controls, column], endpoints_tuple
            )
            for point in points:
                rows.append(
                    {
                        "reference_id": reference_id,
                        "method": method,
                        "score_column": column,
                        **asdict(point),
                    }
                )
    return pd.DataFrame(rows)


def empirical_roc_grid(
    case_scores: Iterable[float],
    control_scores: Iterable[float],
    *,
    max_fpr: float = 0.05,
    grid_points: int = 101,
) -> pd.DataFrame:
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    budgets = np.linspace(0.0, float(max_fpr), int(grid_points))
    return pd.DataFrame(
        [asdict(point) for point in empirical_operating_points(case_scores, control_scores, budgets)]
    )


def roc_by_reference(
    cohort: pd.DataFrame,
    score_columns: dict[str, str],
    *,
    max_fpr: float = 0.05,
    grid_points: int = 101,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for reference_id, target in cohort.groupby("reference_id", sort=True, observed=True):
        case_mask = target["structural_class"].eq("analogue")
        control_mask = target["structural_class"].eq("negative")
        for method, column in score_columns.items():
            curve = empirical_roc_grid(
                target.loc[case_mask, column],
                target.loc[control_mask, column],
                max_fpr=max_fpr,
                grid_points=grid_points,
            )
            curve.insert(0, "score_column", column)
            curve.insert(0, "method", method)
            curve.insert(0, "reference_id", reference_id)
            rows.append(curve)
    return pd.concat(rows, ignore_index=True)
