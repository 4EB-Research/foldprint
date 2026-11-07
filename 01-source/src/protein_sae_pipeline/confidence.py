"""Confidence controls from the paper's threshold-balanced matching analysis.

The one-dimensional dynamic program is the algorithm in mutavault's
threshold_balanced_matched/sampling.py, vectorized to avoid a Numba dependency.
Selection never uses a retrieval score or a generator label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEYS = ["reference_id", "sequence_sha256"]


def match_all_negatives(negative, positive) -> tuple[np.ndarray, float]:
    """Minimum total absolute distance, with one distinct positive per negative.

    Both arrays must be sorted. Equal-cost paths prefer earlier positives,
    matching the original strict ``take < skip`` recurrence and traceback.
    Memory is O(n*m) bytes for traceback and O(m) floating-point working space.
    """
    negative = np.asarray(negative, dtype=float)
    positive = np.asarray(positive, dtype=float)
    if negative.ndim != 1 or positive.ndim != 1:
        raise ValueError("Matching inputs must be one-dimensional")
    n, m = len(negative), len(positive)
    if n > m:
        raise ValueError("Cannot retain all negatives: fewer positives than negatives")
    if not np.isfinite(negative).all() or not np.isfinite(positive).all():
        raise ValueError("Matching values must be finite")
    if (np.diff(negative) < 0).any() or (np.diff(positive) < 0).any():
        raise ValueError("Matching inputs must already be sorted")
    choices = np.zeros((n + 1, m + 1), dtype=bool)
    previous = np.zeros(m + 1)
    for i in range(1, n + 1):
        take = previous[i - 1:m] + np.abs(negative[i - 1] - positive[i - 1:m])
        best = np.minimum.accumulate(take)
        choices[i, i:] = take < np.r_[np.inf, best[:-1]]
        current = np.full(m + 1, np.inf)
        current[i:] = best
        previous = current
    indices = np.empty(n, dtype=np.int64)
    i, j = n, m
    while i:
        if choices[i, j]:
            indices[i - 1] = j - 1
            i -= 1
        j -= 1
    return indices, float(previous[m])


@dataclass(frozen=True)
class MatchedCohort:
    cohort: pd.DataFrame
    pairs: pd.DataFrame
    diagnostics: pd.DataFrame


def match_ptm(remote: pd.DataFrame, *, cutoff: float = 0.5) -> MatchedCohort:
    """Retain all pTM-passing negatives, then match within target without reuse.

    There is no maximum pair distance. Generator identities can differ.
    All distances and generator combinations are retained for auditing.
    """
    required = {*KEYS, "predicted_ptm", "structural_class", "generation_method"}
    if required - set(remote):
        raise ValueError(f"Matching lacks columns: {sorted(required - set(remote))}")
    if remote[KEYS].isna().any(axis=None) or remote.duplicated(KEYS).any():
        raise ValueError("Matching requires unique nonmissing candidate keys")
    if not 0 <= cutoff <= 1:
        raise ValueError("pTM cutoff must be in [0,1]")
    ptm = pd.to_numeric(remote.predicted_ptm, errors="raise")
    if not np.isfinite(ptm).all() or not ptm.between(0, 1).all():
        raise ValueError("pTM must be finite and in [0,1]")
    if not set(remote.structural_class) <= {"analogue", "negative"}:
        raise ValueError("Unknown structural class")
    pool = remote.loc[ptm.ge(cutoff)].copy()
    pieces, pairs, diagnostics = [], [], []
    for reference in sorted(remote.reference_id.unique()):
        target = pool.loc[pool.reference_id.eq(reference)]
        order = ["predicted_ptm", "sequence_sha256"]
        negative = target.loc[target.structural_class.eq("negative")].sort_values(order)
        available = target.loc[target.structural_class.eq("analogue")].sort_values(order)
        if negative.empty:
            raise ValueError(f"{reference}: no pTM-passing negatives for retrieval evaluation")
        selected, cost = match_all_negatives(negative.predicted_ptm, available.predicted_ptm)
        positive = available.iloc[selected].copy().reset_index(drop=True)
        negative = negative.copy().reset_index(drop=True)
        ids = [f"ptm_ge_050:{reference}:{i:05d}" for i in range(len(negative))]
        positive["pair_id"], negative["pair_id"] = ids, ids
        distance = np.abs(positive.predicted_ptm.to_numpy() - negative.predicted_ptm.to_numpy())
        for i, pair_id in enumerate(ids):
            pairs.append({
                "reference_id": reference, "pair_id": pair_id,
                "positive_sequence_sha256": positive.loc[i, "sequence_sha256"],
                "negative_sequence_sha256": negative.loc[i, "sequence_sha256"],
                "positive_predicted_ptm": positive.loc[i, "predicted_ptm"],
                "negative_predicted_ptm": negative.loc[i, "predicted_ptm"],
                "positive_generation_method": positive.loc[i, "generation_method"],
                "negative_generation_method": negative.loc[i, "generation_method"],
                "matching_absolute_difference": distance[i],
            })
        diagnostics.append({
            "reference_id": reference, "available_positives": len(available),
            "available_negatives": len(negative), "pairs": len(negative),
            "total_absolute_difference": cost,
            "mean_absolute_difference": float(distance.mean()),
            "median_absolute_difference": float(np.median(distance)),
            "maximum_absolute_difference": float(distance.max()),
            "negative_retention_fraction": 1.0,
            "same_generator_fraction": float(positive.generation_method.eq(negative.generation_method).mean()),
        })
        pieces.extend([positive, negative])
    cohort = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return MatchedCohort(cohort, pd.DataFrame(pairs), pd.DataFrame(diagnostics))
