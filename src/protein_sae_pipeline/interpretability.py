"""Deterministic sparse-feature analyses used in Figures 1--2.

All activity predicates are strict ``activation > threshold`` comparisons; the
the configured setting is ``threshold=0``. Positions exposed by this module are
one-based to match sequences, structures, and figure labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .representations.sparse import (
    SparseResidueMatrix,
    SparseVector,
    sparse_cosine,
    sparse_dot,
)


@dataclass(frozen=True)
class FeatureRanking:
    feature_ids: np.ndarray
    scores: np.ndarray
    ranks: np.ndarray


@dataclass(frozen=True)
class RecurrenceResult:
    feature_ids: np.ndarray
    active_counts: np.ndarray
    total: int
    recurrence: np.ndarray
    wilson_low: np.ndarray
    wilson_high: np.ndarray


@dataclass(frozen=True)
class ContributionSpectrum:
    feature_ids: np.ndarray
    wt_activations: np.ndarray
    candidate_activations: np.ndarray
    dot_products: np.ndarray
    cosine_contributions: np.ndarray
    cumulative_cosine_contributions: np.ndarray
    numerator_shares: np.ndarray
    cumulative_numerator_shares: np.ndarray
    cosine_similarity: float
    reconstructed_cosine: float


@dataclass(frozen=True)
class CohortContributionMatrix:
    feature_ids: np.ndarray
    cosine_contributions: np.ndarray
    cumulative_cosine_contributions: np.ndarray
    numerator_shares: np.ndarray
    cumulative_numerator_shares: np.ndarray
    cosine_similarities: np.ndarray


@dataclass(frozen=True)
class PositionUniqueResult:
    positions: np.ndarray
    counts: np.ndarray
    feature_ids_by_position: tuple[np.ndarray, ...]
    total_unique_features: int
    activation_threshold: float


@dataclass(frozen=True)
class PositionFeatureAudit:
    feature_ids: np.ndarray
    previous_activation: np.ndarray
    current_activation: np.ndarray
    next_activation: np.ndarray
    gain_from_previous: np.ndarray
    local_mean_contrast: np.ndarray
    local_max_contrast: np.ndarray
    activation_rank: np.ndarray
    gain_rank: np.ndarray
    local_mean_rank: np.ndarray
    local_max_rank: np.ndarray
    position_1based: int


@dataclass(frozen=True)
class PositionActivationCounts:
    active_at_position: int
    increasing_from_previous: int
    new_at_position: int
    local_to_position: int
    active_nowhere_else: int


@dataclass(frozen=True)
class LocalizationMetrics:
    feature_ids: np.ndarray
    effective_residue_span: np.ndarray
    normalized_effective_residue_span: np.ndarray
    top_n_activation_fraction: np.ndarray


def _feature_ids(feature_ids: Any, *, size: int) -> np.ndarray:
    ids = np.asarray(feature_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("feature_ids must be a one-dimensional integer array")
    ids64 = ids.astype(np.int64, copy=False)
    if np.any(ids64 < 0) or np.any(ids64 >= size):
        raise ValueError("feature ID is outside the codebook")
    if len(np.unique(ids64)) != len(ids64):
        raise ValueError("feature IDs must be unique")
    return ids64


def rank_feature_scores(feature_ids: Any, scores: Any) -> FeatureRanking:
    """Rank descending score, breaking ties by ascending feature ID."""

    ids = np.asarray(feature_ids)
    values = np.asarray(scores, dtype=np.float64)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or values.shape != ids.shape:
        raise ValueError("feature_ids and scores must be matching one-dimensional arrays")
    ids64 = ids.astype(np.int64, copy=False)
    if len(np.unique(ids64)) != len(ids64) or np.any(ids64 < 0):
        raise ValueError("feature IDs must be unique and non-negative")
    if not np.isfinite(values).all():
        raise ValueError("feature scores contain NaN or infinity")
    order = np.lexsort((ids64, -values))
    ranks = np.empty(len(ids64), dtype=np.int64)
    ranks[order] = np.arange(1, len(ids64) + 1, dtype=np.int64)
    return FeatureRanking(ids64[order], values[order], ranks)


def rank_wt_features(wt_profile: SparseVector) -> FeatureRanking:
    """Rank every active WT max-pooled feature by activation then feature ID."""

    return rank_feature_scores(wt_profile.indices, wt_profile.values)


def profile_feature_matrix(
    profiles: Sequence[SparseVector],
    feature_ids: Any,
) -> np.ndarray:
    """Materialize selected max-pooled coordinates for a candidate cohort."""

    if not profiles:
        raise ValueError("at least one candidate profile is required")
    size = profiles[0].size
    if any(profile.size != size for profile in profiles):
        raise ValueError("all sparse profiles must use the same codebook")
    ids = _feature_ids(feature_ids, size=size)
    return np.asarray([profile.values_at(ids) for profile in profiles], dtype=np.float32)


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson interval used for feature-recurrence proportions."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total and total > 0")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Wilson z must be positive and finite")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def feature_recurrence(
    profiles: Sequence[SparseVector],
    feature_ids: Any,
    *,
    activation_threshold: float = 0.0,
    z: float = 1.959963984540054,
) -> RecurrenceResult:
    """Fraction of candidates in which each feature is active anywhere.

    Candidate values are normalized max-pooled activations.  Therefore
    ``value > 0`` means the feature occurred at at least one residue.
    """

    if not np.isfinite(activation_threshold) or activation_threshold < 0:
        raise ValueError("activation_threshold must be finite and non-negative")
    matrix = profile_feature_matrix(profiles, feature_ids)
    ids = _feature_ids(feature_ids, size=profiles[0].size)
    active_counts = np.count_nonzero(matrix > activation_threshold, axis=0).astype(np.int64)
    total = matrix.shape[0]
    recurrence = active_counts.astype(np.float64) / total
    intervals = np.asarray(
        [wilson_interval(int(count), total, z=z) for count in active_counts],
        dtype=np.float64,
    )
    return RecurrenceResult(
        ids,
        active_counts,
        total,
        recurrence,
        intervals[:, 0],
        intervals[:, 1],
    )


def recurrence_by_stratum(
    profiles: Sequence[SparseVector],
    strata: Sequence[str],
    feature_ids: Any,
    *,
    activation_threshold: float = 0.0,
) -> dict[str, RecurrenceResult]:
    """Compute recurrence independently within each supplied cohort stratum."""

    if len(profiles) != len(strata):
        raise ValueError("profiles and strata must have equal length")
    ordered_labels = list(dict.fromkeys(str(label) for label in strata))
    if not ordered_labels or any(not label for label in ordered_labels):
        raise ValueError("stratum labels must be non-empty")
    result: dict[str, RecurrenceResult] = {}
    for label in ordered_labels:
        selected = [
            profile
            for profile, value in zip(profiles, strata, strict=True)
            if str(value) == label
        ]
        result[label] = feature_recurrence(
            selected,
            feature_ids,
            activation_threshold=activation_threshold,
        )
    return result


def feature_contribution_spectrum(
    candidate: SparseVector,
    wt: SparseVector,
    *,
    feature_order: Any | None = None,
    require_all_wt_features: bool = True,
) -> ContributionSpectrum:
    """Decompose candidate--WT SAE cosine into additive WT-feature terms.

    By default, features follow normalized WT max-pool rank.  A complete order
    reconstructs the score because features absent from WT have zero dot-product
    contribution, even when they increase the candidate norm.
    """

    if candidate.size != wt.size:
        raise ValueError("candidate and WT profiles use different codebooks")
    if feature_order is None:
        ids = rank_wt_features(wt).feature_ids
    else:
        ids = _feature_ids(feature_order, size=wt.size)
    wt_ids = np.asarray(wt.indices, dtype=np.int64)
    if require_all_wt_features and (len(ids) != len(wt_ids) or set(ids) != set(wt_ids)):
        raise ValueError("the contribution order must contain every active WT feature exactly once")

    wt_values = wt.values_at(ids).astype(np.float64)
    candidate_values = candidate.values_at(ids).astype(np.float64)
    products = candidate_values * wt_values
    denominator = candidate.norm * wt.norm
    contributions = products / denominator if denominator else np.zeros_like(products)
    cumulative = np.cumsum(contributions, dtype=np.float64)
    full_numerator = sparse_dot(candidate, wt)
    if full_numerator > 0:
        numerator_shares = products / full_numerator
    else:
        # The share of an all-zero numerator is undefined mathematically; zeros
        # are operationally useful and preserve the invariant that no feature
        # contributes. Valid endpoint rows have positive numerators.
        numerator_shares = np.zeros_like(products)
    cumulative_numerator = np.cumsum(numerator_shares, dtype=np.float64)
    cosine = sparse_cosine(candidate, wt)
    reconstructed = float(contributions.sum(dtype=np.float64))
    if require_all_wt_features and not np.isclose(reconstructed, cosine, rtol=0.0, atol=1e-12):
        raise ValueError(
            "per-feature contributions do not reconstruct candidate--WT SAE cosine: "
            f"{reconstructed} != {cosine}"
        )
    if require_all_wt_features and cumulative_numerator.size and full_numerator > 0:
        if cumulative_numerator[-1] > 1.0 + 1e-12:
            raise ValueError("cumulative numerator share exceeds one")
    return ContributionSpectrum(
        ids,
        wt_values,
        candidate_values,
        products,
        contributions,
        cumulative,
        numerator_shares,
        cumulative_numerator,
        cosine,
        reconstructed,
    )


def cohort_contribution_matrix(
    profiles: Sequence[SparseVector],
    wt: SparseVector,
    *,
    feature_order: Any | None = None,
) -> CohortContributionMatrix:
    """Contribution and cumulative-contribution rows for every candidate."""

    if not profiles:
        raise ValueError("at least one candidate profile is required")
    ids = rank_wt_features(wt).feature_ids if feature_order is None else _feature_ids(
        feature_order, size=wt.size
    )
    spectra = [
        feature_contribution_spectrum(profile, wt, feature_order=ids)
        for profile in profiles
    ]
    return CohortContributionMatrix(
        ids,
        np.vstack([spectrum.cosine_contributions for spectrum in spectra]),
        np.vstack([spectrum.cumulative_cosine_contributions for spectrum in spectra]),
        np.vstack([spectrum.numerator_shares for spectrum in spectra]),
        np.vstack([spectrum.cumulative_numerator_shares for spectrum in spectra]),
        np.asarray([spectrum.cosine_similarity for spectrum in spectra], dtype=np.float64),
    )


def positional_activations(
    residues: SparseResidueMatrix,
    feature_ids: Any,
) -> np.ndarray:
    """Return normalized residue activations for selected features as ``(L, F)``."""

    ids = _feature_ids(feature_ids, size=residues.size)
    return residues.feature_matrix(ids)


def feature_trace(residues: SparseResidueMatrix, feature_id: int) -> np.ndarray:
    return positional_activations(residues, np.asarray([feature_id], dtype=np.int64))[:, 0]


def position_recurrence(
    residue_profiles: Sequence[SparseResidueMatrix],
    feature_ids: Any,
    *,
    position_1based: int,
    activation_threshold: float = 0.0,
) -> RecurrenceResult:
    """Candidate prevalence of selected features at one directly mapped position."""

    if not residue_profiles:
        raise ValueError("at least one residue profile is required")
    if not np.isfinite(activation_threshold) or activation_threshold < 0:
        raise ValueError("activation_threshold must be finite and non-negative")
    size = residue_profiles[0].size
    if any(profile.size != size for profile in residue_profiles):
        raise ValueError("all residue profiles must use the same codebook")
    ids = _feature_ids(feature_ids, size=size)
    position = position_1based - 1
    if position < 0 or any(position >= profile.sequence_length for profile in residue_profiles):
        raise ValueError("position is outside at least one candidate profile")
    rows = np.vstack([profile.feature_matrix(ids)[position] for profile in residue_profiles])
    counts = np.count_nonzero(rows > activation_threshold, axis=0).astype(np.int64)
    total = len(residue_profiles)
    intervals = np.asarray([wilson_interval(int(value), total) for value in counts])
    return RecurrenceResult(
        ids,
        counts,
        total,
        counts.astype(np.float64) / total,
        intervals[:, 0],
        intervals[:, 1],
    )


def position_unique_features(
    residues: SparseResidueMatrix,
    *,
    activation_threshold: float = 0.0,
) -> PositionUniqueResult:
    """Identify features active at one WT residue and no other WT residue."""

    if not np.isfinite(activation_threshold) or activation_threshold < 0:
        raise ValueError("activation_threshold must be finite and non-negative")
    ids = np.asarray(residues.indices, dtype=np.int64)
    values = np.asarray(residues.values, dtype=np.float32)
    active = values > activation_threshold
    active_ids = ids[active]
    feature_position_counts = np.bincount(active_ids, minlength=residues.size)
    per_position: list[np.ndarray] = []
    for position in range(residues.sequence_length):
        row_ids = ids[position, active[position]]
        unique = row_ids[feature_position_counts[row_ids] == 1]
        per_position.append(np.asarray(unique, dtype=np.int64))
    counts = np.asarray([len(row) for row in per_position], dtype=np.int64)
    return PositionUniqueResult(
        np.arange(1, residues.sequence_length + 1, dtype=np.int64),
        counts,
        tuple(per_position),
        int(np.count_nonzero(feature_position_counts == 1)),
        float(activation_threshold),
    )


def empirical_greater_equal_tail(observed: int, controls: Any) -> float:
    """One-sided empirical tail with the focal observation as the +1 sample."""

    values = np.asarray(controls)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError("controls must be a non-empty numeric vector")
    if not np.isfinite(values).all():
        raise ValueError("controls contain NaN or infinity")
    return float((1 + np.count_nonzero(values >= observed)) / (1 + len(values)))


def position_unique_empirical_p(result: PositionUniqueResult, *, position_1based: int) -> float:
    index = position_1based - 1
    if index < 0 or index >= len(result.counts):
        raise ValueError("position is outside the sequence")
    controls = np.delete(result.counts, index)
    if controls.size == 0:
        raise ValueError("an empirical position null requires at least two residues")
    return empirical_greater_equal_tail(int(result.counts[index]), controls)


def position_feature_audit(
    residues: SparseResidueMatrix,
    feature_ids: Any,
    *,
    position_1based: int,
) -> PositionFeatureAudit:
    """Calculate the exact Figure 2 positional selection scores and ranks."""

    position = position_1based - 1
    if position <= 0 or position >= residues.sequence_length - 1:
        raise ValueError("positional audit requires a previous and next residue")
    ids = _feature_ids(feature_ids, size=residues.size)
    matrix = residues.feature_matrix(ids).astype(np.float64)
    previous = matrix[position - 1]
    current = matrix[position]
    following = matrix[position + 1]
    gain = current - previous
    local_mean = current - 0.5 * (previous + following)
    local_max = current - np.maximum(previous, following)
    activation_ranking = rank_feature_scores(ids, current)
    gain_ranking = rank_feature_scores(ids, gain)
    mean_ranking = rank_feature_scores(ids, local_mean)
    max_ranking = rank_feature_scores(ids, local_max)
    return PositionFeatureAudit(
        ids,
        previous,
        current,
        following,
        gain,
        local_mean,
        local_max,
        activation_ranking.ranks,
        gain_ranking.ranks,
        mean_ranking.ranks,
        max_ranking.ranks,
        position_1based,
    )


def position_activation_counts(
    residues: SparseResidueMatrix,
    feature_ids: Any,
    *,
    position_1based: int,
    activation_threshold: float = 0.0,
) -> PositionActivationCounts:
    """Count active, incoming, local, and WT-position-unique features."""

    if not np.isfinite(activation_threshold) or activation_threshold < 0:
        raise ValueError("activation_threshold must be finite and non-negative")
    position = position_1based - 1
    if position <= 0 or position >= residues.sequence_length - 1:
        raise ValueError("position counts require a previous and next residue")
    ids = _feature_ids(feature_ids, size=residues.size)
    matrix = residues.feature_matrix(ids).astype(np.float64)
    active = matrix > activation_threshold
    gain = matrix[position] - matrix[position - 1]
    return PositionActivationCounts(
        active_at_position=int(active[position].sum()),
        increasing_from_previous=int(np.count_nonzero(gain > 0)),
        new_at_position=int(np.count_nonzero(~active[position - 1] & active[position])),
        local_to_position=int(
            np.count_nonzero(
                ~active[position - 1] & active[position] & ~active[position + 1]
            )
        ),
        active_nowhere_else=int(
            np.count_nonzero(active[position] & (active.sum(axis=0) == 1))
        ),
    )


def feature_localization_metrics(
    residues: SparseResidueMatrix,
    feature_ids: Any,
    *,
    top_n: int = 10,
) -> LocalizationMetrics:
    """Effective activation span and top-N concentration for WT features."""

    ids = _feature_ids(feature_ids, size=residues.size)
    matrix = residues.feature_matrix(ids).astype(np.float64)
    sums = matrix.sum(axis=0)
    if np.any(sums <= 0):
        raise ValueError("every selected feature must be active in the WT profile")
    probabilities = matrix / sums[None, :]
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log(probabilities[positive])
    entropy = -np.sum(probabilities * log_probabilities, axis=0)
    effective = np.exp(entropy)
    selected_n = min(int(top_n), residues.sequence_length)
    if selected_n <= 0:
        raise ValueError("top_n must be positive")
    top_fraction = np.partition(matrix, -selected_n, axis=0)[-selected_n:, :].sum(axis=0) / sums
    return LocalizationMetrics(
        ids,
        effective,
        effective / residues.sequence_length,
        top_fraction,
    )
