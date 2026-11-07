"""Paired, stratified candidate bootstrap shared by all score methods."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd

from .statistics import empirical_operating_points, empirical_roc_grid


DEFAULT_STRATA = ("reference_id", "structural_class", "generation_method")
KEY_COLUMNS = ("reference_id", "sequence_sha256")


@dataclass
class BootstrapPlan:
    seed: int
    strata: tuple[str, ...]
    source_rows: int
    replicate_count: int
    groups: tuple[np.ndarray, ...]
    algorithm: str = "numpy_default_rng_single_stream_stratified_v1"
    _states: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _next_rng: np.random.Generator = field(init=False, repr=False)

    @property
    def replicates(self) -> int:
        return int(self.replicate_count)

    def __post_init__(self) -> None:
        if self.seed < 0 or self.source_rows <= 0 or self.replicate_count <= 0:
            raise ValueError("bootstrap seed, source rows, and replicate count are invalid")
        if not self.groups or sum(len(group) for group in self.groups) != self.source_rows:
            raise ValueError("bootstrap groups must form a complete source-row partition")
        flattened: list[np.ndarray] = []
        for group in self.groups:
            if group.dtype.kind not in "iu" or group.ndim != 1 or not len(group):
                raise ValueError("bootstrap groups must be nonempty integer vectors")
            if np.any(group < 0) or np.any(group >= self.source_rows):
                raise ValueError("bootstrap group contains an out-of-range row index")
            flattened.append(group.astype(np.int64, copy=False))
        combined = np.concatenate(flattened)
        if len(np.unique(combined)) != self.source_rows:
            raise ValueError("bootstrap groups overlap or omit source rows")
        self._next_rng = np.random.default_rng(int(self.seed))

    def _draw_with_rng(self, rng: np.random.Generator) -> np.ndarray:
        sampled = np.empty(self.source_rows, dtype=np.int64)
        cursor = 0
        for group in self.groups:
            stop = cursor + len(group)
            sampled[cursor:stop] = rng.choice(group, size=len(group), replace=True)
            cursor = stop
        return sampled

    def draw(self, replicate: int) -> np.ndarray:
        """Return one deterministic stratified candidate-row sample.

        Compact PCG64 states make arbitrary replicate access possible without
        retaining a 10,000 by 66,653 integer matrix in memory. Sampling uses
        the same single seeded NumPy RNG stream as a fully materialized plan,
        and the result is identical every time a replicate is reused by an
        endpoint, method, paired comparison, or ROC calculation.
        """

        if not 0 <= replicate < self.replicates:
            raise IndexError(replicate)
        if replicate < len(self._states):
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(self._states[replicate])
            return self._draw_with_rng(rng)

        requested: np.ndarray | None = None
        while len(self._states) <= replicate:
            current = len(self._states)
            self._states.append(copy.deepcopy(self._next_rng.bit_generator.state))
            drawn = self._draw_with_rng(self._next_rng)
            if current == replicate:
                requested = drawn
        assert requested is not None
        return requested

    @property
    def sampled_ilocs(self) -> np.ndarray:
        """Materialize small plans for tests and debugging; large plans stay lazy."""

        cells = self.replicates * self.source_rows
        if cells > 50_000_000:
            gib = cells * 8 / 1024**3
            raise MemoryError(
                f"Refusing to materialize a {gib:.2f}-GiB bootstrap matrix; "
                "use draw(replicate) or sample(frame, replicate)"
            )
        return np.vstack([self.draw(replicate) for replicate in range(self.replicates)])

    def definition_sha256(self) -> str:
        """Digest the complete deterministic plan definition without drawing it."""

        payload = json.dumps(
            {
                "schema_version": 1,
                "algorithm": self.algorithm,
                "seed": self.seed,
                "strata": self.strata,
                "source_rows": self.source_rows,
                "replicates": self.replicates,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hashlib.sha256(payload)
        for group in self.groups:
            canonical = np.asarray(group, dtype="<u8")
            digest.update(len(canonical).to_bytes(8, "little"))
            digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()

    def sample(self, frame: pd.DataFrame, replicate: int) -> pd.DataFrame:
        if len(frame) != self.source_rows:
            raise ValueError("Frame row count differs from bootstrap source")
        if not 0 <= replicate < self.replicates:
            raise IndexError(replicate)
        return frame.iloc[self.draw(replicate)].reset_index(drop=True)


def make_bootstrap_plan(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    strata: Iterable[str] = DEFAULT_STRATA,
) -> BootstrapPlan:
    strata_tuple = tuple(strata)
    required = set(strata_tuple) | set(KEY_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap frame lacks columns: {sorted(missing)}")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if frame[list(KEY_COLUMNS)].duplicated().any():
        raise ValueError("Bootstrap source contains duplicate candidate keys")

    # Stable source positions are defined once from the caller's frame. Each
    # stratum is sorted by canonical key before RNG draws, making row order
    # irrelevant while retaining ilocs that can be shared across score methods.
    work = frame.reset_index(drop=True)
    groups: list[np.ndarray] = []
    grouped = work.groupby(list(strata_tuple), sort=True, observed=True, dropna=False)
    for _key, group in grouped:
        ordered = group.sort_values(list(KEY_COLUMNS), kind="stable")
        groups.append(ordered.index.to_numpy(dtype=np.int64))
    if not groups or sum(len(group) for group in groups) != len(work):
        raise RuntimeError("Bootstrap strata do not form a complete partition")

    return BootstrapPlan(
        int(seed),
        strata_tuple,
        len(work),
        int(replicates),
        tuple(groups),
    )


def bootstrap_operating_points(
    frame: pd.DataFrame,
    plan: BootstrapPlan,
    score_columns: dict[str, str],
    *,
    endpoints: Iterable[float] = (0.01, 0.05),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    endpoints_tuple = tuple(float(value) for value in endpoints)
    references, layout = _class_layout(frame, plan)
    scores = _score_arrays(frame, score_columns)
    for replicate in range(plan.replicates):
        draw = plan.draw(replicate)
        for reference_id in references:
            case_ilocs = draw[layout[(reference_id, "analogue")]]
            control_ilocs = draw[layout[(reference_id, "negative")]]
            for method in score_columns:
                # The configured endpoints share one candidate sample and one
                # score ordering within this method/reference.
                points = empirical_operating_points(
                    scores[method][case_ilocs],
                    scores[method][control_ilocs],
                    endpoints_tuple,
                )
                for point in points:
                    rows.append(
                        {
                            "replicate": replicate,
                            "reference_id": reference_id,
                            "method": method,
                            "target_fpr": point.target_fpr,
                            "threshold": point.threshold,
                            "observed_fpr": point.observed_fpr,
                            "recall": point.recall,
                            "case_count": point.case_count,
                            "control_count": point.control_count,
                        }
                    )
    return pd.DataFrame(rows)


def _class_layout(
    frame: pd.DataFrame,
    plan: BootstrapPlan,
) -> tuple[tuple[str, ...], dict[tuple[str, str], np.ndarray]]:
    """Map invariant sampled-layout positions to reference/class groups."""

    if len(frame) != plan.source_rows:
        raise ValueError("Frame row count differs from bootstrap source")
    required = {"reference_id", "structural_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap frame lacks endpoint labels: {sorted(missing)}")
    source_layout = np.concatenate(plan.groups)
    reference = frame["reference_id"].astype(str).to_numpy()[source_layout]
    structural_class = frame["structural_class"].astype(str).to_numpy()[source_layout]
    references = tuple(sorted(set(reference.tolist())))
    layout: dict[tuple[str, str], np.ndarray] = {}
    for reference_id in references:
        for label in ("analogue", "negative"):
            positions = np.flatnonzero(
                (reference == reference_id) & (structural_class == label)
            )
            if not len(positions):
                raise ValueError(f"Reference {reference_id} lacks bootstrap class {label}")
            layout[(reference_id, label)] = positions
    if set(structural_class.tolist()) != {"analogue", "negative"}:
        raise ValueError("Bootstrap frame contains an unknown structural class")
    return references, layout


def _score_arrays(
    frame: pd.DataFrame,
    score_columns: dict[str, str],
) -> dict[str, np.ndarray]:
    missing = set(score_columns.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap frame lacks score columns: {sorted(missing)}")
    arrays = {
        method: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        for method, column in score_columns.items()
    }
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Bootstrap score arrays must be finite")
    return arrays


def percentile_intervals(
    replicates: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Iterable[str],
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> pd.DataFrame:
    groups = list(group_columns)
    rows: list[dict[str, object]] = []
    for key, group in replicates.groupby(groups, sort=True, observed=True):
        values = pd.to_numeric(group[value_column], errors="raise").to_numpy(float)
        keys = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(groups, keys, strict=True)),
                "bootstrap_replicates": int(len(values)),
                "median": float(np.percentile(values, 50)),
                "ci_lower": float(np.percentile(values, percentiles[0])),
                "ci_upper": float(np.percentile(values, percentiles[1])),
            }
        )
    return pd.DataFrame(rows)


def paired_method_differences(
    replicates: pd.DataFrame,
    *,
    baseline: str,
    value_column: str = "recall",
) -> pd.DataFrame:
    index = ["replicate", "reference_id", "target_fpr"]
    wide = replicates.pivot(index=index, columns="method", values=value_column)
    if baseline not in wide:
        raise ValueError(f"Unknown baseline method: {baseline}")
    rows: list[pd.DataFrame] = []
    for method in wide.columns:
        if method == baseline:
            continue
        values = (wide[method] - wide[baseline]).rename("difference").reset_index()
        values["method"] = method
        values["baseline"] = baseline
        rows.append(values)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bootstrap_roc_bands(
    frame: pd.DataFrame,
    plan: BootstrapPlan,
    score_columns: dict[str, str],
    *,
    max_fpr: float = 0.05,
    grid_points: int = 101,
    replicates: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = plan.replicates if replicates is None else int(replicates)
    if not 0 < count <= plan.replicates:
        raise ValueError("ROC replicate count must use a nonempty prefix of the shared plan")
    references, layout = _class_layout(frame, plan)
    scores = _score_arrays(frame, score_columns)
    methods = tuple(score_columns)
    grid = np.linspace(0.0, float(max_fpr), int(grid_points), dtype=np.float64)
    rows_per_curve = len(grid)
    curve_count = count * len(references) * len(methods)
    total_rows = curve_count * rows_per_curve
    replicate_values = np.empty(total_rows, dtype=np.int32)
    reference_values = np.empty(total_rows, dtype=object)
    method_values = np.empty(total_rows, dtype=object)
    target_values = np.empty(total_rows, dtype=np.float64)
    threshold_values = np.empty(total_rows, dtype=np.float64)
    allowed_values = np.empty(total_rows, dtype=np.int32)
    false_positive_values = np.empty(total_rows, dtype=np.int32)
    control_count_values = np.empty(total_rows, dtype=np.int32)
    observed_fpr_values = np.empty(total_rows, dtype=np.float64)
    true_positive_values = np.empty(total_rows, dtype=np.int32)
    case_count_values = np.empty(total_rows, dtype=np.int32)
    recall_values = np.empty(total_rows, dtype=np.float64)
    cursor = 0
    for replicate in range(count):
        draw = plan.draw(replicate)
        for reference_id in references:
            case_ilocs = draw[layout[(reference_id, "analogue")]]
            control_ilocs = draw[layout[(reference_id, "negative")]]
            for method in methods:
                curve = empirical_roc_grid(
                    scores[method][case_ilocs],
                    scores[method][control_ilocs],
                    max_fpr=max_fpr,
                    grid_points=grid_points,
                )
                stop = cursor + rows_per_curve
                replicate_values[cursor:stop] = replicate
                reference_values[cursor:stop] = reference_id
                method_values[cursor:stop] = method
                target_values[cursor:stop] = curve["target_fpr"].to_numpy(float)
                threshold_values[cursor:stop] = curve["threshold"].to_numpy(float)
                allowed_values[cursor:stop] = curve["allowed_false_positives"].to_numpy(int)
                false_positive_values[cursor:stop] = curve["observed_false_positives"].to_numpy(int)
                control_count_values[cursor:stop] = curve["control_count"].to_numpy(int)
                observed_fpr_values[cursor:stop] = curve["observed_fpr"].to_numpy(float)
                true_positive_values[cursor:stop] = curve["true_positives"].to_numpy(int)
                case_count_values[cursor:stop] = curve["case_count"].to_numpy(int)
                recall_values[cursor:stop] = curve["recall"].to_numpy(float)
                cursor = stop
    if cursor != total_rows:
        raise RuntimeError("ROC bootstrap preallocation was not filled exactly")
    values = pd.DataFrame(
        {
            "replicate": replicate_values,
            "reference_id": reference_values,
            "method": method_values,
            "target_fpr": target_values,
            "threshold": threshold_values,
            "allowed_false_positives": allowed_values,
            "observed_false_positives": false_positive_values,
            "control_count": control_count_values,
            "observed_fpr": observed_fpr_values,
            "true_positives": true_positive_values,
            "case_count": case_count_values,
            "recall": recall_values,
        }
    )
    recall_cube = recall_values.reshape(
        count,
        len(references),
        len(methods),
        rows_per_curve,
    )
    interval_rows: list[dict[str, object]] = []
    for reference_index, reference_id in enumerate(references):
        for method_index, method in enumerate(methods):
            matrix = recall_cube[:, reference_index, method_index, :]
            lower, median, upper = np.percentile(matrix, [2.5, 50.0, 97.5], axis=0)
            interval_rows.extend(
                {
                    "reference_id": reference_id,
                    "method": method,
                    "target_fpr": float(target_fpr),
                    "bootstrap_replicates": count,
                    "median": float(median[index]),
                    "ci_lower": float(lower[index]),
                    "ci_upper": float(upper[index]),
                }
                for index, target_fpr in enumerate(grid)
            )
    bands = pd.DataFrame(interval_rows)
    return values, bands
