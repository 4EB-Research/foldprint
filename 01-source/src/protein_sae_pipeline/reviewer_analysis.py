"""Portable analysis tables for confidence controls and the reviewer appendix.

Accepts the public canonical master schema or the historical mutavault column
names. Explicit identity and TM rules are recomputed, rather than trusting
previously filtered tables or boolean flags read from CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .bootstrap import make_bootstrap_plan
from .figures.common import finite_numeric, require_columns, require_unique_keys
from .figures.figure3 import SCORE_COLUMNS

KEYS = ["reference_id", "sequence_sha256"]
ALIASES = {
    "protein_key": "reference_id", "origin": "generation_method",
    "hamming_sequence_identity": "positional_identity_fraction",
    "usalign_alignment_optimized_tm_score_wt_norm": "usalign_tm_score_wt_norm",
    "raw_esmc_layer60_cosine_to_wt": "esmc_cosine_to_wt",
}
PAPER_LABELS = {"lambda_gpd": "gpD", "ramr": "RamR", "fniii_10": "FNIII-10",
                "pdz3": "PDZ3", "rnase_a": "RNase A", "dhfr": "DHFR"}


def canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for old, new in ALIASES.items():
        if old in result:
            if new in result:
                if not np.array_equal(result[old].to_numpy(), result[new].to_numpy()):
                    raise ValueError(f"Conflicting historical and canonical columns: {old}, {new}")
                result = result.drop(columns=old)
            else:
                result = result.rename(columns={old: new})
    return result


def read_table(path: Path) -> pd.DataFrame:
    # round_trip is required for strict threshold comparisons at score ties.
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def validate_landscape(frame: pd.DataFrame) -> pd.DataFrame:
    result = canonical_columns(frame)
    numeric = ["positional_identity_fraction", "usalign_tm_score_wt_norm", *SCORE_COLUMNS.values()]
    require_columns(result, [*KEYS, "generation_method", *numeric], label="Reviewer landscape")
    require_unique_keys(result, label="Reviewer landscape")
    finite_numeric(result, numeric, label="Reviewer landscape")
    if result.empty or result.generation_method.isna().any():
        raise ValueError("Landscape must be nonempty with complete generator labels")
    for column in numeric[:2]:
        if not result[column].between(0, 1).all():
            raise ValueError(f"{column} must be a fraction in [0,1]")
    result["is_remote"] = result.positional_identity_fraction.lt(.3)
    result["is_structural_analogue"] = result.usalign_tm_score_wt_norm.ge(.5)
    result["structural_class"] = np.where(result.is_structural_analogue, "analogue", "negative")
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def join_metrics(frame: pd.DataFrame, metrics: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    metrics = canonical_columns(metrics)
    require_unique_keys(metrics, label="Supplemental metrics")
    require_columns(metrics, columns, label="Supplemental metrics")
    finite_numeric(metrics, columns, label="Supplemental metrics")
    overlap = set(columns) & set(frame)
    if overlap:
        raise ValueError(f"Supplemental columns already in master: {sorted(overlap)}")
    result = frame.merge(metrics[KEYS + columns], on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Supplemental metrics do not cover every requested candidate key")
    return result.drop(columns="_merge")


def auc(case, control) -> float:
    """Rank AUROC with half credit for ties; larger values predict analogs."""
    case, control = np.asarray(case, float), np.asarray(control, float)
    if not len(case) or not len(control) or not np.isfinite(np.r_[case, control]).all():
        raise ValueError("AUROC requires finite scores in both classes")
    ranks = rankdata(np.r_[case, control], method="average")
    return float((ranks[:len(case)].sum() - len(case) * (len(case) + 1) / 2) / (len(case) * len(control)))


def grid_values(case, control, grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Historical conservative calibration, including the full-range endpoint."""
    case, control, grid = np.sort(np.asarray(case, float)), np.sort(np.asarray(control, float)), np.asarray(grid, float)
    if not len(case) or not len(control) or not np.isfinite(np.r_[case, control]).all():
        raise ValueError("Retrieval requires finite scores in both classes")
    if not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError("FPR grid must be finite and in [0,1]")
    allowed = np.floor(grid * len(control) + 1e-12).astype(int)
    threshold = control[len(control) - np.minimum(allowed, len(control) - 1) - 1].copy()
    threshold[grid == 1] = -np.inf
    recall = 1 - np.searchsorted(case, threshold, side="right") / len(case)
    fpr = 1 - np.searchsorted(control, threshold, side="right") / len(control)
    return recall, fpr, threshold


def empirical_curve(case, control) -> pd.DataFrame:
    """All observed thresholds; never interpolate across ties."""
    case, control = np.sort(case), np.sort(control)
    thresholds = np.r_[np.inf, np.unique(np.r_[case, control])[::-1], -np.inf]
    return pd.DataFrame({
        "threshold": thresholds,
        "observed_fpr": 1 - np.searchsorted(control, thresholds, side="right") / len(control),
        "recall": 1 - np.searchsorted(case, thresholds, side="right") / len(case),
    })


@dataclass(frozen=True)
class Retrieval:
    curves: pd.DataFrame
    bands: pd.DataFrame
    endpoints: pd.DataFrame
    aucs: pd.DataFrame
    endpoint_draws: pd.DataFrame
    paired_differences: pd.DataFrame
    bootstrap_roc: dict[str, np.ndarray]


def evaluate_retrieval(frame: pd.DataFrame, *, seed: int = 20260827,
                       endpoint_replicates: int = 2000, roc_replicates: int = 2000,
                       matched: bool = False) -> Retrieval:
    """Shared row draws across methods, FPR endpoints, and ROC band grids.

    Full cohorts resample within target/class/generator. Matched cohorts
    resample frozen complete pairs within target/ordered generator combination,
    reproducing the uncertainty unit of the actual matching source code.
    """
    if min(endpoint_replicates, roc_replicates) < 1:
        raise ValueError("Bootstrap replicate counts must be positive")
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    if matched:
        require_columns(frame, ["pair_id"], label="Matched bootstrap")
        grouped = frame.groupby("pair_id", sort=True)
        if frame.pair_id.isna().any() or not grouped.size().eq(2).all() or not grouped.structural_class.nunique().eq(2).all() or not grouped.reference_id.nunique().eq(1).all():
            raise ValueError("Bootstrap requires complete within-target analogue/negative pairs")
        positive = frame.loc[frame.structural_class.eq("analogue")].sort_values("pair_id")
        negative = frame.loc[frame.structural_class.eq("negative")].sort_values("pair_id")
        units = positive[KEYS + ["generation_method"]].reset_index(drop=True)
        units["negative_generator"] = negative.generation_method.to_numpy()
        strata = ("reference_id", "generation_method", "negative_generator")
    else:
        units = frame
        strata = ("reference_id", "structural_class", "generation_method")
    plan = make_bootstrap_plan(units, replicates=max(endpoint_replicates, roc_replicates), seed=seed, strata=strata)
    layout = np.concatenate(plan.groups)
    references = sorted(frame.reference_id.unique())
    # The union gives exact shared 1%/5% endpoints, low-FPR and full-range bands.
    grid = np.unique(np.round(np.r_[np.linspace(0, .05, 101), np.linspace(0, 1, 101)], 10))
    endpoint_indices = [int(np.flatnonzero(grid == q)[0]) for q in (.01, .05)]
    curve_rows, band_rows, point_rows, auc_rows, draw_rows, difference_rows = [], [], [], [], [], []
    stored = {"fpr_grid": grid}
    # Work target-wise; store only the compact recall cube rather than millions
    # of repeated threshold/count dataframe rows.
    for reference in references:
        target = frame.loc[frame.reference_id.eq(reference)]
        cases = target.loc[target.structural_class.eq("analogue")]
        controls = target.loc[target.structural_class.eq("negative")]
        if cases.empty or controls.empty:
            raise ValueError(f"{reference}: retrieval requires both structural classes")
        boot = np.empty((roc_replicates, len(SCORE_COLUMNS), len(grid)), dtype=np.float32)
        endpoint_boot = np.empty((endpoint_replicates, len(SCORE_COLUMNS), 2))
        target_layout = units.iloc[layout].reference_id.eq(reference).to_numpy()
        if not matched:
            labels = units.iloc[layout].structural_class.to_numpy()
            ci = target_layout & (labels == "analogue")
            ni = target_layout & (labels == "negative")
        else:
            ci = ni = target_layout
        for replicate in range(max(endpoint_replicates, roc_replicates)):
            sample = plan.draw(replicate)
            if matched:
                case_draw = positive.iloc[sample[ci]]
                control_draw = negative.iloc[sample[ni]]
            else:
                case_draw, control_draw = frame.iloc[sample[ci]], frame.iloc[sample[ni]]
            for mi, (method, column) in enumerate(SCORE_COLUMNS.items()):
                budgets = grid if replicate < roc_replicates else np.array([.01, .05])
                recalls, _, _ = grid_values(case_draw[column].to_numpy(), control_draw[column].to_numpy(), budgets)
                if replicate < roc_replicates:
                    boot[replicate, mi] = recalls
                if replicate < endpoint_replicates:
                    endpoint_boot[replicate, mi] = recalls[endpoint_indices] if len(budgets) > 2 else recalls
        stored[str(reference)] = boot
        for mi, (method, column) in enumerate(SCORE_COLUMNS.items()):
            x, y = cases[column].to_numpy(), controls[column].to_numpy()
            curve = empirical_curve(x, y)
            curve["reference_id"], curve["method"] = reference, method
            curve_rows.append(curve)
            lower, upper = np.quantile(boot[:, mi], [.025, .975], axis=0)
            band_rows.append(pd.DataFrame({"reference_id": reference, "method": method, "target_fpr": grid,
                                          "ci_lower": lower, "ci_upper": upper, "bootstrap_replicates": roc_replicates}))
            recall, fpr, threshold = grid_values(x, y, [.01, .05])
            for ei, q in enumerate((.01, .05)):
                lo, hi = np.quantile(endpoint_boot[:, mi, ei], [.025, .975])
                point_rows.append({"reference_id": reference, "method": method, "target_fpr": q,
                    "recall": recall[ei], "observed_fpr": fpr[ei], "threshold": threshold[ei],
                    "case_count": len(x), "control_count": len(y), "ci_lower": lo, "ci_upper": hi,
                    "bootstrap_replicates": endpoint_replicates})
                draw_rows.append(pd.DataFrame({"reference_id": reference, "method": method, "target_fpr": q,
                    "replicate": np.arange(endpoint_replicates), "recall": endpoint_boot[:, mi, ei]}))
                baseline_index = list(SCORE_COLUMNS).index("ESM-C")
                if method != "ESM-C":
                    delta = endpoint_boot[:, mi, ei] - endpoint_boot[:, baseline_index, ei]
                    dlow, dhigh = np.quantile(delta, [.025, .975])
                    difference_rows.append({"reference_id": reference, "method": method, "baseline": "ESM-C",
                        "target_fpr": q, "ci_lower": dlow, "ci_upper": dhigh,
                        "bootstrap_replicates": endpoint_replicates})
            auc_rows.append({"reference_id": reference, "method": method, "auc": auc(x, y),
                             "case_count": len(x), "control_count": len(y)})
    return Retrieval(pd.concat(curve_rows, ignore_index=True), pd.concat(band_rows, ignore_index=True),
        pd.DataFrame(point_rows), pd.DataFrame(auc_rows), pd.concat(draw_rows, ignore_index=True),
        pd.DataFrame(difference_rows), stored)


def score_separation(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in cohorts.items():
        for reference, target in frame.groupby("reference_id", sort=True):
            for method, column in SCORE_COLUMNS.items():
                positive = target.loc[target.structural_class.eq("analogue"), column].to_numpy()
                negative = target.loc[target.structural_class.eq("negative"), column].to_numpy()
                if min(len(positive), len(negative)) < 2:
                    raise ValueError(f"{name}/{reference}: standardized separation requires >=2 rows per class")
                denominator = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2)
                rows.append({"cohort": name, "reference_id": reference, "method": method,
                    "case_count": len(positive), "control_count": len(negative),
                    "case_mean": positive.mean(), "control_mean": negative.mean(),
                    "case_variance": positive.var(ddof=1), "control_variance": negative.var(ddof=1),
                    "standardizing_denominator": denominator,
                    "D": (positive.mean() - negative.mean()) / denominator if denominator else np.nan})
    return pd.DataFrame(rows)


def distribution_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for (reference, label), group in frame.groupby(["reference_id", "structural_class"], sort=True):
        x = group[column].to_numpy()
        q = np.quantile(x, [0, .25, .5, .75, 1])
        rows.append({"reference_id": reference, "structural_class": label, "n": len(x), "mean": x.mean(),
                     **dict(zip(["minimum", "q25", "median", "q75", "maximum"], q, strict=True))})
    return pd.DataFrame(rows)
