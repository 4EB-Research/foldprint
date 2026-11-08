"""Figure 3: all-reference low-identity retrieval and paired bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..bootstrap import (
    bootstrap_operating_points,
    bootstrap_roc_bands,
    make_bootstrap_plan,
    paired_method_differences,
    percentile_intervals,
)
from ..cohorts import construct_cohorts
from ..config import PipelineConfig
from ..statistics import operating_points_by_reference, roc_by_reference
from .common import (
    GRID,
    METHOD_COLORS,
    METHOD_LINEWIDTHS,
    METHOD_MARKERS,
    METHOD_ORDER,
    configure_style,
    finite_numeric,
    require_columns,
    require_unique_keys,
    save_figure,
    write_panel_bundle,
)


SCORE_COLUMNS = {
    "SAE": "sae_cosine_to_wt",
    "ESM-C": "esmc_cosine_to_wt",
    "Pfam": "pfam_full_sequence_bit_score",
    "MMseqs2": "mmseqs2_bit_score",
}


@dataclass(frozen=True)
class Figure3Data:
    roc: pd.DataFrame
    operating_points: pd.DataFrame
    endpoint_bootstrap: pd.DataFrame
    endpoint_intervals: pd.DataFrame
    paired_difference_intervals: pd.DataFrame
    roc_bootstrap: pd.DataFrame
    roc_bands: pd.DataFrame
    source_rows: int
    source_cases: int
    source_controls: int
    endpoint_replicates: int
    roc_replicates: int
    bootstrap_seed: int
    bootstrap_plan_sha256: str
    strata: tuple[str, ...]


def prepare_figure3(master: pd.DataFrame, config: PipelineConfig) -> Figure3Data:
    cohorts = construct_cohorts(master, config)
    remote = cohorts["all_references_remote"].copy().reset_index(drop=True)
    required = {
        "reference_id",
        "sequence_sha256",
        "generation_method",
        "structural_class",
        *SCORE_COLUMNS.values(),
    }
    require_columns(remote, required, label="Figure 3 remote cohort")
    require_unique_keys(remote, label="Figure 3 remote cohort")
    finite_numeric(remote, SCORE_COLUMNS.values(), label="Figure 3 remote cohort")
    if set(remote["structural_class"].astype(str)) != {"analogue", "negative"}:
        raise ValueError("Figure 3 requires both structural endpoint classes")

    endpoints = tuple(float(value) for value in config.raw["thresholds"]["fpr_endpoints"])
    bootstrap = config.bootstrap
    endpoint_replicates = int(bootstrap["endpoint_replicates"])
    roc_replicates = int(bootstrap["roc_replicates"])
    seed = int(bootstrap["seed"])
    strata = tuple(str(value) for value in bootstrap["strata"])
    grid_points = int(bootstrap["roc_grid_points"])
    percentiles = tuple(float(value) for value in bootstrap["interval_percentiles"])

    operating = operating_points_by_reference(remote, SCORE_COLUMNS, endpoints=endpoints)
    roc = roc_by_reference(
        remote,
        SCORE_COLUMNS,
        max_fpr=max(endpoints),
        grid_points=grid_points,
    )
    plan = make_bootstrap_plan(
        remote,
        replicates=endpoint_replicates,
        seed=seed,
        strata=strata,
    )
    plan_sha256 = plan.definition_sha256()
    endpoint_values = bootstrap_operating_points(
        remote,
        plan,
        SCORE_COLUMNS,
        endpoints=endpoints,
    )
    endpoint_intervals = percentile_intervals(
        endpoint_values,
        value_column="recall",
        group_columns=["reference_id", "method", "target_fpr"],
        percentiles=(percentiles[0], percentiles[1]),
    )
    differences = paired_method_differences(
        endpoint_values,
        baseline="ESM-C",
        value_column="recall",
    )
    difference_intervals = percentile_intervals(
        differences,
        value_column="difference",
        group_columns=["reference_id", "method", "baseline", "target_fpr"],
        percentiles=(percentiles[0], percentiles[1]),
    )
    roc_values, roc_bands = bootstrap_roc_bands(
        remote,
        plan,
        SCORE_COLUMNS,
        max_fpr=max(endpoints),
        grid_points=grid_points,
        replicates=roc_replicates,
    )
    # bootstrap_roc_bands currently fixes the configured 2.5/97.5
    # percentiles. Fail closed if a future config asks for something else.
    if percentiles != (2.5, 97.5):
        raise ValueError("ROC bands require configured percentiles [2.5, 97.5]")

    expected_endpoint_rows = (
        endpoint_replicates
        * remote["reference_id"].nunique()
        * len(SCORE_COLUMNS)
        * len(endpoints)
    )
    if len(endpoint_values) != expected_endpoint_rows:
        raise RuntimeError("Endpoint bootstrap does not cover every shared replicate")
    per_replicate = endpoint_values.groupby("replicate", observed=True).size()
    if per_replicate.nunique() != 1:
        raise RuntimeError("Endpoint bootstrap replicate coverage is inconsistent")
    # Each replicate index is common to every method and both FPR endpoints.
    endpoint_keys = endpoint_values.groupby(
        ["replicate", "reference_id"], observed=True
    )["method"].nunique()
    if not endpoint_keys.eq(len(SCORE_COLUMNS)).all():
        raise RuntimeError("Methods did not reuse the shared bootstrap samples")
    endpoint_counts = endpoint_values.groupby(
        ["replicate", "reference_id", "method"], observed=True
    )["target_fpr"].nunique()
    if not endpoint_counts.eq(len(endpoints)).all():
        raise RuntimeError("FPR endpoints did not reuse the shared bootstrap samples")

    return Figure3Data(
        roc,
        operating,
        endpoint_values,
        endpoint_intervals,
        difference_intervals,
        roc_values,
        roc_bands,
        len(remote),
        int(remote["structural_class"].eq("analogue").sum()),
        int(remote["structural_class"].eq("negative").sum()),
        endpoint_replicates,
        roc_replicates,
        seed,
        plan_sha256,
        strata,
    )


def export_figure3(data: Figure3Data, output_root: Path, config: PipelineConfig) -> None:
    write_panel_bundle(
        output_root / "figure3" / "data",
        {
            "figure3_roc_plot_data.csv.gz": data.roc,
            "figure3_operating_points.csv": data.operating_points,
            "figure3_endpoint_bootstrap_replicates.csv.gz": data.endpoint_bootstrap,
            "figure3_endpoint_intervals.csv": data.endpoint_intervals,
            "figure3_paired_difference_intervals.csv": data.paired_difference_intervals,
            "figure3_roc_bootstrap_replicates.csv.gz": data.roc_bootstrap,
            "figure3_roc_bands.csv.gz": data.roc_bands,
        },
        metadata={
            "schema_version": 1,
            "dataset_id": config.dataset_id,
            "source_cohort": "all_references_remote",
            "source_rows": data.source_rows,
            "structural_analogues": data.source_cases,
            "structural_negatives": data.source_controls,
            "score_columns": SCORE_COLUMNS,
            "operating_points": (
                "tie-conservative empirical thresholds; positives require score > threshold"
            ),
            "bootstrap": {
                "seed": data.bootstrap_seed,
                "strata": list(data.strata),
                "endpoint_replicates": data.endpoint_replicates,
                "roc_replicates": data.roc_replicates,
                "shared_plan_sha256": data.bootstrap_plan_sha256,
                "reuse": (
                    "one sampled candidate-row plan reused across methods, 1%/5% endpoints, "
                    "and the ROC prefix"
                ),
                "interval": "pointwise percentile; conditional on observed targets and labels",
            },
        },
        metadata_name="figure3_metadata.json",
    )


def _reference_label(reference_id: str, config: PipelineConfig) -> str:
    return str(config.raw["references"][reference_id]["display_name"])


def _reference_order(values: pd.Series, config: PipelineConfig) -> list[str]:
    available = set(values.astype(str))
    ordered = [
        str(reference_id)
        for reference_id in config.raw["references"]
        if str(reference_id) in available
    ]
    if set(ordered) != available:
        raise ValueError("Figure 3 contains references absent from the frozen config order")
    return ordered


def _draw_figure3_roc(
    container: plt.Figure | plt.SubFigure,
    data: Figure3Data,
    config: PipelineConfig,
) -> None:
    references = _reference_order(data.roc["reference_id"], config)
    columns = min(3, len(references))
    rows = int(np.ceil(len(references) / columns))
    axes = container.subplots(
        rows,
        columns,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, reference_id in zip(axes.ravel(), references, strict=False):
        for method in METHOD_ORDER:
            curve = data.roc.loc[
                data.roc["reference_id"].eq(reference_id)
                & data.roc["method"].eq(method)
            ].sort_values("target_fpr")
            band = data.roc_bands.loc[
                data.roc_bands["reference_id"].eq(reference_id)
                & data.roc_bands["method"].eq(method)
            ].sort_values("target_fpr")
            color = METHOD_COLORS[method]
            axis.fill_between(
                100.0 * band["target_fpr"].to_numpy(float),
                band["ci_lower"].to_numpy(float),
                band["ci_upper"].to_numpy(float),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            axis.step(
                100.0 * curve["target_fpr"].to_numpy(float),
                curve["recall"].to_numpy(float),
                where="post",
                color=color,
                linewidth=METHOD_LINEWIDTHS[method],
                label=method,
            )
        axis.set_title(_reference_label(reference_id, config), loc="left", fontweight="bold")
        axis.set_xlim(0, 5)
        axis.set_ylim(0, 1.02)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=GRID, linewidth=0.5)
    for axis in axes.ravel()[len(references) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Tie-conservative FPR budget (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Recall")
    axes.ravel()[0].legend(frameon=False, ncol=2, loc="lower right")
    container.suptitle(
        "A  Low-FPR retrieval across the complete remote cohort",
        fontweight="bold",
    )


def render_figure3_roc(
    data: Figure3Data,
    output_root: Path,
    config: PipelineConfig,
) -> None:
    configure_style()
    references = _reference_order(data.roc["reference_id"], config)
    columns = min(3, len(references))
    rows = int(np.ceil(len(references) / columns))
    figure = plt.figure(
        figsize=(3.25 * columns, 2.75 * rows),
        constrained_layout=True,
    )
    _draw_figure3_roc(figure, data, config)
    save_figure(figure, output_root / "figure3" / "figure3a_roc")


def _draw_figure3_endpoints(
    container: plt.Figure | plt.SubFigure,
    data: Figure3Data,
    config: PipelineConfig,
) -> None:
    references = _reference_order(data.operating_points["reference_id"], config)
    endpoints = sorted(data.operating_points["target_fpr"].astype(float).unique())
    axes = container.subplots(
        1,
        len(endpoints),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    method_offsets = np.linspace(-0.24, 0.24, len(METHOD_ORDER))
    interval_index = data.endpoint_intervals.set_index(
        ["reference_id", "method", "target_fpr"]
    )
    reference_y = np.arange(len(references), dtype=float)
    for column, endpoint in enumerate(endpoints):
        axis = axes[0, column]
        for offset, method in zip(method_offsets, METHOD_ORDER, strict=True):
            points = data.operating_points.loc[
                data.operating_points["target_fpr"].eq(endpoint)
                & data.operating_points["method"].eq(method)
            ].set_index("reference_id")
            point = 100.0 * np.asarray(
                [points.loc[reference, "recall"] for reference in references],
                dtype=float,
            )
            low = 100.0 * np.asarray(
                [
                    interval_index.loc[(reference, method, endpoint), "ci_lower"]
                    for reference in references
                ],
                dtype=float,
            )
            high = 100.0 * np.asarray(
                [
                    interval_index.loc[(reference, method, endpoint), "ci_upper"]
                    for reference in references
                ],
                dtype=float,
            )
            axis.errorbar(
                point,
                reference_y + offset,
                xerr=np.vstack((point - low, high - point)),
                fmt=METHOD_MARKERS[method],
                markersize=3.2,
                linewidth=0.9,
                capsize=1.5,
                color=METHOD_COLORS[method],
                label=method,
            )
        axis.set_xlim(0, 100)
        axis.set_xticks(
            [0, 25, 50, 75, 100],
            ["0%", "25%", "50%", "75%", "100%"],
        )
        axis.set_xlabel(
            f"Recall at {100 * endpoint:g}% FPR (95% bootstrap CI)"
        )
        axis.set_title(f"{100 * endpoint:g}% FPR", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
    axes[0, 0].set_yticks(
        reference_y,
        [_reference_label(reference, config) for reference in references],
    )
    axes[0, 0].invert_yaxis()
    container.text(0.0, 1.01, "B", ha="left", va="bottom", fontsize=12, fontweight="bold")


def render_figure3_endpoints(
    data: Figure3Data,
    output_root: Path,
    config: PipelineConfig,
) -> None:
    configure_style()
    endpoints = sorted(data.operating_points["target_fpr"].astype(float).unique())
    figure = plt.figure(
        figsize=(8.8, 3.4),
        constrained_layout=True,
    )
    _draw_figure3_endpoints(figure, data, config)
    save_figure(figure, output_root / "figure3" / "figure3b_endpoints")


def render_figure3(data: Figure3Data, output_root: Path, config: PipelineConfig) -> None:
    render_figure3_roc(data, output_root, config)
    render_figure3_endpoints(data, output_root, config)
    configure_style()
    figure = plt.figure(figsize=(10.0, 8.4), constrained_layout=True)
    upper, lower = figure.subfigures(2, 1, height_ratios=[1.35, 0.65])
    _draw_figure3_roc(upper, data, config)
    _draw_figure3_endpoints(lower, data, config)
    save_figure(figure, output_root / "figure3" / "figure3")
