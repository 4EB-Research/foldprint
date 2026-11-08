"""Figure 1: landscape, SAE scores, class medians, and distributed WT signature."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..interpretability import (
    feature_contribution_spectrum,
    feature_recurrence,
    rank_wt_features,
)
from ..keys import CandidateKey
from ..representations.store import IndexedRepresentations
from .common import (
    ANALOGUE_COLOR,
    FIGURE1_NEGATIVE_COLOR,
    GENERATOR_COLORS,
    GENERATOR_LABELS,
    GRID,
    categorical_colors,
    configure_style,
    exact_key_equality,
    finite_numeric,
    require_columns,
    require_unique_keys,
    save_figure,
    write_panel_bundle,
)


@dataclass(frozen=True)
class Figure1Data:
    panel_a: pd.DataFrame
    panel_b: pd.DataFrame
    feature_summary: pd.DataFrame
    candidate_audit: pd.DataFrame
    reference_id: str
    wt_feature_count: int


def _panel_frames(
    master: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "manifest_row",
        "reference_id",
        "sequence_sha256",
        "generation_method",
        "positional_identity_fraction",
        "positional_identity_percent",
        "usalign_tm_score_wt_norm",
        "sae_cosine_to_wt",
        "structural_class",
        "is_remote",
    }
    require_columns(master, required, label="master table")
    reference_id = str(config.raw["interpretability"]["reference_id"])
    full = master.loc[master["reference_id"].astype(str).eq(reference_id)].copy()
    full = full.sort_values("manifest_row", kind="stable").reset_index(drop=True)
    expected = int(config.raw["references"][reference_id]["expected_full"])
    if len(full) != expected:
        raise ValueError(f"Figure 1 full target cohort has {len(full)} rows; expected {expected}")
    require_unique_keys(full, label="Figure 1 full cohort")
    finite_numeric(
        full,
        [
            "positional_identity_fraction",
            "positional_identity_percent",
            "usalign_tm_score_wt_norm",
            "sae_cosine_to_wt",
        ],
        label="Figure 1 full cohort",
    )
    if not set(full["structural_class"].astype(str)).issubset({"analogue", "negative"}):
        raise ValueError("Figure 1 contains an unknown structural class")

    common = [
        "reference_id",
        "sequence_sha256",
        "manifest_row",
        "generation_method",
        "positional_identity_fraction",
        "positional_identity_percent",
        "usalign_tm_score_wt_norm",
        "structural_class",
        "is_remote",
    ]
    panel_a = full[common].copy()
    panel_b = full[common + ["sae_cosine_to_wt"]].copy()
    exact_key_equality(panel_a, panel_b, labels=("Figure 1A", "Figure 1B"))
    return panel_a, panel_b


def _contribution_tables(
    panel_b: pd.DataFrame,
    store: IndexedRepresentations,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    reference_id = str(config.raw["interpretability"]["reference_id"])
    remote = panel_b.loc[panel_b["is_remote"].astype(bool)].copy()
    expected_remote = int(config.expected_counts["by_reference"][reference_id]["remote_total"])
    if len(remote) != expected_remote:
        raise ValueError(
            f"Figure 1C remote cohort has {len(remote)} rows; expected {expected_remote}"
        )
    wt = store.wt_profile(reference_id)
    ranking = rank_wt_features(wt)
    feature_ids = ranking.feature_ids.astype(np.int64)
    wt_values = ranking.scores.astype(np.float64)
    if not len(feature_ids):
        raise ValueError("WT SAE profile has no active features")

    summaries: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, object]] = []
    for structural_class in ("analogue", "negative"):
        cohort = remote.loc[remote["structural_class"].eq(structural_class)].copy()
        if cohort.empty:
            raise ValueError(f"Figure 1C lacks structural class {structural_class}")
        profiles = [
            store.candidate_profile(CandidateKey(reference_id, str(row.sequence_sha256)))
            for row in cohort.itertuples(index=False)
        ]
        recurrence = feature_recurrence(profiles, feature_ids)
        cumulative = np.empty((len(profiles), len(feature_ids)), dtype=np.float64)
        cumulative_numerator = np.empty_like(cumulative)
        for row_index, (row, profile) in enumerate(
            zip(cohort.itertuples(index=False), profiles, strict=True)
        ):
            spectrum = feature_contribution_spectrum(
                profile,
                wt,
                feature_order=feature_ids,
            )
            cumulative[row_index] = spectrum.cumulative_cosine_contributions
            cumulative_numerator[row_index] = spectrum.cumulative_numerator_shares
            recorded = float(row.sae_cosine_to_wt)
            error = abs(recorded - spectrum.cosine_similarity)
            if error > 1e-6:
                raise ValueError(
                    "Sparse profiles do not reproduce a recorded Figure 1C SAE score: "
                    f"{row.sequence_sha256}, error={error:.3g}"
                )
            candidate_rows.append(
                {
                    "reference_id": reference_id,
                    "sequence_sha256": str(row.sequence_sha256),
                    "structural_class": structural_class,
                    "recorded_sae_cosine": recorded,
                    "recomputed_sae_cosine": spectrum.cosine_similarity,
                    "absolute_reconstruction_error": error,
                    "candidate_profile_norm": profile.norm,
                    "candidate_profile_nnz": len(profile.indices),
                }
            )

        summary = pd.DataFrame(
            {
                "structural_class": structural_class,
                "wt_feature_rank": np.arange(1, len(feature_ids) + 1),
                "feature_id": feature_ids,
                "wt_normalized_activation": wt_values,
                "active_candidate_count": recurrence.active_counts,
                "candidate_count": recurrence.total,
                "feature_recurrence": recurrence.recurrence,
                "recurrence_wilson_low": recurrence.wilson_low,
                "recurrence_wilson_high": recurrence.wilson_high,
                "cumulative_cosine_median": np.median(cumulative, axis=0),
                "cumulative_cosine_q25": np.percentile(cumulative, 25, axis=0),
                "cumulative_cosine_q75": np.percentile(cumulative, 75, axis=0),
                "cumulative_numerator_share_median": np.median(cumulative_numerator, axis=0),
                "cumulative_numerator_share_q25": np.percentile(cumulative_numerator, 25, axis=0),
                "cumulative_numerator_share_q75": np.percentile(cumulative_numerator, 75, axis=0),
            }
        )
        summaries.append(summary)
        del cumulative, cumulative_numerator
    feature_summary = pd.concat(summaries, ignore_index=True)
    candidate_audit = pd.DataFrame(candidate_rows).sort_values(
        ["structural_class", "sequence_sha256"], kind="stable"
    )
    return feature_summary, candidate_audit.reset_index(drop=True), len(feature_ids)


def prepare_figure1(
    master: pd.DataFrame,
    store: IndexedRepresentations,
    config: PipelineConfig,
) -> Figure1Data:
    panel_a, panel_b = _panel_frames(master, config)
    feature_summary, candidate_audit, feature_count = _contribution_tables(panel_b, store, config)
    return Figure1Data(
        panel_a,
        panel_b,
        feature_summary,
        candidate_audit,
        str(config.raw["interpretability"]["reference_id"]),
        feature_count,
    )


def export_figure1(data: Figure1Data, output_root: Path, config: PipelineConfig) -> None:
    exact_key_equality(data.panel_a, data.panel_b, labels=("Figure 1A", "Figure 1B"))
    data_dir = output_root / "figure1" / "data"
    write_panel_bundle(
        data_dir,
        {
            "figure1a_plot_data.csv.gz": data.panel_a,
            "figure1b_plot_data.csv.gz": data.panel_b,
            "figure1c_class_medians.csv": class_medians(data.panel_b),
            "figure1d_feature_summary.csv.gz": data.feature_summary,
            # Retained as a legacy file alias; the paper now calls this panel D.
            "figure1c_feature_summary.csv.gz": data.feature_summary,
            "figure1c_candidate_score_audit.csv.gz": data.candidate_audit,
        },
        metadata={
            "schema_version": 1,
            "dataset_id": config.dataset_id,
            "reference_id": data.reference_id,
            "figure1a_rows": len(data.panel_a),
            "figure1b_rows": len(data.panel_b),
            "figure1a_equals_figure1b_keys": True,
            "figure1c_source_rows": len(data.panel_b),
            "figure1c_cohort": "complete focal-reference landscape, same rows as 1B",
            "figure1d_rows": len(data.candidate_audit),
            "figure1c_rows": len(data.candidate_audit),
            "wt_active_feature_count": data.wt_feature_count,
            "figure1c_feature_order": (
                "WT normalized max-pooled activation descending; feature ID ascending tie-break"
            ),
            "feature_recurrence": "candidate normalized max-pooled activation > 0",
        },
        metadata_name="figure1_metadata.json",
    )


def _draw_panel_a(axis: plt.Axes, frame: pd.DataFrame) -> None:
    colors = categorical_colors(frame["generation_method"].astype(str))
    colors.update(GENERATOR_COLORS)
    for method, group in frame.groupby("generation_method", sort=True, observed=True):
        method = str(method)
        axis.scatter(
            100.0 * group["positional_identity_fraction"],
            group["usalign_tm_score_wt_norm"],
            s=4,
            alpha=0.30,
            linewidth=0,
            color=colors[method],
            label=GENERATOR_LABELS.get(method, method),
            rasterized=True,
        )
    axis.axhline(0.5, color=GRID, linewidth=0.8, linestyle="--")
    axis.axvline(30.0, color=GRID, linewidth=0.8, linestyle="--")
    axis.set(xlabel="Positional identity to WT (%)", ylabel="WT-normalized TM-score")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("A  Complete design landscape", loc="left", fontweight="bold")


def _draw_panel_b(axis: plt.Axes, frame: pd.DataFrame) -> None:
    styles = (
        ("analogue", ANALOGUE_COLOR, "TM-score ≥0.5"),
        ("negative", FIGURE1_NEGATIVE_COLOR, "TM-score <0.5"),
    )
    for label, color, display_label in styles:
        group = frame.loc[frame["structural_class"].eq(label)]
        axis.scatter(
            group["usalign_tm_score_wt_norm"],
            group["sae_cosine_to_wt"],
            s=4,
            alpha=0.30,
            linewidth=0,
            color=color,
            label=display_label,
            rasterized=True,
        )
    axis.axvline(0.5, color=GRID, linewidth=0.8, linestyle="--")
    axis.set(xlabel="WT-normalized TM-score", ylabel="SAE cosine to WT")
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("B  Sparse-profile similarity", loc="left", fontweight="bold")


def _draw_panel_d(axis: plt.Axes, strip: plt.Axes, summary: pd.DataFrame) -> None:
    class_styles = (
        ("analogue", ANALOGUE_COLOR, "TM-score ≥0.5"),
        ("negative", FIGURE1_NEGATIVE_COLOR, "TM-score <0.5"),
    )
    for label, color, display_label in class_styles:
        group = summary.loc[summary["structural_class"].eq(label)].sort_values("wt_feature_rank")
        x = group["wt_feature_rank"].to_numpy(float)
        median = group["cumulative_cosine_median"].to_numpy(float)
        q25 = group["cumulative_cosine_q25"].to_numpy(float)
        q75 = group["cumulative_cosine_q75"].to_numpy(float)
        axis.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
        axis.plot(x, median, color=color, linewidth=1.5, label=display_label)
    axis.set_ylabel("Cumulative contribution to SAE cosine")
    axis.set_xlim(1, int(summary["wt_feature_rank"].max()))
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("D  Distributed WT-feature signature", loc="left", fontweight="bold")
    maximum_rank = int(summary["wt_feature_rank"].max())
    for row, (label, color, _display_label) in enumerate(class_styles):
        recurrence = (
            summary.loc[summary["structural_class"].eq(label)]
            .sort_values("wt_feature_rank")["feature_recurrence"]
            .to_numpy(float)
        )
        colormap = mcolors.LinearSegmentedColormap.from_list(
            f"white_to_{label}", ["#FFFFFF", color]
        )
        strip.imshow(
            recurrence[np.newaxis, :],
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=0,
            vmax=1,
            extent=(0.5, maximum_rank + 0.5, row + 0.5, row - 0.5),
        )
    strip.set_ylim(1.5, -0.5)
    strip.set_yticks([0, 1], ["TM-score ≥0.5", "TM-score <0.5"])
    strip.set_xlabel("WT feature rank")
    strip.tick_params(axis="y", length=0)
    strip.spines[:].set_visible(False)


def class_medians(frame: pd.DataFrame) -> pd.DataFrame:
    """Panel C is derived from exactly the same complete cohort as panel B."""
    return (
        frame.groupby("structural_class", sort=True)
        .agg(n=("sae_cosine_to_wt", "size"), median_sae_cosine=("sae_cosine_to_wt", "median"))
        .reset_index()
    )


def _draw_class_medians(axis: plt.Axes, frame: pd.DataFrame) -> None:
    summary = class_medians(frame).set_index("structural_class")
    order = ["analogue", "negative"]
    values = summary.loc[order, "median_sae_cosine"].to_numpy()
    axis.bar([0, 1], values, color=[ANALOGUE_COLOR, FIGURE1_NEGATIVE_COLOR], width=0.55)
    for i, value in enumerate(values):
        axis.text(i, value + 0.025, f"{value:.3f}", ha="center", va="bottom")
    axis.set(
        xticks=[0, 1],
        xticklabels=["TM-score ≥0.5", "TM-score <0.5"],
        ylim=(0, 1.1),
        ylabel="SAE cosine to WT",
    )
    axis.set_title("C  Median SAE similarity", loc="left", fontweight="bold")


def _draw_density(axis: plt.Axes, frame: pd.DataFrame) -> None:
    # Same histogram smoothing used for the paper's score landscapes.
    from .reviewer_appendix import smoothed_relative_density

    for label, color in (("analogue", ANALOGUE_COLOR), ("negative", FIGURE1_NEGATIVE_COLOR)):
        values = frame.loc[frame.structural_class.eq(label), "sae_cosine_to_wt"]
        centers, density = smoothed_relative_density(values, np.linspace(0, 1, 101))
        axis.fill_betweenx(centers, 0, density, color=color, alpha=0.16)
        axis.plot(density, centers, color=color, linewidth=1)
    axis.set(xlim=(0, 1.05), ylim=(0, 1.02), xlabel="Relative\ndensity")
    axis.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axis.spines[["top", "right", "bottom"]].set_visible(False)


def render_figure1(data: Figure1Data, output_root: Path) -> None:
    configure_style()
    figure = plt.figure(figsize=(10.2, 6.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1, 1.3])
    axis_a = figure.add_subplot(grid[0, 0])
    upper = grid[0, 1].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.08)
    axis_b = figure.add_subplot(upper[0])
    density = figure.add_subplot(upper[1], sharey=axis_b)
    axis_c = figure.add_subplot(grid[1, 0])
    lower = grid[1, 1].subgridspec(2, 1, height_ratios=[4.5, 1.0], hspace=0.05)
    axis_d = figure.add_subplot(lower[0])
    strip = figure.add_subplot(lower[1], sharex=axis_d)
    _draw_panel_a(axis_a, data.panel_a)
    _draw_panel_b(axis_b, data.panel_b)
    _draw_density(density, data.panel_b)
    _draw_class_medians(axis_c, data.panel_b)
    _draw_panel_d(axis_d, strip, data.feature_summary)
    axis_d.tick_params(axis="x", labelbottom=False)
    for axis in (axis_a, axis_b, axis_c, axis_d):
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output_root / "figure1" / "figure1")
    for label, drawer, frame in (
        ("figure1a", _draw_panel_a, data.panel_a),
        ("figure1b", _draw_panel_b, data.panel_b),
        ("figure1c", _draw_class_medians, data.panel_b),
    ):
        standalone, axis = plt.subplots(figsize=(4.2, 3.3), constrained_layout=True)
        drawer(axis, frame)
        axis.spines[["top", "right"]].set_visible(False)
        save_figure(standalone, output_root / "figure1" / label)
    standalone = plt.figure(figsize=(5.5, 3.4), constrained_layout=True)
    lower = standalone.add_gridspec(2, 1, height_ratios=[4.5, 1.0], hspace=0.05)
    axis = standalone.add_subplot(lower[0])
    strip_axis = standalone.add_subplot(lower[1], sharex=axis)
    _draw_panel_d(axis, strip_axis, data.feature_summary)
    axis.tick_params(axis="x", labelbottom=False)
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(standalone, output_root / "figure1" / "figure1d")
