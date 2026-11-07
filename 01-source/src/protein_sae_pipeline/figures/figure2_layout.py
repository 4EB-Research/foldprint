"""Five-panel Figure 2 renderer.

The scientific tables are prepared in :mod:`figure2`; this module maps those
computed values and same-run, code-generated PyMOL renders onto the intended
combined layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    ANALOGUE_COLOR,
    FOCAL_COLOR,
    FIGURE1_NEGATIVE_COLOR,
    GRID,
    INK,
    MUTED,
    configure_style,
    save_figure,
)


PALE_BLUE = "#E5F1F7"
TRACE_FILL = "#E5E5E5"


def _panel_letter(container, letter: str) -> None:
    container.text(0.0, 1.01, letter, ha="left", va="bottom", fontsize=12, fontweight="bold")


def draw_panel_a(container, data, structure_root) -> None:
    _panel_letter(container, "A")
    window_start, window_end = data.panel_b_window
    if data.structural_panels_included:
        grid = container.add_gridspec(
            5, 2, left=0.08, right=0.99, bottom=0.14, top=0.94,
            width_ratios=[4.0, 1.0], hspace=0.42, wspace=0.08,
        )
    else:
        grid = container.add_gridspec(
            5, 1, left=0.08, right=0.99, bottom=0.14, top=0.94,
            hspace=0.42,
        )
    for row, feature_id in enumerate(data.panel_a_feature_ids):
        trace = data.panel_a_traces.loc[data.panel_a_traces["feature_id"].eq(feature_id)]
        x = trace["sequence_position"].to_numpy(float)
        y = np.clip(trace["normalized_activation"].to_numpy(float), 0, 1)
        axis = container.add_subplot(grid[row, 0])
        if row == 0:
            axis.axvspan(window_start - 0.5, window_end + 0.5, color=PALE_BLUE, zorder=0)
        axis.fill_between(x, 0, y, color=TRACE_FILL, linewidth=0, zorder=1)
        axis.plot(x, y, color=INK, linewidth=0.9, zorder=2)
        axis.set(xlim=(1, data.sequence_length), ylim=(-0.02, 1.02))
        axis.set_yticks([0, 1])
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(f"Feature {feature_id}", loc="left", fontsize=7.2, fontweight="bold", pad=2)
        if row < 4:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xticks([1, 20, 40, 60, 80, data.sequence_length])
            axis.set_xlabel("Residue position", labelpad=1)
        if row == 0:
            axis.annotate(
                "Expanded in B",
                xy=(window_end, 0.92), xycoords="data",
                xytext=(0.33, 1.28), textcoords="axes fraction",
                color=FOCAL_COLOR, fontsize=5.6, fontweight="bold",
                ha="center", va="center", clip_on=False,
                bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": FOCAL_COLOR, "lw": 0.7},
                arrowprops={"arrowstyle": "-", "color": FOCAL_COLOR, "lw": 0.7},
            )
        if data.structural_panels_included:
            image_axis = container.add_subplot(grid[row, 1])
            image_path = structure_root / f"feature_{feature_id}.png"
            if image_path.is_file():
                image_axis.imshow(plt.imread(image_path))
            image_axis.set_axis_off()
    container.text(0.015, 0.53, "Feature activation", rotation=90, ha="center", va="center", fontsize=7)


def _role_color(role: str, structural_class: str) -> str:
    if role == "WT":
        return INK
    return FIGURE1_NEGATIVE_COLOR if structural_class == "negative" else ANALOGUE_COLOR


def draw_panel_b(container, data) -> None:
    _panel_letter(container, "B")
    window_start, window_end = data.panel_b_window
    container.text(
        0.52, 1.01, f"Feature {data.panel_b_feature_id}, residues {window_start}–{window_end}",
        ha="center", va="bottom", fontsize=8.2, fontweight="bold", fontstyle="italic",
        color=FOCAL_COLOR,
    )
    traces = data.panel_b_traces.loc[
        data.panel_b_traces["sequence_position"].between(window_start, window_end)
    ]
    groups = list(traces.groupby(["example_order", "panel_role"], sort=True, observed=True))
    grid = container.add_gridspec(
        len(groups), 2, left=0.12, right=0.98, bottom=0.15, top=0.91,
        width_ratios=[4.0, 1.15], hspace=0.38, wspace=0.05,
    )
    summaries = data.panel_b_examples.set_index("panel_role") if not data.panel_b_examples.empty else None
    for row, ((_order, role), group) in enumerate(groups):
        group = group.sort_values("sequence_position")
        cell = grid[row, 0].subgridspec(2, 1, height_ratios=[0.46, 1.0], hspace=0.02)
        letters = container.add_subplot(cell[0, 0])
        axis = container.add_subplot(cell[1, 0])
        label_axis = container.add_subplot(grid[row, 1])
        letters.set_axis_off()
        label_axis.set_axis_off()
        structural_class = str(group.iloc[0]["structural_class"])
        color = _role_color(str(role), structural_class)
        x = group["sequence_position"].to_numpy(float)
        y = np.clip(group["normalized_activation"].to_numpy(float), 0, 1)
        axis.fill_between(x, 0, y, color=color, alpha=0.10, linewidth=0)
        axis.plot(x, y, color=color, linewidth=0.9)
        focal = group.loc[group["is_focal_position"].astype(bool)]
        axis.scatter(focal["sequence_position"], np.clip(focal["normalized_activation"], 0, 1),
                     s=8, color=FOCAL_COLOR, edgecolor="white", linewidth=0.3, zorder=4)
        letters.set(xlim=(window_start - 0.45, window_end + 0.45), ylim=(0, 1))
        for record in group.itertuples(index=False):
            letters.text(
                int(record.sequence_position), 0.45, str(record.residue),
                ha="center", va="center", fontsize=5.8, family="DejaVu Sans Mono",
                color="#C84E39" if bool(record.is_substitution) else INK,
            )
        axis.set(xlim=(window_start - 0.45, window_end + 0.45), ylim=(-0.03, 1.02))
        axis.set_yticks([0, 1])
        axis.spines[["top", "right"]].set_visible(False)
        if row < len(groups) - 1:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xticks(np.linspace(window_start, window_end, 7, dtype=int))
            axis.set_xlabel("Residue position", labelpad=1)
        if str(role) == "WT":
            label = "WT focal protein"
        else:
            summary = summaries.loc[str(role)]
            label = (
                f"Sequence identity: {100 * float(summary['positional_identity_fraction']):.1f}%\n"
            )
            if structural_class == "negative" and "local_identity_fraction" in summary:
                label += f"Local ID: {100 * float(summary['local_identity_fraction']):.1f}%\n"
            label += f"TM-score {float(summary['usalign_tm_score_wt_norm']):.3f}"
        label_axis.text(0, 0.45, label, ha="left", va="center", fontsize=5.6,
                        color=color, fontweight="bold", linespacing=1.1)
    container.text(0.025, 0.52, f"Feature {data.panel_b_feature_id} activation",
                   rotation=90, ha="center", va="center", fontsize=7)


def draw_panel_c(container, data) -> None:
    _panel_letter(container, "C")
    grid = container.add_gridspec(
        4, 2, left=0.10, right=0.99, bottom=0.16, top=0.91,
        width_ratios=[3.6, 1.25], hspace=0.45, wspace=0.12,
    )
    ymax = max(0.45, float(data.panel_c_traces["normalized_activation"].max()))
    for row, feature_id in enumerate(data.panel_c_feature_ids):
        trace = data.panel_c_traces.loc[data.panel_c_traces["feature_id"].eq(feature_id)]
        axis = container.add_subplot(grid[row, 0])
        x = trace["sequence_position"].to_numpy(float)
        y = trace["normalized_activation"].to_numpy(float)
        axis.fill_between(x, 0, y, color=TRACE_FILL, linewidth=0)
        axis.plot(x, y, color=INK, linewidth=0.8)
        focal = trace.loc[trace["is_focal_position"].astype(bool)]
        axis.scatter(focal["sequence_position"], focal["normalized_activation"],
                     color=FOCAL_COLOR, edgecolor="white", linewidth=0.3, s=9, zorder=4)
        axis.set(xlim=(1, data.sequence_length), ylim=(-0.01, ymax))
        axis.set_yticks([0, ymax])
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(f"Feature {feature_id}", loc="left", fontsize=6.8, fontweight="bold", pad=2)
        if row < 3:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xticks([1, 20, 40, 60, 80, data.sequence_length])
            axis.set_xlabel("Residue position", labelpad=1)
        prevalence = data.panel_c_prevalence.loc[
            data.panel_c_prevalence["feature_id"].eq(feature_id)
        ].set_index("structural_class")
        bars_axis = container.add_subplot(grid[row, 1])
        values = [100 * prevalence.loc["analogue", "prevalence"],
                  100 * prevalence.loc["negative", "prevalence"]]
        bars = bars_axis.bar([0, 1], values, width=0.62,
                             color=[ANALOGUE_COLOR, FIGURE1_NEGATIVE_COLOR])
        for bar, value in zip(bars, values, strict=True):
            bars_axis.text(bar.get_x() + bar.get_width()/2, value + 2, f"{value:.1f}%",
                           ha="center", va="bottom", fontsize=5.2, fontweight="bold")
        bars_axis.set(xlim=(-0.55, 1.55), ylim=(0, 112))
        bars_axis.set_yticks([])
        bars_axis.axhline(100, color=GRID, linewidth=0.5)
        bars_axis.spines[["top", "right", "left"]].set_visible(False)
        if row < 3:
            bars_axis.set_xticks([])
        else:
            bars_axis.set_xticks([0, 1], ["Analogue", "Negative"], rotation=30, ha="right")
    container.text(0.015, 0.52, "Feature activation", rotation=90,
                   ha="center", va="center", fontsize=7)
    container.text(0.78, 0.94, f"Position-{data.focal_position} prevalence", ha="center",
                   va="bottom", fontsize=5.5, fontweight="bold")


def draw_panel_d(axis, data) -> None:
    axis.set_title("D", loc="left", fontsize=12, fontweight="bold", pad=6)
    frame = data.panel_d_position_unique.sort_values("sequence_position")
    focal = frame.loc[frame["is_focal_position"].astype(bool)].iloc[0]
    other = frame.loc[~frame["is_focal_position"].astype(bool)]
    x = frame["sequence_position"].to_numpy(float)
    y = frame["position_unique_feature_count"].to_numpy(float)
    axis.vlines(x, 0, y, color="#D6DDE0", linewidth=0.65)
    axis.scatter(x, y, s=5, color="#7C8A90", linewidth=0)
    fx, fy = float(focal["sequence_position"]), float(focal["position_unique_feature_count"])
    axis.vlines(fx, 0, fy, color=FOCAL_COLOR, linewidth=1.2)
    axis.scatter([fx], [fy], s=15, color=FOCAL_COLOR, edgecolor="white", linewidth=0.4, zorder=4)
    mean = float(other["position_unique_feature_count"].mean())
    axis.axhline(mean, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.7,
                 label=f"other position mean = {mean:.2f}")
    axis.annotate(f"Position {int(fx)}", xy=(fx, fy), xytext=(fx + 7, fy + 1.4),
                  color=FOCAL_COLOR, fontsize=6.3,
                  arrowprops={"arrowstyle": "-", "color": FOCAL_COLOR, "lw": 0.7})
    axis.set(xlim=(0, data.sequence_length + 1), ylim=(0, max(32, fy + 3)))
    axis.set_xticks([1, 20, 40, 60, 80, data.sequence_length])
    axis.set_xlabel("Residue position")
    axis.set_ylabel("")
    axis.text(
        0.025, 0.50, "Position-unique active features",
        transform=axis.transAxes, rotation=90, ha="left", va="center",
        fontsize=5.7,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.legend(frameon=False, loc="upper right", fontsize=5.2)


def draw_panel_e(axis, data, structure_root) -> None:
    axis.set_title("E", loc="left", fontsize=12, fontweight="bold", pad=6)
    image_path = structure_root / "focal_structure.png"
    if image_path.is_file():
        axis.imshow(plt.imread(image_path))
    axis.set_axis_off()


def render_figure2_layout(data, output_root: Path) -> None:
    configure_style()
    structure_root = output_root / "figure2" / "structure_renders"
    figure = plt.figure(figsize=(10.1, 5.65))
    outer = figure.add_gridspec(
        2, 12, left=0.025, right=0.99, top=0.97, bottom=0.07,
        height_ratios=[1.05, 0.82], hspace=0.22, wspace=0.25,
    )
    draw_panel_a(figure.add_subfigure(outer[0, :6]), data, structure_root)
    draw_panel_b(figure.add_subfigure(outer[0, 6:]), data)
    draw_panel_c(figure.add_subfigure(outer[1, :6]), data)
    if data.structural_panels_included:
        draw_panel_d(figure.add_subplot(outer[1, 6:9]), data)
        draw_panel_e(figure.add_subplot(outer[1, 9:]), data, structure_root)
    else:
        draw_panel_d(figure.add_subplot(outer[1, 6:]), data)
    save_figure(figure, output_root / "figure2" / "figure2")

    for name, size, drawer in (
        (
            "figure2a",
            (5.1, 3.3),
            lambda container, datum: draw_panel_a(container, datum, structure_root),
        ),
        ("figure2b", (5.1, 3.3), draw_panel_b),
        ("figure2c", (5.1, 3.3), draw_panel_c),
    ):
        standalone = plt.figure(figsize=size)
        drawer(standalone, data)
        save_figure(standalone, output_root / "figure2" / name)
    standalone_panels = [("figure2d", draw_panel_d)]
    if data.structural_panels_included:
        standalone_panels.append(
            ("figure2e", lambda axis, datum: draw_panel_e(axis, datum, structure_root))
        )
    for name, drawer in standalone_panels:
        standalone, axis = plt.subplots(figsize=(3.6, 3.0))
        drawer(axis, data)
        save_figure(standalone, output_root / "figure2" / name)
