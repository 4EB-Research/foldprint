"""Paper Figure 4 and IUPred appendix, adapted from the original sieve plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .common import configure_style

POS, NEG = "#2F9E44", "#CA6D39"
METHODS = [("SAE", "#174A70"), ("ESM-C", "#74ADD1"), ("Pfam", "#7A5195"), ("MMseqs2", "#E15759")]


def style():
    configure_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "xtick.labelsize": 9, "ytick.labelsize": 9})


def save(fig, stem: Path, formats=("pdf", "png", "svg")) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(fig)


def class_handles():
    return [Patch(facecolor=POS, label="Structural analog\n(TM-score ≥ 0.5)"),
            Patch(facecolor=NEG, label="Structural negative\n(TM-score < 0.5)")]


def render_confidence_controls(remote, matched, points, *, labels: dict[str, str], output: Path,
                               formats=("pdf", "png", "svg")) -> None:
    style()
    order = list(labels)
    fig, axes = plt.subplots(2, len(order), figsize=(max(14.2, 2.05 * len(order) + 1.9), 5.1),
        sharey="row", squeeze=False, gridspec_kw=dict(left=.07, right=.79, top=.94,
        bottom=.12, wspace=.22, hspace=.44, height_ratios=[1.45, 1]))
    rng = np.random.default_rng(20261104)
    for col, reference in enumerate(order):
        ax = axes[0, col]
        for x, frame in enumerate((remote, matched)):
            target = frame.loc[frame.reference_id.eq(reference)]
            for flag, shift, color in (("analogue", -.14, POS), ("negative", .14, NEG)):
                values = target.loc[target.structural_class.eq(flag), "predicted_ptm"].to_numpy()
                jitter = rng.uniform(-.105, .105, len(values))
                ax.scatter(x + shift + jitter, values, s=2.4, alpha=.14 if x == 0 else .25, color=color,
                           linewidths=0, rasterized=True)
                q25, median, q75 = np.quantile(values, [.25, .5, .75])
                ax.plot([x + shift] * 2, [q25, q75], color="white", linewidth=4, zorder=4)
                ax.plot([x + shift] * 2, [q25, q75], color=color, linewidth=1.8, zorder=5)
                ax.plot([x + shift - .07, x + shift + .07], [median] * 2, color="black", lw=1.4, zorder=6)
        ax.axhline(.5, color="#9A9A9A", ls=(0, (3, 3)), lw=.8, zorder=0)
        ax.set(xlim=(-.48, 1.48), ylim=(-.02, 1.02), xticks=[0, 1],
               xticklabels=["Full", "Matched\nsubsample"], yticks=[0, .25, .5, .75, 1], title=labels[reference])
        ax = axes[1, col]
        for y, (method, color) in enumerate(METHODS):
            rows = points.loc[points.reference_id.eq(reference) & points.method.eq(method) & points.target_fpr.eq(.01)]
            before = 100 * rows.loc[rows.cohort.eq("remote"), "recall"].item()
            after = 100 * rows.loc[rows.cohort.eq("matched"), "recall"].item()
            ax.plot([before, after], [y, y], color=color, lw=1.5, alpha=.65)
            ax.scatter(before, y, facecolors="white", edgecolors=color, s=45, linewidths=1.4, zorder=3)
            ax.scatter(after, y, color=color, s=31, linewidths=.8, zorder=4)
        ax.set(yticks=np.arange(4), yticklabels=["HMM" if m == "Pfam" else m for m, _ in METHODS],
               ylim=(3.5, -.5), xlim=(-3, 103), xticks=[0, 25, 50, 75, 100])
        ax.set_xlabel("Recall at ≤1% FPR (%)", fontsize=8)
        ax.tick_params(labelleft=col == 0, labelbottom=True, labelsize=8)
        for text, (_, color) in zip(ax.get_yticklabels(), METHODS):
            text.set_color(color)
    axes[0, 0].set_ylabel("pTM")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=class_handles(), title="Structural class", loc="upper left",
        bbox_to_anchor=(.815, .95), frameon=False, fontsize=9, title_fontsize=9)
    fig.legend(handles=[Line2D([], [], marker="o", mfc="white", mec="#555555", ls="", label="Full"),
        Line2D([], [], marker="o", color="#555555", ls="", label="Matched subsample")],
        title="Cohort", loc="upper left", bbox_to_anchor=(.815, .40), frameon=False, fontsize=9, title_fontsize=9)
    save(fig, output / "figure4" / "combined_ptm_recall_only", formats)


def render_iupred(scores: pd.DataFrame, *, labels: dict[str, str], output: Path,
                  formats=("pdf", "png", "svg")) -> None:
    style()
    fig, axes = plt.subplots(1, len(labels), figsize=(max(14.2, 2.05 * len(labels) + 1.9), 3.8),
        sharey=True, squeeze=False, gridspec_kw=dict(left=.07, right=.79, top=.89, bottom=.14, wspace=.22))
    for ax, (reference, label) in zip(axes.flat, labels.items()):
        values = [scores.loc[scores.reference_id.eq(reference) & scores.structural_class.eq(flag),
                             "iupred_long_mean"].to_numpy() for flag in ("analogue", "negative")]
        # Matplotlib's KDE is singular for a singleton/constant fictional class.
        # Render its exact value as a line; never perturb the underlying scores.
        for x, (array, color) in enumerate(zip(values, (POS, NEG))):
            if len(array) > 1 and np.ptp(array) > 0:
                violin = ax.violinplot([array], positions=[x], widths=.75, points=200,
                    bw_method="scott", showextrema=False)
                violin["bodies"][0].set(facecolor=color, edgecolor=color, alpha=.55, linewidth=.8)
            else:
                ax.plot([x - .2, x + .2], [array[0]] * 2, color=color)
            q25, median, q75 = np.quantile(array, [.25, .5, .75])
            ax.plot([x, x], [q25, q75], color="black", lw=1.1)
            ax.plot([x - .12, x + .12], [median, median], color="black", lw=1.1)
            ax.scatter(x, array.mean(), marker="D", s=18, color="black", edgecolors="white", linewidths=.4, zorder=5)
        ax.set(xlim=(-.65, 1.65), ylim=(0, 1), xticks=[], yticks=[0, .25, .5, .75, 1], title=label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("IUPred3 score")
    fig.legend(handles=class_handles(), title="Structural class", loc="upper left",
        bbox_to_anchor=(.815, .90), frameon=False, fontsize=9, title_fontsize=9)
    fig.legend(handles=[Line2D([], [], color="black", lw=1.1, label="Median"),
        Line2D([], [], color="black", ls="", marker="|", markersize=12, label="Interquartile range"),
        Line2D([], [], color="black", ls="", marker="D", markersize=4, label="Mean")],
        title="Summary", loc="upper left", bbox_to_anchor=(.815, .46), frameon=False, fontsize=9, title_fontsize=9)
    save(fig, output / "appendix" / "iupred_vertical_violins", formats)
