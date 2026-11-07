"""Full/matched score landscapes and ROC/recall panels, using identical rows."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from ..reviewer_analysis import grid_values
from .figure3 import SCORE_COLUMNS
from .common import (configure_style, METHOD_COLORS, METHOD_MARKERS,
                     GENERATOR_COLORS, GENERATOR_LABELS, categorical_colors)
from .confidence_controls import POS, NEG, save


def smoothed_relative_density(values, edges):
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    kernel_x = np.arange(-10, 11)
    kernel = np.exp(-.5 * (kernel_x / 2.4) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(counts.astype(float), kernel, mode="same")
    if density.max(initial=0) > 0:
        density /= density.max()
    return (edges[:-1] + edges[1:]) / 2, density


def score_limits(method, values):
    if method in {"SAE", "ESM-C"}:
        return (0 if values.min() >= 0 else -1), 1
    low = 0 if method == "MMseqs2" else 10 * math.floor(values.min() / 10)
    high = max(low + 10, 10 * math.ceil(values.max() / 10))
    return low, high


def landscape(frame, labels, output, formats=("pdf", "png", "svg")):
    configure_style()
    cols = min(3, len(labels))
    rows = math.ceil(len(labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.65 * cols, 3.4 * rows), squeeze=False,
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.12, top=.85, wspace=.18, hspace=.3)
    colors = categorical_colors(frame.generation_method)
    colors.update(GENERATOR_COLORS)
    for ax, (reference, label) in zip(axes.flat, labels.items()):
        target = frame.loc[frame.reference_id.eq(reference)]
        for generator, group in target.groupby("generation_method", sort=True):
            ax.scatter(100 * group.positional_identity_fraction, group.usalign_tm_score_wt_norm,
                color=colors[generator], s=3.4, alpha=.27, linewidths=0, rasterized=True)
        ax.axhline(.5, color="#999999", lw=.7, ls=":")
        ax.axvline(30, color="#999999", lw=.7, ls="--")
        ax.set(xlim=(0, 100), ylim=(0, 1), title=label)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(labels):]:
        ax.set_visible(False)
    fig.supxlabel("Sequence identity to WT (%)", y=.02)
    fig.supylabel("TM-score to WT", x=.01)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[g], label=GENERATOR_LABELS.get(g, g))
        for g in sorted(frame.generation_method.unique())], loc="upper center", ncol=2, frameon=False)
    save(fig, output / "appendix" / "six_protein_landscapes", formats)


def score_scatters(frame, full, labels, method, *, cutoffs: bool = False, identity: bool = False):
    """Original histogram kernel; full-cohort ranges/edges shared by both cohorts."""
    configure_style()
    columns = min(2, len(labels))
    rows = math.ceil(len(labels) / columns)
    fig = plt.figure(figsize=(6.8 * columns + .65 * (columns - 1), 3.75 * rows))
    grid = fig.add_gridspec(rows, columns, left=.075, right=.985, bottom=.075, top=.89,
                            hspace=.42, wspace=.36)
    score = SCORE_COLUMNS[method]
    for i, (reference, label) in enumerate(labels.items()):
        inner = grid[i // columns, i % columns].subgridspec(1, 2, width_ratios=[5.7, 1.25], wspace=.13)
        ax = fig.add_subplot(inner[0])
        marginal = fig.add_subplot(inner[1], sharey=ax)
        target = frame.loc[frame.reference_id.eq(reference)]
        original = full.loc[full.reference_id.eq(reference)]
        limits = score_limits(method, original[score])
        edges = np.linspace(*limits, 101)
        for flag, color in (("negative", NEG), ("analogue", POS)):
            group = target.loc[target.structural_class.eq(flag)]
            x = 100 * group.positional_identity_fraction if identity else group.usalign_tm_score_wt_norm
            ax.scatter(x, group[score], s=3.8, color=color, alpha=.26, linewidths=0, rasterized=True)
            centers, density = smoothed_relative_density(group[score], edges)
            marginal.fill_betweenx(centers, 0, density, color=color, alpha=.16)
            marginal.plot(density, centers, color=color, lw=1.45)
        ax.set(xlim=(0, 100 if identity else 1), ylim=limits, title=label, ylabel=f"{method} score")
        ax.set_xlabel("Sequence identity to WT (%)" if identity else "TM-score to WT")
        ax.spines[["top", "right"]].set_visible(False)
        marginal.set_xlim(0, 1.06)
        marginal.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        marginal.spines[["top", "right", "bottom"]].set_visible(False)
        marginal.spines["left"].set_color("#E1E6E8")
        marginal.set_xlabel("Relative\ndensity")
        if identity:
            ax.axvline(30, color="#999999", ls=":", lw=.8)
        if cutoffs:
            case = target.loc[target.structural_class.eq("analogue"), score].to_numpy()
            control = target.loc[target.structural_class.eq("negative"), score].to_numpy()
            _, _, thresholds = grid_values(case, control, [.01, .05])
            for threshold, color, style in zip(thresholds, ("#444444", "#287DAF"), (":", "--")):
                # One path also spans the gap between scatter and marginal axes.
                yfig = fig.transFigure.inverted().transform(ax.transData.transform((0, threshold)))[1]
                fig.add_artist(Line2D([ax.get_position().x0, marginal.get_position().x1], [yfig] * 2,
                    transform=fig.transFigure, color=color, ls=style, lw=.8))
    fig.suptitle(method, y=.985, fontsize=12, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=l) for c, l in
        ((POS, "TM-score ≥0.5"), (NEG, "TM-score <0.5"))], loc="upper center",
        bbox_to_anchor=(.53, .96), ncol=2, frameon=False)
    return fig


def render_score_appendix(full, matched, labels, output, formats=("pdf", "png", "svg")):
    root = output / "appendix"
    root.mkdir(parents=True, exist_ok=True)
    with PdfPages(root / "score_distributions_full_and_matched.pdf") as review:
        for name, frame in (("full", full), ("matched", matched)):
            for method in SCORE_COLUMNS:
                slug = {"ESM-C": "esmc", "Pfam": "hmm"}.get(method, method.lower())
                # A2--A5 use the complete all-identity landscape without FPR cutoffs.
                if name == "full":
                    plain = score_scatters(frame, full, labels, method)
                    save(plain, root / f"six_protein_{slug}_tm_scatters", formats)
                fig = score_scatters(frame, full, labels, method, cutoffs=True)
                review.savefig(fig)
                save(fig, root / f"{name}_{slug}_tm_scatters", formats)
        fig = score_scatters(full, full, labels, "Pfam", identity=True)
        save(fig, root / "six_protein_hmm_sequence_identity_scatters", formats)


def roc_panel(ax, result, reference, *, max_fpr=.05):
    for method in SCORE_COLUMNS:
        curve = result.curves.loc[result.curves.reference_id.eq(reference) & result.curves.method.eq(method)]
        curve = curve.loc[curve.observed_fpr.le(max_fpr)]
        x = np.r_[curve.observed_fpr, max_fpr]
        y = np.r_[curve.recall, curve.recall.iloc[-1]]
        band = result.bands.loc[result.bands.reference_id.eq(reference) & result.bands.method.eq(method)
                                & result.bands.target_fpr.le(max_fpr)]
        ax.fill_between(band.target_fpr, band.ci_lower, band.ci_upper,
                        step="post", color=METHOD_COLORS[method], alpha=.115, lw=0)
        ax.step(x, y, where="post", color=METHOD_COLORS[method], lw=1.6, label=method)
    ax.set(xlim=(0, max_fpr), ylim=(0, 1.01), xlabel="False-positive rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)


def render_rocs(results, labels, output, *, max_fpr=.05, stem="figure3_by_generator", formats=("pdf", "png", "svg")):
    configure_style()
    fig, axes = plt.subplots(len(results), len(labels), figsize=(2.55 * len(labels) + .7, 2.7 * len(results) + .7),
        squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.985, top=.87, bottom=.18, hspace=.70, wspace=.25)
    for row, (name, result) in enumerate(results.items()):
        for col, (reference, label) in enumerate(labels.items()):
            ax = axes[row, col]
            roc_panel(ax, result, reference, max_fpr=max_fpr)
            ax.set_title(label, fontweight="bold")
            ax.tick_params(labelbottom=True, labelleft=col == 0)
            if col == 0:
                ax.set_ylabel("Recall")
        axes[row, 0].text(-.03, 1.35, GENERATOR_LABELS.get(name, name), transform=axes[row, 0].transAxes,
                             fontsize=11, fontweight="bold", ha="left")
    fig.legend(handles=[Line2D([], [], color=METHOD_COLORS[m], lw=1.6, label="HMM" if m == "Pfam" else m)
                        for m in SCORE_COLUMNS], loc="lower center", ncol=4, frameon=False)
    save(fig, output / stem, formats)


def render_recall_summaries(results, labels, output, formats=("pdf", "png", "svg")):
    configure_style()
    fig, axes = plt.subplots(2, len(results), figsize=(5.7 * len(results), 8.8), squeeze=False)
    fig.subplots_adjust(left=.1, right=.98, top=.9, bottom=.12, hspace=.35, wspace=.35)
    for col, (generator, result) in enumerate(results.items()):
        for row, q in enumerate((.01, .05)):
            ax = axes[row, col]
            for mi, method in enumerate(SCORE_COLUMNS):
                for ri, reference in enumerate(labels):
                    p = result.endpoints.loc[result.endpoints.reference_id.eq(reference) & result.endpoints.method.eq(method)
                                             & result.endpoints.target_fpr.eq(q)].iloc[0]
                    y = ri + (mi - 1.5) * .15
                    color = METHOD_COLORS[method]
                    ax.plot([p.ci_lower, p.ci_upper], [y, y], color=color, lw=1)
                    ax.scatter(p.recall, y, s=27, color=color, marker=METHOD_MARKERS[method], zorder=3)
            ax.set(yticks=range(len(labels)), yticklabels=labels.values(), ylim=(len(labels) - .5, -.5),
                   xlim=(-.02, 1.02), xlabel="Recall (95% CI)")
            ax.set_title(f"{GENERATOR_LABELS.get(generator, generator)} · {q:.0%} FPR", fontweight="bold")
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], ls="", label=m)
        for m in SCORE_COLUMNS], loc="lower center", ncol=4, frameon=False)
    save(fig, output / "figure3" / "figure3_recall_by_generator", formats)
