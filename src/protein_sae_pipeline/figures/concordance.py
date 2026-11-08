"""Figure AX from complete private score tables, without submitting inference."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .common import configure_style, save_figure

COLORS = {
    "Lambda_gpD": "#B95C19",
    "RamR": "#E69F00",
    "FNIII_10": "#009E73",
    "PDZ3": "#8B62AD",
    "RNaseA": "#6E7074",
    "DHFR": "#0072B2",
}
LABELS = {"Lambda_gpD": "gpD", "FNIII_10": "FNIII-10", "RNaseA": "RNase A"}


def render_concordance(rows, output: Path) -> None:
    """Private figures retain all main-cohort points; ranges are not CIs."""
    configure_style()
    generators = {"esm3": "ESM3", "rfd-mpnn": "RFdiffusion/ProteinMPNN"}
    if set(rows.generator) != set(generators) or not set(rows.family).issubset(COLORS):
        raise ValueError("Paper renderer requires the six-target generator/family vocabulary")
    for endpoint, data in rows.loc[rows.main].groupby("endpoint", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(8, 8.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.12, hspace=0.50, wspace=0.30)
        for i, (generator, title) in enumerate(generators.items()):
            for j, (method, label) in enumerate((("esmfold2", "ESMFold2"), ("colabfold", "AF2"))):
                ax = axes[i, j]
                part = data.loc[data.generator.eq(generator) & data.method.eq(method)]
                for corner in ((0, 0.5), (0.5, 0)):
                    ax.add_patch(Rectangle(corner, 0.5, 0.5, color="#BA2453", alpha=0.045, lw=0))
                ax.plot([0, 1], [0, 1], color="#525B65", lw=0.9)
                ax.axvline(0.5, color="#91979b", lw=0.8, ls="--")
                ax.axhline(0.5, color="#91979b", lw=0.8, ls="--")
                for family, color in COLORS.items():
                    group = part.loc[part.family.eq(family)]
                    ax.errorbar(
                        group.x,
                        group.y,
                        yerr=np.vstack([group.y - group.y_min, group.y_max - group.y]),
                        fmt="none",
                        ecolor=color,
                        alpha=0.23,
                        elinewidth=0.6,
                        capsize=1,
                    )
                    ax.scatter(
                        group.x,
                        group.y,
                        s=14,
                        color=color,
                        edgecolors="white",
                        linewidths=0.2,
                        alpha=0.85,
                    )
                ax.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    aspect="equal",
                    xlabel="ESMFold WT TM-score",
                    ylabel=f"{label} WT TM-score\n(median of 5)",
                )
                ax.tick_params(labelbottom=True)
            fig.text(0.55, 0.97 if i == 0 else 0.51, title, ha="center", weight="bold")
        fig.legend(
            handles=[
                Line2D([], [], marker="o", ls="", color=c, label=LABELS.get(f, f))
                for f, c in COLORS.items()
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
        )
        save_figure(fig, output / f"quadrants-{endpoint}")
