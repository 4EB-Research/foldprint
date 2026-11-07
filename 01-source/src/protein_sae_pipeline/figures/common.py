"""Shared, data-first figure helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io import atomic_csv, atomic_json, sha256_file


KEY_COLUMNS = ("reference_id", "sequence_sha256")
ANALOGUE_COLOR = "#2F9E44"
NEGATIVE_COLOR = "#D87532"
FIGURE1_NEGATIVE_COLOR = "#D87532"
INK = "#000000"
MUTED = "#000000"
GRID = "#E1E6E8"
FOCAL_COLOR = "#287DAF"
GENERATOR_COLORS = {
    "ESM3": "#4085AB",
    "RFD": "#C45A57",
    "fictional_method_a": "#4085AB",
    "fictional_method_b": "#C45A57",
}
GENERATOR_LABELS = {
    "ESM3": "ESM3",
    "RFD": "RFdiffusion + ProteinMPNN",
    "fictional_method_a": "Method 1",
    "fictional_method_b": "Method 2",
}
METHOD_COLORS = {
    "Pfam": "#7A5195",
    "MMseqs2": "#E15759",
    "ESM-C": "#74ADD1",
    "SAE": "#174A70",
}
METHOD_MARKERS = {
    "Pfam": "o",
    "MMseqs2": "X",
    "ESM-C": "s",
    "SAE": "D",
}
METHOD_LINEWIDTHS = {
    "Pfam": 1.65,
    "MMseqs2": 1.45,
    "ESM-C": 1.55,
    "SAE": 2.2,
}
METHOD_ORDER = ("Pfam", "MMseqs2", "ESM-C", "SAE")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "axes.edgecolor": "#8E9BA0",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "text.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")


def require_unique_keys(frame: pd.DataFrame, *, label: str) -> None:
    require_columns(frame, KEY_COLUMNS, label=label)
    if frame[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{label} contains a missing canonical key")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{label} contains duplicate canonical keys")


def exact_key_equality(left: pd.DataFrame, right: pd.DataFrame, *, labels: tuple[str, str]) -> None:
    """Require identical key sets and row order for two plotted panels."""

    require_unique_keys(left, label=labels[0])
    require_unique_keys(right, label=labels[1])
    left_keys = list(left.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None))
    right_keys = list(right.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None))
    if set(left_keys) != set(right_keys):
        missing = len(set(left_keys) - set(right_keys))
        unexpected = len(set(right_keys) - set(left_keys))
        raise ValueError(
            f"{labels[0]}/{labels[1]} key sets differ: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if left_keys != right_keys:
        raise ValueError(f"{labels[0]}/{labels[1]} key order differs")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic CSV/CSV.GZ without materializing record dictionaries."""

    columns = [str(column) for column in frame.columns]
    rows = (
        dict(zip(columns, row, strict=True))
        for row in frame.itertuples(index=False, name=None)
    )
    atomic_csv(path, rows, fieldnames=columns)


def write_panel_bundle(
    data_dir: Path,
    tables: dict[str, pd.DataFrame],
    *,
    metadata: dict[str, Any],
    metadata_name: str,
) -> dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for filename, frame in tables.items():
        path = data_dir / filename
        write_frame(path, frame)
        checksums[filename] = sha256_file(path)
    payload = {**metadata, "plotted_data_sha256": checksums}
    atomic_json(data_dir / metadata_name, payload)
    return checksums


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "protein-sae-analysis-pipeline", "Author": "Anonymous Authors"},
    )
    figure.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Creator": "Anonymous Authors", "Date": None},
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "Anonymous Authors",
            "Author": "Anonymous Authors",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def categorical_colors(values: Iterable[str]) -> dict[str, str]:
    labels = sorted(set(str(value) for value in values))
    palette = plt.get_cmap("tab10")
    return {
        label: matplotlib.colors.to_hex(palette(index % 10))
        for index, label in enumerate(labels)
    }


def finite_numeric(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} contains NaN or infinity")
