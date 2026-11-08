from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.figures.common import (
    ANALOGUE_COLOR,
    FIGURE1_NEGATIVE_COLOR,
    FOCAL_COLOR,
    GENERATOR_COLORS,
    GRID,
    INK,
    METHOD_COLORS,
    METHOD_LINEWIDTHS,
    METHOD_MARKERS,
    METHOD_ORDER,
    NEGATIVE_COLOR,
)
from protein_sae_pipeline.figures.figure3 import _reference_order
ROOT = Path(__file__).resolve().parents[1]


class FigureStyleTests(unittest.TestCase):
    def test_fictional_figure2_inputs_are_tabular_and_self_contained(self) -> None:
        example_root = ROOT / "example"
        examples = pd.read_csv(
            example_root / "inputs" / "figure_examples.csv",
            dtype={"sequence_sha256": str},
        )
        self.assertEqual(
            examples["panel_role"].tolist(),
            ["trace_analogue", "trace_negative"],
        )
        self.assertTrue(examples["reference_id"].eq("fictional_alpha").all())
        self.assertEqual(
            set(examples["required_structural_class"]),
            {"analogue", "negative"},
        )
        self.assertNotIn("inset_image_path", examples.columns)

    def test_palette_matches_configured_figures(self) -> None:
        self.assertEqual(INK, "#000000")
        self.assertEqual(GRID, "#E1E6E8")
        self.assertEqual(ANALOGUE_COLOR, "#2F9E44")
        self.assertEqual(FIGURE1_NEGATIVE_COLOR, "#D87532")
        self.assertEqual(NEGATIVE_COLOR, "#D87532")
        self.assertEqual(FOCAL_COLOR, "#287DAF")
        self.assertEqual(
            GENERATOR_COLORS,
            {
                "ESM3": "#4085AB",
                "RFD": "#C45A57",
                "fictional_method_a": "#4085AB",
                "fictional_method_b": "#C45A57",
            },
        )
        self.assertEqual(
            METHOD_COLORS,
            {
                "Pfam": "#7A5195",
                "MMseqs2": "#E15759",
                "ESM-C": "#74ADD1",
                "SAE": "#174A70",
            },
        )

    def test_method_encodings_match_frozen_figure3(self) -> None:
        self.assertEqual(METHOD_ORDER, ("Pfam", "MMseqs2", "ESM-C", "SAE"))
        self.assertEqual(
            METHOD_MARKERS,
            {"Pfam": "o", "MMseqs2": "X", "ESM-C": "s", "SAE": "D"},
        )
        self.assertEqual(
            METHOD_LINEWIDTHS,
            {"Pfam": 1.65, "MMseqs2": 1.45, "ESM-C": 1.55, "SAE": 2.2},
        )

    def test_figure3_uses_configured_panel_order(self) -> None:
        config = load_config(ROOT / "configs" / "example.yaml")
        shuffled = pd.Series(["fictional_beta", "fictional_alpha"])
        self.assertEqual(
            _reference_order(shuffled, config),
            ["fictional_alpha", "fictional_beta"],
        )


if __name__ == "__main__":
    unittest.main()
