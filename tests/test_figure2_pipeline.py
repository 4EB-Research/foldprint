from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from figure_fixtures import fictional_context
from protein_sae_pipeline.figures.figure2 import (
    export_figure2,
    prepare_figure2,
    render_figure2,
)


class Figure2PipelineTests(unittest.TestCase):
    def test_sparse_profiles_pdb_schematic_and_examples_share_inputs(self) -> None:
        config, inputs, store, master = fictional_context()
        data = prepare_figure2(master.copy(), store, inputs, config)
        self.assertEqual(data.reference_id, "fictional_alpha")
        self.assertEqual(data.focal_position, 20)
        self.assertEqual(len(data.panel_a_feature_ids), 5)
        self.assertEqual(len(data.panel_c_feature_ids), 4)
        self.assertEqual(
            set(data.panel_b_examples["panel_role"]),
            {"trace_analogue", "trace_negative"},
        )
        self.assertEqual(
            set(data.panel_a_structure["feature_id"]),
            set(data.panel_a_feature_ids),
        )
        self.assertTrue(
            data.panel_a_structure[["projected_x", "projected_y"]].notna().all().all()
        )
        self.assertTrue(
            data.panel_a_structure[
                ["pymol_selection", "pymol_color_value_0_1"]
            ]
            .notna()
            .all()
            .all()
        )
        self.assertTrue(
            data.panel_a_structure["pymol_color_value_0_1"].between(0, 1).all()
        )
        self.assertEqual(len(data.position_unique_features), data.sequence_length)
        self.assertEqual(
            int(data.position_unique_features["is_focal_position"].sum()),
            1,
        )
        self.assertTrue(
            data.position_unique_features["empirical_greater_equal_tail_p"]
            .between(0, 1)
            .all()
        )
        self.assertEqual(
            set(data.panel_b_residue_composition["structural_class"]),
            {"analogue", "negative"},
        )
        composition_sums = data.panel_b_residue_composition.groupby(
            "structural_class", observed=True
        )["fraction"].sum()
        self.assertTrue(composition_sums.round(12).eq(1.0).all())
        self.assertEqual(
            set(data.feature_selection_audit["wt_pooled_rank"]),
            set(range(1, len(data.feature_selection_audit) + 1)),
        )
        self.assertNotIn("atlas_label", data.feature_selection_audit)
        self.assertNotIn("atlas_category", data.feature_selection_audit)

    def test_exports_all_panels_and_renders_offline(self) -> None:
        config, inputs, store, master = fictional_context()
        data = prepare_figure2(master.copy(), store, inputs, config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_figure2(data, root, config, pymol_available=False)
            with patch(
                "protein_sae_pipeline.figures.figure2._draw_ca_schematic",
                side_effect=AssertionError("2D structural fallback was used"),
            ):
                render_figure2(data, root)
            metadata = json.loads(
                (root / "figure2" / "data" / "figure2_metadata.json").read_text()
            )
            self.assertEqual(
                metadata["main_figure_panels"]["B"],
                f"feature {data.panel_b_feature_id} residues "
                f"{data.panel_b_window[0]}-{data.panel_b_window[1]} in configured examples",
            )
            self.assertIn(
                "complete focal-reference remote cohort",
                metadata["main_figure_panels"]["C"],
            )
            self.assertEqual(
                metadata["main_figure_panels"]["E"],
                "structural image blank because PyMOL is unavailable",
            )
            self.assertEqual(
                metadata["wt_structural_inset"]["representation"],
                "blank when PyMOL renders are unavailable",
            )
            expected = [
                root / "figure2" / "data" / "figure2a_wt_feature_traces.csv.gz",
                root
                / "figure2"
                / "data"
                / "figure2a_wt_pdb_residue_colouring.csv.gz",
                root / "figure2" / "data" / "figure2b_example_traces.csv.gz",
                root / "figure2" / "data" / "figure2c_position_prevalence.csv",
                root / "figure2" / "data" / "figure2_position_unique_features.csv",
                root / "figure2" / "data" / "figure2d_position_unique_plot_data.csv",
                root
                / "figure2"
                / "data"
                / "figure2_feature_selection_audit.csv.gz",
                root / "figure2" / "figure2.png",
                root / "figure2" / "figure2a.png",
                root / "figure2" / "figure2b.png",
                root / "figure2" / "figure2c.png",
                root / "figure2" / "figure2d.png",
                root / "figure2" / "figure2e.png",
            ]
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in expected))

    def test_structure_free_mode_renders_numeric_panels_only(self) -> None:
        config, inputs, store, master = fictional_context()
        data = prepare_figure2(
            master.copy(),
            store,
            inputs,
            config,
            include_structural_panels=False,
        )
        self.assertFalse(data.structural_panels_included)
        self.assertTrue(data.panel_a_structure.empty)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_figure2(data, root, config)
            render_figure2(data, root)
            for name in ("figure2", "figure2a", "figure2b", "figure2c", "figure2d"):
                self.assertTrue((root / "figure2" / f"{name}.png").is_file())
            self.assertFalse((root / "figure2" / "figure2e.png").exists())
            self.assertFalse(
                (
                    root
                    / "figure2"
                    / "data"
                    / "figure2a_wt_pdb_residue_colouring.csv.gz"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
