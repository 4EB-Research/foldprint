from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from protein_sae_pipeline.figures.pymol_ribbons import (
    build_focal_structure_script,
    build_pymol_script,
)


class PyMOLRibbonHandoffTests(unittest.TestCase):
    def test_builds_cartoon_from_author_residue_mapping_without_fixing_camera(self) -> None:
        rows = pd.DataFrame(
            {
                "feature_id": [17, 17],
                "sequence_position": [1, 2],
                "author_chain": ["A", "A"],
                "author_residue_number": [20, 21],
                "author_insertion_code": ["", "B"],
                "pymol_color_value_0_1": [0.25, 1.0],
            }
        )
        script = build_pymol_script(
            Path("fictional_wt.pdb"),
            rows,
            Path("feature_17.png"),
        )
        self.assertIn("show cartoon, wt", script)
        self.assertIn("chain A and resi 20", script)
        self.assertIn("chain A and resi 21B", script)
        self.assertIn(
            "set_color feature_colour_1, [0.660000, 0.360000, 0.360000]",
            script,
        )
        self.assertIn("orient wt", script)
        self.assertIn("set orthoscopic, 1", script)
        self.assertIn("turn x, 75", script)
        self.assertIn("dpi=600", script)
        self.assertIn("ray=1", script)

    def test_builds_focal_residue_and_adjacent_strand_from_author_mapping(self) -> None:
        rows = pd.DataFrame(
            {
                "feature_id": [17] * 8,
                "sequence_position": list(range(1, 9)),
                "author_chain": ["A"] * 8,
                "author_residue_number": list(range(101, 109)),
                "author_insertion_code": [""] * 8,
                "pymol_color_value_0_1": [0.0] * 8,
            }
        )
        script = build_focal_structure_script(
            Path("wt.pdb"),
            rows,
            Path("focus.png"),
            focal_position=4,
            window=(2, 6),
            adjacent_strand=(5, 6),
            packing_positions=(7, 8),
        )
        self.assertIn("select focal_residue, (wt and chain A and resi 104)", script)
        self.assertIn("chain A and resi 105", script)
        self.assertIn("show spheres, focal_residue", script)
        self.assertIn("color dark_rose, adjacent_strand", script)
        self.assertIn("zoom closeup_extent, 3.0", script)
        self.assertIn("width=3600, height=3000, dpi=600, ray=1", script)

    def test_rejects_more_than_one_feature_per_render(self) -> None:
        rows = pd.DataFrame(
            {
                "feature_id": [1, 2],
                "sequence_position": [1, 2],
                "author_chain": ["A", "A"],
                "author_residue_number": [1, 2],
                "author_insertion_code": ["", ""],
                "pymol_color_value_0_1": [0.0, 1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "one feature"):
            build_pymol_script(Path("wt.pdb"), rows, Path("out.png"))


if __name__ == "__main__":
    unittest.main()
