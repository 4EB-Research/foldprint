from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from figure_fixtures import fictional_context
from protein_sae_pipeline.figures.common import exact_key_equality
from protein_sae_pipeline.figures.figure1 import (
    export_figure1,
    prepare_figure1,
    render_figure1,
)


class Figure1PipelineTests(unittest.TestCase):
    def test_full_panels_share_exact_keys_and_remote_signature_reconstructs_scores(self) -> None:
        config, _inputs, store, master = fictional_context()
        data = prepare_figure1(master.copy(), store, config)
        self.assertEqual(len(data.panel_a), 24)
        self.assertEqual(len(data.panel_b), 24)
        exact_key_equality(data.panel_a, data.panel_b, labels=("1A", "1B"))
        self.assertEqual(len(data.candidate_audit), 16)
        self.assertEqual(set(data.feature_summary["structural_class"]), {"analogue", "negative"})
        self.assertEqual(
            len(data.feature_summary),
            2 * data.wt_feature_count,
        )
        self.assertLessEqual(
            float(data.candidate_audit["absolute_reconstruction_error"].max()),
            1e-6,
        )

    def test_key_equality_fails_closed_when_one_panel_loses_a_row(self) -> None:
        config, _inputs, store, master = fictional_context()
        data = prepare_figure1(master.copy(), store, config)
        with self.assertRaisesRegex(ValueError, "key sets differ"):
            exact_key_equality(
                data.panel_a,
                data.panel_b.iloc[:-1],
                labels=("1A", "1B"),
            )

    def test_exports_every_panel_and_renders_offline(self) -> None:
        config, _inputs, store, master = fictional_context()
        data = prepare_figure1(master.copy(), store, config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_figure1(data, root, config)
            render_figure1(data, root)
            expected = [
                root / "figure1" / "data" / "figure1a_plot_data.csv.gz",
                root / "figure1" / "data" / "figure1b_plot_data.csv.gz",
                root / "figure1" / "data" / "figure1c_feature_summary.csv.gz",
                root / "figure1" / "figure1.png",
                root / "figure1" / "figure1.svg",
                root / "figure1" / "figure1.pdf",
            ]
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in expected))


if __name__ == "__main__":
    unittest.main()
