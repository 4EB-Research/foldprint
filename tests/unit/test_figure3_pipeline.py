from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from figure_fixtures import fictional_context
from protein_sae_pipeline.figures.figure3 import (
    SCORE_COLUMNS,
    export_figure3,
    prepare_figure3,
    render_figure3,
)


class Figure3PipelineTests(unittest.TestCase):
    def test_uses_complete_remote_cohort_and_one_shared_bootstrap_plan(self) -> None:
        config, _inputs, _store, master = fictional_context()
        data = prepare_figure3(master.copy(), config)
        self.assertEqual(data.source_rows, 32)
        self.assertEqual(data.source_cases, 16)
        self.assertEqual(data.source_controls, 16)
        self.assertEqual(data.endpoint_replicates, 100)
        self.assertEqual(data.roc_replicates, 50)
        expected_endpoint_rows = 100 * 2 * len(SCORE_COLUMNS) * 2
        self.assertEqual(len(data.endpoint_bootstrap), expected_endpoint_rows)
        counts = data.endpoint_bootstrap.groupby(
            ["replicate", "reference_id", "method"], observed=True
        )["target_fpr"].nunique()
        self.assertTrue(counts.eq(2).all())
        self.assertEqual(len(data.bootstrap_plan_sha256), 64)

    def test_exports_empirical_and_bootstrap_plot_data_and_renders(self) -> None:
        config, _inputs, _store, master = fictional_context()
        data = prepare_figure3(master.copy(), config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_figure3(data, root, config)
            render_figure3(data, root, config)
            expected = [
                root / "figure3" / "data" / "figure3_roc_plot_data.csv.gz",
                root / "figure3" / "data" / "figure3_operating_points.csv",
                root / "figure3" / "data" / "figure3_endpoint_bootstrap_replicates.csv.gz",
                root / "figure3" / "data" / "figure3_roc_bands.csv.gz",
                root / "figure3" / "figure3.png",
                root / "figure3" / "figure3.svg",
                root / "figure3" / "figure3.pdf",
                root / "figure3" / "figure3a_roc.png",
                root / "figure3" / "figure3b_endpoints.png",
            ]
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in expected))


if __name__ == "__main__":
    unittest.main()
