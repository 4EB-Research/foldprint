from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from protein_sae_pipeline.bootstrap import (
    bootstrap_operating_points,
    make_bootstrap_plan,
)
from protein_sae_pipeline.statistics import empirical_operating_point, empirical_roc_grid
from protein_sae_pipeline.statistics import empirical_operating_points


class OperatingPointTests(unittest.TestCase):
    def test_ties_are_conservative(self) -> None:
        controls = [10, 9, 9, 8, 7, 6, 5, 4, 3, 2]
        cases = [11, 10, 9, 1]
        point = empirical_operating_point(cases, controls, 0.20)
        self.assertEqual(point.allowed_false_positives, 2)
        self.assertEqual(point.threshold, 9)
        self.assertEqual(point.observed_false_positives, 1)
        self.assertEqual(point.true_positives, 2)

    def test_one_percent_with_small_control_set_allows_zero(self) -> None:
        point = empirical_operating_point([3, 2], [2, 1, 0], 0.01)
        self.assertEqual(point.allowed_false_positives, 0)
        self.assertEqual(point.observed_false_positives, 0)
        self.assertEqual(point.threshold, 2)

    def test_roc_uses_fixed_grid(self) -> None:
        curve = empirical_roc_grid([4, 3], list(range(100)), max_fpr=0.05, grid_points=6)
        self.assertEqual(len(curve), 6)
        np.testing.assert_allclose(curve.target_fpr, np.linspace(0, 0.05, 6))

    def test_batched_endpoints_are_exactly_equivalent_to_individual_calls(self) -> None:
        cases = [11, 10, 9, 1, 1]
        controls = [10, 9, 9, 8, 7, 6, 5, 4, 3, 2]
        endpoints = (0.0, 0.01, 0.20, 0.50)
        batched = empirical_operating_points(cases, controls, endpoints)
        individual = tuple(
            empirical_operating_point(cases, controls, endpoint) for endpoint in endpoints
        )
        self.assertEqual(batched, individual)


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        rows = []
        for reference in ("a", "b"):
            for method in ("m1", "m2"):
                for structural_class in ("analogue", "negative"):
                    for index in range(8):
                        rows.append(
                            {
                                "reference_id": reference,
                                "sequence_sha256": f"{len(rows):064x}",
                                "generation_method": method,
                                "structural_class": structural_class,
                                "score_a": float(index + (structural_class == "analogue")),
                                "score_b": float(index * 2 + (structural_class == "analogue")),
                            }
                        )
        return pd.DataFrame(rows)

    def test_plan_is_score_independent_and_stratum_preserving(self) -> None:
        frame = self.frame()
        first = make_bootstrap_plan(frame, replicates=16, seed=1729)
        changed = frame.copy()
        changed["score_a"] = changed["score_a"] * -100
        second = make_bootstrap_plan(changed, replicates=16, seed=1729)
        np.testing.assert_array_equal(first.sampled_ilocs, second.sampled_ilocs)
        for replicate in range(first.replicates):
            observed = first.sample(frame, replicate).groupby(
                ["reference_id", "structural_class", "generation_method"]
            ).size()
            expected = frame.groupby(
                ["reference_id", "structural_class", "generation_method"]
            ).size()
            pd.testing.assert_series_equal(observed, expected)

    def test_methods_and_endpoints_share_replicate_rows(self) -> None:
        frame = self.frame()
        plan = make_bootstrap_plan(frame, replicates=3, seed=11)
        results = bootstrap_operating_points(
            frame, plan, {"A": "score_a", "B": "score_b"}, endpoints=(0.01, 0.05)
        )
        counts = results.groupby(["replicate", "reference_id"]).size()
        self.assertTrue(counts.eq(4).all())
        # The shared plan exposes one common sampled-row matrix for every method.
        self.assertEqual(plan.sampled_ilocs.shape, (3, len(frame)))

    def test_lazy_plan_matches_one_materialized_numpy_rng_stream(self) -> None:
        frame = self.frame()
        plan = make_bootstrap_plan(frame, replicates=5, seed=47)
        rng = np.random.default_rng(47)
        expected = []
        for _replicate in range(5):
            expected.append(
                np.concatenate(
                    [rng.choice(group, size=len(group), replace=True) for group in plan.groups]
                )
            )
        np.testing.assert_array_equal(plan.sampled_ilocs, np.vstack(expected))
        np.testing.assert_array_equal(plan.draw(2), expected[2])


if __name__ == "__main__":
    unittest.main()
