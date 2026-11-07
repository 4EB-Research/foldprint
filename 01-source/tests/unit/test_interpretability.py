from __future__ import annotations

import unittest

import numpy as np

from protein_sae_pipeline.interpretability import (
    feature_contribution_spectrum,
    feature_recurrence,
    position_activation_counts,
    position_feature_audit,
    position_recurrence,
    position_unique_features,
    rank_wt_features,
)
from protein_sae_pipeline.representations.sparse import SparseResidueMatrix, SparseVector


class InterpretabilityTests(unittest.TestCase):
    def test_wt_ranking_recurrence_and_cosine_contributions(self) -> None:
        wt = SparseVector(
            np.asarray([1, 2, 3], dtype=np.uint16),
            np.asarray([0.8, 0.5, 0.5], dtype=np.float32),
            size=8,
        )
        candidates = [
            SparseVector(
                np.asarray([1, 3], dtype=np.uint16),
                np.asarray([0.4, 0.2], dtype=np.float32),
                8,
            ),
            SparseVector(
                np.asarray([2, 4], dtype=np.uint16),
                np.asarray([0.7, 0.9], dtype=np.float32),
                8,
            ),
            SparseVector(
                np.asarray([1, 2], dtype=np.uint16),
                np.asarray([0.3, 0.6], dtype=np.float32),
                8,
            ),
        ]
        ranking = rank_wt_features(wt)
        np.testing.assert_array_equal(ranking.feature_ids, [1, 2, 3])

        recurrence = feature_recurrence(candidates, ranking.feature_ids)
        np.testing.assert_array_equal(recurrence.active_counts, [2, 2, 1])
        np.testing.assert_allclose(recurrence.recurrence, [2 / 3, 2 / 3, 1 / 3])
        self.assertTrue(np.all(recurrence.wilson_low <= recurrence.recurrence))
        self.assertTrue(np.all(recurrence.recurrence <= recurrence.wilson_high))

        spectrum = feature_contribution_spectrum(candidates[0], wt)
        self.assertAlmostEqual(
            float(spectrum.cosine_contributions.sum()),
            spectrum.cosine_similarity,
            places=14,
        )
        self.assertAlmostEqual(spectrum.cumulative_numerator_shares[-1], 1.0, places=14)

    @staticmethod
    def _residues() -> SparseResidueMatrix:
        # Feature 1 is unique to position 1; 3 to position 2; 4 to position 3.
        return SparseResidueMatrix(
            np.asarray([[1, 2], [2, 3], [2, 4]], dtype=np.uint16),
            np.asarray([[0.8, 0.2], [0.3, 0.9], [0.4, 0.7]], dtype=np.float32),
            size=8,
        )

    def test_position_unique_uses_active_here_and_nowhere_else(self) -> None:
        unique = position_unique_features(self._residues())
        np.testing.assert_array_equal(unique.counts, [1, 1, 1])
        np.testing.assert_array_equal(unique.feature_ids_by_position[0], [1])
        np.testing.assert_array_equal(unique.feature_ids_by_position[1], [3])
        np.testing.assert_array_equal(unique.feature_ids_by_position[2], [4])
        self.assertEqual(unique.total_unique_features, 3)

    def test_position_metrics_use_one_based_direct_mapping_and_strict_activity(self) -> None:
        residues = self._residues()
        feature_ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
        counts = position_activation_counts(residues, feature_ids, position_1based=2)
        self.assertEqual(counts.active_at_position, 2)
        self.assertEqual(counts.increasing_from_previous, 2)  # feature 2 rises; feature 3 appears
        self.assertEqual(counts.new_at_position, 1)
        self.assertEqual(counts.local_to_position, 1)
        self.assertEqual(counts.active_nowhere_else, 1)

        audit = position_feature_audit(residues, feature_ids, position_1based=2)
        self.assertEqual(int(feature_ids[np.argmin(audit.activation_rank)]), 3)
        self.assertEqual(int(feature_ids[np.argmin(audit.gain_rank)]), 3)

        recurrence = position_recurrence(
            [residues, residues],
            [1, 3],
            position_1based=2,
        )
        np.testing.assert_array_equal(recurrence.active_counts, [0, 2])


if __name__ == "__main__":
    unittest.main()
