from __future__ import annotations

import unittest

import numpy as np

from protein_sae_pipeline.representations.pooling import (
    cosine_similarity,
    mean_pool_layer60,
    mean_pool_layer60_batch,
    residue_hidden_states,
    validate_bos_eos_layout,
)


class Layer60PoolingTests(unittest.TestCase):
    def test_removes_exactly_bos_and_eos_and_means_in_float32(self) -> None:
        states = np.asarray(
            [
                [1_000.0, 1_000.0],  # BOS: must not enter the mean
                [1.0, 2.0],
                [3.0, 8.0],
                [5.0, 2.0],
                [-1_000.0, -1_000.0],  # EOS: must not enter the mean
            ],
            dtype=np.float16,
        )
        residues = residue_hidden_states(states, sequence_length=3, expected_dimension=2)
        pooled = mean_pool_layer60(states, sequence_length=3, expected_dimension=2)
        np.testing.assert_array_equal(residues, states[1:4].astype(np.float32))
        np.testing.assert_array_equal(pooled, np.asarray([3.0, 4.0], dtype=np.float32))
        self.assertEqual(pooled.dtype, np.float32)

    def test_rejects_hidden_states_without_bos_and_eos(self) -> None:
        with self.assertRaisesRegex(ValueError, r"BOS \+ residues \+ EOS"):
            mean_pool_layer60(np.ones((3, 2), dtype=np.float32), sequence_length=3)

    def test_validates_right_padded_token_layout_and_batch_pooling(self) -> None:
        input_ids = np.asarray([[10, 1, 2, 11, 0], [10, 3, 4, 5, 11]])
        mask = np.asarray([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        validate_bos_eos_layout(
            input_ids,
            mask,
            [2, 3],
            bos_token_id=10,
            eos_token_id=11,
            pad_token_id=0,
        )
        states = np.zeros((2, 5, 1), dtype=np.float32)
        states[0, 1:3, 0] = [2.0, 4.0]
        states[1, 1:4, 0] = [3.0, 6.0, 9.0]
        pooled = mean_pool_layer60_batch(states, [2, 3], attention_mask=mask)
        np.testing.assert_array_equal(pooled[:, 0], np.asarray([3.0, 6.0], dtype=np.float32))

    def test_cosine_uses_float64_operands_and_defines_zero_norm_as_zero(self) -> None:
        left = np.asarray([1.0e10, 1.0, -1.0e10, 3.0], dtype=np.float32)
        right = np.asarray([1.0, 2.0, 1.0, 4.0], dtype=np.float32)
        expected = float(
            np.dot(left.astype(np.float64), right.astype(np.float64))
            / (
                np.linalg.norm(left.astype(np.float64))
                * np.linalg.norm(right.astype(np.float64))
            )
        )
        self.assertEqual(cosine_similarity(left, right), expected)
        self.assertEqual(cosine_similarity(np.zeros(2), np.ones(2)), 0.0)


if __name__ == "__main__":
    unittest.main()
