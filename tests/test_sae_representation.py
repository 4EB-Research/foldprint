from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from protein_sae_pipeline.representations.sae import (
    SAENormalization,
    extract_topk_residue_activations,
    load_sae_normalization,
    normalize_sae_activations,
)


class SAERepresentationTests(unittest.TestCase):
    def test_applies_float32_divide_then_idf_in_that_order(self) -> None:
        raw = np.asarray([[7.1, 3.3, 1.9, 0.5]], dtype=np.float32)
        maximum = np.asarray([3.0, 7.0, 0.7, 0.11], dtype=np.float32)
        idf = np.asarray([0.9, 1.7, 2.3, 0.4], dtype=np.float32)
        assets = SAENormalization(maximum, idf)
        expected = np.empty_like(raw)
        np.divide(raw, maximum, out=expected)
        np.multiply(expected, idf, out=expected)
        observed = normalize_sae_activations(raw, assets)
        np.testing.assert_array_equal(observed, expected)
        self.assertEqual(observed.dtype, np.float32)

    def test_topk_is_deterministic_then_max_pools_over_residues(self) -> None:
        normalized = np.asarray(
            [
                [0.0, 0.7, 0.0, 0.4, 0.2],
                [0.0, 0.3, 0.0, 0.9, 0.1],
            ],
            dtype=np.float32,
        )
        residues = extract_topk_residue_activations(normalized, topk=2)
        np.testing.assert_array_equal(
            residues.indices,
            np.asarray([[1, 3], [1, 3]], dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            residues.values,
            np.asarray([[0.7, 0.4], [0.3, 0.9]], dtype=np.float32),
        )
        profile = residues.max_pool()
        np.testing.assert_array_equal(profile.indices, np.asarray([1, 3], dtype=np.uint16))
        np.testing.assert_array_equal(profile.values, np.asarray([0.7, 0.9], dtype=np.float32))

    def test_loads_offline_npz_asset_with_exact_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "normalization.npz"
            np.savez(
                path,
                max_per_feature=np.asarray([1.0, 2.0], dtype=np.float32),
                idf_per_feature=np.asarray([0.5, 1.5], dtype=np.float32),
            )
            asset = load_sae_normalization(path, expected_codebook_size=2)
        self.assertEqual(asset.codebook_size, 2)
        self.assertEqual(len(asset.source_sha256 or ""), 64)

    def test_rejects_a_self_declared_asset_with_wrong_array_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "normalization.npz"
            np.savez(
                path,
                maximum=np.asarray([1.0, 2.0], dtype=np.float32),
                idf=np.asarray([0.5, 1.5], dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "maximum array checksum"):
                load_sae_normalization(
                    path,
                    expected_codebook_size=2,
                    expected_maximum_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
