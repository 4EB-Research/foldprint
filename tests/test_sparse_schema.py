from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.io import sha256_file
from protein_sae_pipeline.representations.sparse import (
    SparseResidueMatrix,
    build_representation_chunk,
    load_representation_chunk,
    save_representation_chunk,
)
from protein_sae_pipeline.representations.store import load_indexed_representations
from protein_sae_pipeline.validation import load_and_validate_inputs


ROOT = Path(__file__).resolve().parents[1]


class SparseChunkSchemaTests(unittest.TestCase):
    @staticmethod
    def _chunk():
        first = SparseResidueMatrix(
            np.asarray([[1, 4], [1, 3]], dtype=np.uint16),
            np.asarray([[0.5, 0.2], [0.7, 0.9]], dtype=np.float32),
            size=8,
        )
        second = SparseResidueMatrix(
            np.asarray([[0, 5]], dtype=np.uint16),
            np.asarray([[0.4, 0.6]], dtype=np.float32),
            size=8,
        )
        return build_representation_chunk(
            manifest_rows=[4, 9],
            sequence_hashes=["a" * 64, "b" * 64],
            raw_mean=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            residue_matrices=[first, second],
        )

    def test_round_trips_exact_sparse_schema_without_pickle(self) -> None:
        chunk = self._chunk()
        chunk.validate(expected_embedding_dimension=3, expected_topk=2, codebook_size=8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunk.npz"
            save_representation_chunk(
                path,
                chunk,
                expected_embedding_dimension=3,
                expected_topk=2,
                codebook_size=8,
            )
            observed = load_representation_chunk(
                path,
                expected_embedding_dimension=3,
                expected_topk=2,
                codebook_size=8,
            )
        self.assertEqual(observed.hashes(), ("a" * 64, "b" * 64))
        np.testing.assert_array_equal(observed.profile(0, codebook_size=8).indices, [1, 3, 4])
        np.testing.assert_array_equal(
            observed.profile(0, codebook_size=8).values,
            np.asarray([0.7, 0.9, 0.2], dtype=np.float32),
        )
        self.assertEqual(
            observed.profile(0, codebook_size=8).norm,
            float(observed.sae_profile_norms[0]),
        )

    def test_rejects_a_profile_that_is_not_the_residue_max_pool(self) -> None:
        chunk = self._chunk()
        bad_values = chunk.sae_profile_values.copy()
        bad_values[0] = np.float32(0.6)
        bad_norms = chunk.sae_profile_norms.copy()
        first_stop = int(chunk.sae_profile_offsets[1])
        bad_norms[0] = np.float32(
            np.linalg.norm(bad_values[:first_stop].astype(np.float64))
        )
        corrupted = replace(chunk, sae_profile_values=bad_values, sae_profile_norms=bad_norms)
        with self.assertRaisesRegex(ValueError, "not the residue max pool"):
            corrupted.validate(codebook_size=8)

    def test_allows_one_sequence_hash_against_different_references(self) -> None:
        chunk = self._chunk()
        hashes = chunk.sequence_sha256.copy()
        hashes[1] = hashes[0]
        repeated = replace(chunk, sequence_sha256=hashes)
        repeated.validate(expected_embedding_dimension=3, expected_topk=2, codebook_size=8)

    def test_index_rejects_an_unknown_role(self) -> None:
        config = load_config(ROOT / "configs/example.yaml")
        inputs = load_and_validate_inputs(config, ROOT / "example")
        chunk_path = ROOT / "example/representations/landscape_representations.npz"
        index = pd.read_csv(ROOT / "example/representations/representation_index.csv")
        index.loc[0, "role"] = "unknown"
        with tempfile.TemporaryDirectory() as temporary:
            chunk_copy = Path(temporary) / "landscape_representations.npz"
            index_path = Path(temporary) / "representation_index.csv"
            metadata_path = Path(temporary) / "landscape_representations.json"
            shutil.copyfile(chunk_path, chunk_copy)
            index.to_csv(index_path, index=False)
            metadata = json.loads(
                (chunk_path.with_suffix(".json")).read_text(encoding="utf-8")
            )
            metadata["index_sha256"] = sha256_file(index_path)
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown role"):
                load_indexed_representations(
                    chunk_copy,
                    index_path,
                    inputs,
                    config=config,
                )

    def test_indexed_store_rejects_stale_or_self_declared_provenance(self) -> None:
        config = load_config(ROOT / "configs/example.yaml")
        inputs = load_and_validate_inputs(config, ROOT / "example")
        source_root = ROOT / "example/representations"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "landscape_representations.npz",
                "landscape_representations.json",
                "representation_index.csv",
            ):
                shutil.copyfile(source_root / name, root / name)
            metadata_path = root / "landscape_representations.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["dataset_id"] = "stale_private_dataset"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                load_indexed_representations(
                    root / "landscape_representations.npz",
                    root / "representation_index.csv",
                    inputs,
                    config=config,
                )


if __name__ == "__main__":
    unittest.main()
