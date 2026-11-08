from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import unittest

import yaml

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.keys import CandidateKey, normalize_sequence, sequence_sha256
from protein_sae_pipeline.validation import load_and_validate_inputs


ROOT = Path(__file__).resolve().parents[2]


class KeyTests(unittest.TestCase):
    def test_normalization_and_hash_are_stable(self) -> None:
        self.assertEqual(normalize_sequence(" acd\nEF "), "ACDEF")
        self.assertEqual(sequence_sha256(" acdef "), sequence_sha256("ACDEF"))

    def test_noncanonical_sequence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_sequence("ACDX")

    def test_composite_key_requires_lower_hex(self) -> None:
        CandidateKey("ref", "a" * 64)
        with self.assertRaises(ValueError):
            CandidateKey("ref", "A" * 64)


class ConfigurationTests(unittest.TestCase):
    @staticmethod
    def _copy_reference_validation_inputs(destination: Path) -> None:
        (destination / "inputs").mkdir(parents=True)
        (destination / "profiles").mkdir()
        (destination / "structures").mkdir()
        shutil.copyfile(
            ROOT / "example/inputs/references.csv",
            destination / "inputs/references.csv",
        )
        shutil.copyfile(
            ROOT / "example/profiles/fictional_alpha.hmm",
            destination / "profiles/fictional_alpha.hmm",
        )
        shutil.copyfile(
            ROOT / "example/structures/fictional_alpha_wt.pdb",
            destination / "structures/fictional_alpha_wt.pdb",
        )

    def test_example_config_and_inputs_validate(self) -> None:
        config = load_config(ROOT / "configs/example.yaml")
        inputs = load_and_validate_inputs(config, ROOT / "example")
        self.assertEqual(len(inputs.candidates), 48)
        self.assertEqual(set(inputs.references), {"fictional_alpha", "fictional_beta"})
        self.assertEqual(len(inputs.structures), 48)
        for reference_id, reference in inputs.references.items():
            configured = config.raw["references"][reference_id]
            self.assertEqual(reference.pfam_accession, configured["pfam_accession"])

    def test_invalid_pfam_accession_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = yaml.safe_load((ROOT / "configs/example.yaml").read_text())
            source["references"]["fictional_alpha"]["pfam_accession"] = "PF00001"
            path = Path(temporary) / "bad.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pfam_accession"):
                load_config(path)

    def test_reference_manifest_pfam_accession_must_match_config(self) -> None:
        config = load_config(ROOT / "configs/example.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            self._copy_reference_validation_inputs(data_root)
            manifest = data_root / "inputs/references.csv"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    ",FICTIONAL.1,", ",FICTIONAL.2,", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Pfam accession differs from config"):
                load_and_validate_inputs(config, data_root)

    def test_pfam_profile_acc_must_match_config_accession(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            self._copy_reference_validation_inputs(data_root)
            profile = data_root / "profiles/fictional_alpha.hmm"
            payload = profile.read_bytes().replace(
                b"ACC   FICTIONAL.1", b"ACC   DIFFERENT.1", 1
            )
            profile.write_bytes(payload)

            config = load_config(ROOT / "configs/example.yaml")

            with self.assertRaisesRegex(ValueError, "HMM ACC differs from config"):
                load_and_validate_inputs(config, data_root)

    def test_unknown_config_field_is_rejected(self) -> None:
        source = yaml.safe_load((ROOT / "configs/example.yaml").read_text())
        source["surprise"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_path_escape_is_rejected(self) -> None:
        source = yaml.safe_load((ROOT / "configs/example.yaml").read_text())
        source["paths"]["candidates"] = "../outside.csv"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            config = load_config(path)
            with self.assertRaises(ValueError):
                load_and_validate_inputs(config, ROOT / "example")


if __name__ == "__main__":
    unittest.main()
