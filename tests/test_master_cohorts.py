from __future__ import annotations

import unittest

import pandas as pd

from protein_sae_pipeline.cohorts import construct_cohorts
from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.master import assemble_master, validate_master_against_inputs
from protein_sae_pipeline.validation import load_and_validate_inputs


class MasterCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_config(cls.root / "configs/example.yaml")
        cls.inputs = load_and_validate_inputs(cls.config, cls.root / "example")

    def table(self, columns: dict[str, object]) -> pd.DataFrame:
        rows = []
        for candidate in self.inputs.candidates:
            rows.append(
                {
                    "reference_id": candidate.reference_id,
                    "sequence_sha256": candidate.sequence_sha256,
                    **columns,
                }
            )
        return pd.DataFrame(rows)

    def score_tables(self):  # type: ignore[no-untyped-def]
        identity_rows = []
        for candidate in self.inputs.candidates:
            remote = candidate.manifest_row % 3 != 0
            matches = 8 if remote else 30
            identity_rows.append(
                {
                    "reference_id": candidate.reference_id,
                    "sequence_sha256": candidate.sequence_sha256,
                    "positional_matches": matches,
                    "positional_mismatches": 40 - matches,
                    "positional_identity_fraction": matches / 40,
                    "positional_identity_percent": matches * 2.5,
                }
            )
        identity = pd.DataFrame(identity_rows)
        structure = self.table(
            {
                "candidate_coordinate_length": 40,
                "wt_coordinate_length": 40,
                "usalign_tm_score_candidate_norm": 0.7,
                "usalign_tm_score_wt_norm": 0.7,
                "usalign_aligned_length": 40,
                "usalign_rmsd_angstrom": 1.0,
                "usalign_aligned_sequence_identity": 0.2,
            }
        )
        pfam = self.table(
            {
                "pfam_accession": "FICTIONAL.1",
                "pfam_full_sequence_evalue": 1.0,
                "pfam_full_sequence_bit_score": -2.0,
                "pfam_best_domain_evalue": 1.0,
                "pfam_best_domain_bit_score": -3.0,
                "pfam_score_left_censored": False,
                "pfam_censor_floor": -4.0,
            }
        )
        mmseqs = self.table(
            {
                "mmseqs2_bit_score": 0.0,
                "mmseqs2_evalue": None,
                "mmseqs2_percent_identity": None,
                "mmseqs2_alignment_length": None,
                "mmseqs2_query_coverage": None,
                "mmseqs2_target_coverage": None,
                "mmseqs2_hit": False,
            }
        )
        representations = self.table(
            {
                "esmc_cosine_to_wt": 0.5,
                "esmc_zero_norm": False,
                "sae_cosine_to_wt": 0.5,
                "sae_profile_norm": 1.0,
                "sae_profile_nnz": 4,
                "sae_zero_norm": False,
            }
        )
        return identity, structure, pfam, mmseqs, representations

    def test_missing_score_key_is_rejected(self) -> None:
        identity, structure, pfam, mmseqs, representations = self.score_tables()
        with self.assertRaises(ValueError):
            assemble_master(
                self.inputs,
                self.config,
                identity=identity.iloc[:-1],
                structure=structure,
                pfam=pfam,
                mmseqs=mmseqs,
                representations=representations,
            )

    def test_tm_boundary_is_inclusive_and_identity_boundary_strict(self) -> None:
        identity, structure, pfam, mmseqs, representations = self.score_tables()
        identity["positional_matches"] = 12
        identity["positional_mismatches"] = 28
        identity["positional_identity_fraction"] = 0.30
        identity["positional_identity_percent"] = 30.0
        structure["usalign_tm_score_wt_norm"] = 0.50
        master = assemble_master(
            self.inputs,
            self.config,
            identity=identity,
            structure=structure,
            pfam=pfam,
            mmseqs=mmseqs,
            representations=representations,
        )
        self.assertFalse(master["is_remote"].any())
        self.assertTrue(master["is_structural_analogue"].all())

    def test_stale_same_sized_master_is_rejected_before_figures(self) -> None:
        tables = self.score_tables()
        master = assemble_master(
            self.inputs,
            self.config,
            identity=tables[0],
            structure=tables[1],
            pfam=tables[2],
            mmseqs=tables[3],
            representations=tables[4],
        )
        validate_master_against_inputs(master, self.inputs, self.config)

        stale_key = master.copy()
        stale_key.loc[0, "sequence_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "candidate keys differ"):
            validate_master_against_inputs(stale_key, self.inputs, self.config)

        stale_method = master.copy()
        stale_method.loc[0, "generation_method"] = self.config.generation_methods[1]
        with self.assertRaisesRegex(ValueError, "generation_method"):
            validate_master_against_inputs(stale_method, self.inputs, self.config)


if __name__ == "__main__":
    unittest.main()
