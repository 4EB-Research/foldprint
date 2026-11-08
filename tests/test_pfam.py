from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from protein_sae_pipeline.scoring.pfam import (
    PfamHit,
    build_hmmsearch_command,
    complete_pfam_scores,
    parse_tblout,
    require_hmmer_3_4,
    run_hmmsearch,
)
from protein_sae_pipeline.workflows import load_pfam_censor_floors


def _tblout_row(
    sequence_id: str,
    *,
    full_evalue: str = "1e-8",
    full_score: str = "31.5",
    domain_evalue: str = "2e-8",
    domain_score: str = "30.1",
) -> str:
    return (
        f"seq_{sequence_id} - HDPD PF02924.20 {full_evalue} {full_score} 0.0 "
        f"{domain_evalue} {domain_score} 0.0 1.0 1 0 0 1 1 1 1 description\n"
    )


class PfamScoringTests(unittest.TestCase):
    def test_parse_tblout_reads_exact_full_sequence_and_domain_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.tblout"
            path.write_text(
                "# hmmsearch output\n"
                + _tblout_row("candidate_a", full_score="4", domain_score="2")
                + _tblout_row("candidate_a", full_score="7", domain_score="3"),
                encoding="utf-8",
            )
            hits = parse_tblout(path, {"candidate_a"})
        self.assertEqual(
            hits["candidate_a"],
            PfamHit(
                full_sequence_evalue=1e-8,
                full_sequence_bit_score=7.0,
                best_domain_evalue=2e-8,
                best_domain_bit_score=3.0,
            ),
        )

    def test_complete_scores_left_censors_every_missing_key_once_per_target(
        self,
    ) -> None:
        hits = {
            "a": PfamHit(1e-4, -2.5, 2e-4, -3.0),
            "c": PfamHit(1e-9, 8.0, 2e-9, 7.0),
        }
        scores = complete_pfam_scores(["a", "b", "c"], hits)
        self.assertEqual([score.sequence_id for score in scores], ["a", "b", "c"])
        self.assertTrue(scores[1].pfam_score_left_censored)
        self.assertEqual(scores[1].pfam_full_sequence_bit_score, -3.5)
        self.assertEqual(scores[1].pfam_best_domain_bit_score, -4.0)
        self.assertEqual(scores[1].pfam_full_sequence_evalue, float("inf"))
        self.assertFalse(scores[0].pfam_score_left_censored)

    def test_complete_scores_requires_a_finite_hit_and_exact_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "without a finite hit"):
            complete_pfam_scores(["a"], {})
        with self.assertRaisesRegex(ValueError, "unexpected sequence IDs"):
            complete_pfam_scores(
                ["a"], {"other": PfamHit(1.0, 1.0, 1.0, 1.0)}
            )

    def test_complete_scores_uses_frozen_full_landscape_floors_for_a_sample(self) -> None:
        scores = complete_pfam_scores(
            ["sample_no_hit"],
            {},
            censor_floor=(-12.5, -14.0),
        )
        self.assertEqual(scores[0].pfam_full_sequence_bit_score, -12.5)
        self.assertEqual(scores[0].pfam_best_domain_bit_score, -14.0)
        self.assertTrue(scores[0].pfam_score_left_censored)

        with self.assertRaisesRegex(ValueError, "not above the frozen floor"):
            complete_pfam_scores(
                ["finite"],
                {"finite": PfamHit(1.0, -13.0, 1.0, -10.0)},
                censor_floor=(-12.5, -14.0),
            )

    def test_parse_tblout_fails_closed_on_bad_rows(self) -> None:
        rows = [
            _tblout_row("unexpected"),
            _tblout_row("candidate", full_score="nan"),
            "seq_candidate too few fields\n",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.tblout"
            for row in rows:
                with self.subTest(row=row):
                    path.write_text(row, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        parse_tblout(path, {"candidate"})

    def test_hmmsearch_command_matches_configured_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = build_hmmsearch_command(
                executable="/tools/hmmsearch",
                profile_path=root / "PF02924.hmm",
                candidate_fasta_path=root / "candidates.fasta",
                tblout_path=root / "scores.tblout",
                cpus=8,
            )
            self.assertEqual(
                command,
                (
                    "/tools/hmmsearch",
                    "--cpu",
                    "8",
                    "--max",
                    "--noali",
                    "-T",
                    "-1000",
                    "--domT",
                    "-1000",
                    "--incT",
                    "-1000",
                    "--incdomT",
                    "-1000",
                    "--tblout",
                    str(root / "scores.tblout"),
                    str(root / "PF02924.hmm"),
                    str(root / "candidates.fasta"),
                ),
            )

    def test_hmmsearch_wrapper_parses_and_completes_all_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tblout = root / "scores.tblout"

            def fake_run(
                command: tuple[str, ...], **_: object
            ) -> subprocess.CompletedProcess[str]:
                tblout.write_text(_tblout_row("a"), encoding="utf-8")
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="diagnostic"
                )

            with patch(
                "protein_sae_pipeline.scoring.pfam.subprocess.run",
                side_effect=fake_run,
            ):
                result = run_hmmsearch(
                    profile_path=root / "profile.hmm",
                    candidate_fasta_path=root / "candidates.fasta",
                    tblout_path=tblout,
                    expected_sequence_ids=["a", "b"],
                    executable="hmmsearch",
                    cpus=2,
                    check_version=False,
                )
        self.assertEqual(result.command[:3], ("hmmsearch", "--cpu", "2"))
        self.assertEqual(result.stderr, "diagnostic")
        self.assertEqual([score.sequence_id for score in result.scores], ["a", "b"])
        self.assertFalse(result.scores[0].pfam_score_left_censored)
        self.assertTrue(result.scores[1].pfam_score_left_censored)

    def test_hmmer_version_is_pinned_to_3_4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "hmmsearch"
            executable.write_bytes(b"fake")
            completed = subprocess.CompletedProcess(
                [str(executable), "-h"],
                0,
                stdout="# HMMER 3.4 (Nov 2023)\nhelp\n",
                stderr="",
            )
            with patch(
                "protein_sae_pipeline.scoring.pfam.shutil.which",
                return_value=str(executable),
            ), patch(
                "protein_sae_pipeline.scoring.pfam.subprocess.run",
                return_value=completed,
            ):
                observed = require_hmmer_3_4(str(executable))
            self.assertEqual(observed.executable, str(executable.resolve()))
            self.assertIn("HMMER 3.4", observed.banner)

            wrong = subprocess.CompletedProcess(
                [str(executable), "-h"],
                0,
                stdout="# HMMER 3.3.2\n",
                stderr="",
            )
            with patch(
                "protein_sae_pipeline.scoring.pfam.shutil.which",
                return_value=str(executable),
            ), patch(
                "protein_sae_pipeline.scoring.pfam.subprocess.run",
                return_value=wrong,
            ), self.assertRaisesRegex(RuntimeError, "Expected HMMER 3.4"):
                require_hmmer_3_4(str(executable))

    def test_frozen_floor_table_is_strict_and_byte_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "floors.csv"
            path.write_text(
                "reference_id,pfam_full_sequence_censor_floor,pfam_best_domain_censor_floor\n"
                "target,-12.5,-14.0\n",
                encoding="utf-8",
            )
            floors, digest = load_pfam_censor_floors(path, {"target"})
            self.assertEqual(floors, {"target": (-12.5, -14.0)})
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "references differ"):
                load_pfam_censor_floors(path, {"different"})


if __name__ == "__main__":
    unittest.main()
