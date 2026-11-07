from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from protein_sae_pipeline.scoring.mmseqs import (
    EXPECTED_MMSEQS_COMMIT,
    MMseqsHit,
    build_mmseqs_easy_search_command,
    complete_mmseqs_scores,
    parse_raw_results,
    require_mmseqs_18_8cc5c,
    run_mmseqs_easy_search,
)


class MMseqsScoringTests(unittest.TestCase):
    def test_parse_results_selects_highest_bits_then_lowest_evalue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.tsv"
            raw.write_text(
                "candidate\ttarget__WT\t1e-20\t40\t25\t80\t0.8\t0.7\n"
                "candidate\ttarget__WT\t1e-10\t42\t30\t85\t0.85\t0.75\n"
                "candidate\ttarget__WT\t1e-30\t42\t35\t90\t0.9\t0.8\n",
                encoding="utf-8",
            )
            hits = parse_raw_results(raw, {"candidate"}, "target__WT")
        self.assertEqual(
            hits["candidate"],
            MMseqsHit(
                target_id="target__WT",
                evalue=1e-30,
                bit_score=42.0,
                percent_identity=35.0,
                alignment_length=90,
                query_coverage=0.9,
                target_coverage=0.8,
            ),
        )

    def test_missing_raw_output_and_missing_rows_are_explicit_no_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hits = parse_raw_results(
                Path(temporary) / "missing.tsv", {"a", "b"}, "target__WT"
            )
        scores = complete_mmseqs_scores(
            ["a", "b"], wt_sequence_id="target__WT", hits=hits
        )
        self.assertEqual([score.sequence_id for score in scores], ["a", "b"])
        self.assertTrue(all(not score.mmseqs2_hit for score in scores))
        self.assertTrue(all(score.mmseqs2_bit_score == 0.0 for score in scores))
        self.assertTrue(all(score.mmseqs2_evalue is None for score in scores))
        self.assertTrue(all(score.mmseqs2_alignment_length is None for score in scores))

    def test_parser_rejects_unknown_non_cognate_or_malformed_rows(self) -> None:
        lines = [
            "unexpected\ttarget__WT\t1e-3\t2\t20\t4\t1\t1\n",
            "candidate\tother__WT\t1e-3\t2\t20\t4\t1\t1\n",
            "candidate\ttarget__WT\t-1\t2\t20\t4\t1\t1\n",
            "candidate\ttarget__WT\t1e-3\t2\t101\t4\t1\t1\n",
            "candidate\ttarget__WT\t1e-3\t2\t20\t0\t1\t1\n",
            "candidate\ttarget__WT\t1e-3\t2\t20\t4\t1.1\t1\n",
            "candidate\ttarget__WT\t1e-3\tnan\t20\t4\t1\t1\n",
            "candidate\ttarget__WT\ttoo\tfew\tfields\n",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.tsv"
            for line in lines:
                with self.subTest(line=line):
                    raw.write_text(line, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        parse_raw_results(raw, {"candidate"}, "target__WT")

    def test_complete_scores_rejects_non_cognate_or_extra_hits(self) -> None:
        non_cognate = MMseqsHit("other", 1e-3, 10.0, 20.0, 4, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "not cognate WT"):
            complete_mmseqs_scores(
                ["candidate"],
                wt_sequence_id="target__WT",
                hits={"candidate": non_cognate},
            )
        with self.assertRaisesRegex(ValueError, "unexpected query IDs"):
            complete_mmseqs_scores(
                ["candidate"],
                wt_sequence_id="target__WT",
                hits={"extra": non_cognate},
            )

    def test_mmseqs_command_matches_configured_protocol_and_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = build_mmseqs_easy_search_command(
                executable="/tools/mmseqs",
                query_fasta_path=root / "candidates.fasta",
                target_wt_fasta_path=root / "cognate_wt.fasta",
                raw_output_path=root / "raw.tsv",
                temporary_directory=root / "scratch",
                threads=6,
            )
            self.assertEqual(
                command,
                (
                    "/tools/mmseqs",
                    "easy-search",
                    str(root / "candidates.fasta"),
                    str(root / "cognate_wt.fasta"),
                    str(root / "raw.tsv"),
                    str(root / "scratch"),
                    "-e",
                    "1.0",
                    "-s",
                    "7.5",
                    "--max-seqs",
                    "2000",
                    "--threads",
                    "6",
                    "--format-output",
                    "query,target,evalue,bits,pident,alnlen,qcov,tcov",
                ),
            )

    def test_mmseqs_wrapper_parses_hit_and_completes_no_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.tsv"

            def fake_run(
                command: tuple[str, ...], **_: object
            ) -> subprocess.CompletedProcess[str]:
                raw.write_text(
                    "a\ttarget__WT\t1e-9\t12.5\t40\t10\t0.8\t0.7\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    command, 0, stdout="progress", stderr=""
                )

            with patch(
                "protein_sae_pipeline.scoring.mmseqs.subprocess.run",
                side_effect=fake_run,
            ):
                result = run_mmseqs_easy_search(
                    query_fasta_path=root / "candidates.fasta",
                    target_wt_fasta_path=root / "wt.fasta",
                    raw_output_path=raw,
                    temporary_directory=root / "scratch",
                    expected_query_ids=["a", "b"],
                    target_id="target__WT",
                    executable="mmseqs",
                    threads=3,
                    check_version=False,
                )
            self.assertEqual(
                result.command[1:4],
                (
                    "easy-search",
                    str(root / "candidates.fasta"),
                    str(root / "wt.fasta"),
                ),
            )
        self.assertEqual(result.stdout, "progress")
        self.assertTrue(result.scores[0].mmseqs2_hit)
        self.assertEqual(result.scores[0].mmseqs2_bit_score, 12.5)
        self.assertFalse(result.scores[1].mmseqs2_hit)
        self.assertEqual(result.scores[1].mmseqs2_bit_score, 0.0)

    def test_mmseqs_version_accepts_only_the_pinned_release(self) -> None:
        banners = ["18-8cc5c", "18.8cc5c", EXPECTED_MMSEQS_COMMIT]
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "mmseqs"
            executable.write_bytes(b"fake mmseqs")
            for banner in banners:
                with self.subTest(banner=banner):
                    completed = subprocess.CompletedProcess(
                        [str(executable), "version"],
                        0,
                        stdout=banner + "\n",
                        stderr="",
                    )
                    with patch(
                        "protein_sae_pipeline.scoring.mmseqs.shutil.which",
                        return_value=str(executable),
                    ), patch(
                        "protein_sae_pipeline.scoring.mmseqs.subprocess.run",
                        return_value=completed,
                    ):
                        observed = require_mmseqs_18_8cc5c(str(executable))
                    self.assertEqual(observed.normalized_release, "18-8cc5c")
                    self.assertEqual(observed.executable, str(executable.resolve()))

    def test_mmseqs_version_rejects_another_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "mmseqs"
            executable.write_bytes(b"fake mmseqs")
            completed = subprocess.CompletedProcess(
                [str(executable), "version"],
                0,
                stdout="17-b804f\n",
                stderr="",
            )
            with patch(
                "protein_sae_pipeline.scoring.mmseqs.shutil.which",
                return_value=str(executable),
            ), patch(
                "protein_sae_pipeline.scoring.mmseqs.subprocess.run",
                return_value=completed,
            ), self.assertRaisesRegex(
                RuntimeError, "Expected MMseqs2 release 18-8cc5c"
            ):
                require_mmseqs_18_8cc5c(str(executable))


if __name__ == "__main__":
    unittest.main()
