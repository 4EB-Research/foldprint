from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from protein_sae_pipeline.confidence import match_all_negatives, match_ptm
from protein_sae_pipeline.reviewer_analysis import (auc, evaluate_retrieval, grid_values,
    join_metrics, validate_landscape)
from protein_sae_pipeline.scoring.iupred import predict


def landscape_fixture():
    rows = []
    for reference in ("one", "two"):
        for flag in ("analogue", "negative"):
            for i in range(8):
                rows.append({"reference_id": reference,
                    "sequence_sha256": hashlib.sha256(f"{reference}/{flag}/{i}".encode()).hexdigest(),
                    "generation_method": "A" if i % 2 else "B",
                    "positional_identity_fraction": .15 if i < 7 else .3,
                    "usalign_tm_score_wt_norm": .7 if flag == "analogue" else .2,
                    "predicted_ptm": .51 + .04 * i if flag == "analogue" else .4 + .05 * i,
                    "sae_cosine_to_wt": .7 + .03 * i if flag == "analogue" else .1 + .08 * i,
                    "esmc_cosine_to_wt": .5 + .04 * i if flag == "analogue" else .2 + .08 * i,
                    "pfam_full_sequence_bit_score": 10 + i if flag == "analogue" else i,
                    "mmseqs2_bit_score": i if i > 3 else 0,
                    "iupred_long_mean": .25 + .02 * i if flag == "analogue" else .2 + .03 * i})
    return validate_landscape(pd.DataFrame(rows))


def test_optimal_matching_against_independent_rectangular_assignment():
    rng = np.random.default_rng(817)
    for n, m in ((0, 3), (1, 3), (3, 3), (3, 8), (7, 11)):
        for _ in range(20):
            negative = np.sort(rng.integers(0, 8, n) / 10)
            positive = np.sort(rng.integers(0, 8, m) / 10)
            selected, cost = match_all_negatives(negative, positive)
            ii, jj = linear_sum_assignment(np.abs(negative[:, None] - positive[None, :]))
            assert len(set(selected)) == n
            assert cost == pytest.approx(np.abs(negative[ii] - positive[jj]).sum())
            assert cost == pytest.approx(np.abs(negative - positive[selected]).sum())
    assert match_all_negatives([.5, .5], [.5, .5, .5])[0].tolist() == [0, 1]
    for n, p in (([0, 1], [0]), ([1, 0], [0, 1]), ([np.nan], [0])):
        with pytest.raises(ValueError):
            match_all_negatives(n, p)


def test_negative_retention_gate_and_no_reuse_are_target_specific():
    full = landscape_fixture()
    remote = full.loc[full.is_remote]
    result = match_ptm(remote)
    assert not result.cohort.duplicated(["reference_id", "sequence_sha256"]).any()
    expected = remote.loc[remote.structural_class.eq("negative") & remote.predicted_ptm.ge(.5)]
    observed = result.cohort.loc[result.cohort.structural_class.eq("negative")]
    assert set(zip(expected.reference_id, expected.sequence_sha256)) == set(zip(observed.reference_id, observed.sequence_sha256))
    assert result.cohort.groupby("pair_id").size().eq(2).all()
    assert result.cohort.groupby("pair_id").structural_class.nunique().eq(2).all()
    # Changing ranking scores cannot change cohort membership or pairs.
    changed = remote.copy()
    changed["sae_cosine_to_wt"] = np.arange(len(changed))
    pd.testing.assert_frame_equal(result.pairs, match_ptm(changed).pairs)


def test_strict_cutoffs_full_roc_endpoint_and_half_credit_auc():
    recall, fpr, threshold = grid_values(np.array([1, 2, 3]), np.array([0, 1, 1, 2]), [0, .25, 1])
    assert threshold[:2].tolist() == [2, 1]
    assert fpr.tolist() == [0, .25, 1]
    assert recall.tolist() == pytest.approx([1/3, 2/3, 1])
    assert auc([1, 1], [1, 1]) == .5


def test_bootstrap_recalibrates_and_reuses_draws_across_methods():
    full = landscape_fixture()
    remote = full.loc[full.is_remote].copy()
    remote["esmc_cosine_to_wt"] = remote.sae_cosine_to_wt
    for frame, paired in ((remote, False), (match_ptm(remote).cohort, True)):
        result = evaluate_retrieval(frame, endpoint_replicates=12, roc_replicates=8, seed=12, matched=paired)
        assert (result.endpoints.observed_fpr <= result.endpoints.target_fpr + 1e-12).all()
        differences = result.paired_differences.loc[result.paired_differences.method.eq("SAE")]
        assert differences.ci_lower.eq(0).all() and differences.ci_upper.eq(0).all()
        for reference in frame.reference_id.unique():
            assert result.bootstrap_roc[reference].shape[0] == 8
        repeat = evaluate_retrieval(frame, endpoint_replicates=12, roc_replicates=8, seed=12, matched=paired)
        pd.testing.assert_frame_equal(result.endpoints, repeat.endpoints)


def test_metrics_join_fails_closed_for_missing_or_duplicate_keys():
    full = landscape_fixture().drop(columns="iupred_long_mean")
    metrics = landscape_fixture()[["reference_id", "sequence_sha256", "iupred_long_mean"]]
    with pytest.raises(ValueError, match="cover"):
        join_metrics(full, metrics.iloc[:-1], ["iupred_long_mean"])
    with pytest.raises(ValueError, match="duplicate"):
        join_metrics(full, pd.concat([metrics, metrics.iloc[:1]]), ["iupred_long_mean"])


def test_iupred_preserves_smoothed_overshoot_and_residue_order():
    class Library:
        @staticmethod
        def iupred(sequence, mode, smoothing):
            return [[-.02, 1.01, .5, .51] + [.25] * (len(sequence) - 4)]
    result = predict(Library, "ACDEFGHIKLMNPQRSTVWY")
    assert result["scores"][:4] == [-.02, 1.01, .5, .51]
    assert result["summary"]["residues_outside_unit_interval"] == 2
    assert result["summary"]["fraction_disordered"] == pytest.approx(2/20)


def test_cli_creates_cohort_figures_and_appendix_from_fictional_rows(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("confidence_cli", root / "scripts/make_confidence_figures.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    full = landscape_fixture()
    master, metrics = tmp_path / "master.csv", tmp_path / "iupred.csv"
    full.drop(columns="iupred_long_mean").to_csv(master, index=False)
    full[["reference_id", "sequence_sha256", "iupred_long_mean"]].to_csv(metrics, index=False)
    out = tmp_path / "out"
    assert cli.main(["--master-table", str(master), "--iupred-table", str(metrics),
        "--output-root", str(out), "--endpoint-replicates", "4", "--roc-replicates", "3",
        "--formats", "pdf"]) == 0
    for relative in ("figure4/combined_ptm_recall_only.pdf", "appendix/iupred_vertical_violins.pdf",
        "appendix/score_distributions_full_and_matched.pdf", "appendix/full_and_ptm_matched_rocs.pdf",
        "figure3/figure3_roc_by_generator.pdf", "confidence_figure_manifest.json"):
        assert (out / relative).is_file()
    # A partial rerun must not attribute old appendix/bootstrap files to it.
    assert cli.main(["--master-table", str(master), "--iupred-table", str(metrics),
        "--output-root", str(out), "--sections", "iupred", "--prepare-only"]) == 0
    manifest = json.loads((out / "confidence_figure_manifest.json").read_text())
    assert set(manifest["outputs"]) == {"confidence_data/cohort_counts.csv",
        "confidence_data/iupred_distribution_summary.csv", "confidence_data/iupred_plot_data.csv.gz",
        "confidence_data/iupred_auc.csv"}
