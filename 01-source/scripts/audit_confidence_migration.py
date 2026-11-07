#!/usr/bin/env python3
"""Read-only parity audit against an existing private mutavault checkout.

Writes aggregate checks only; never copies sequences, scores or pair identities
into the public repository. It is optional and not needed to run the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from protein_sae_pipeline.confidence import match_ptm
from protein_sae_pipeline.figures.figure3 import SCORE_COLUMNS
from protein_sae_pipeline.io import atomic_json, sha256_file
from protein_sae_pipeline.reviewer_analysis import (distribution_summary, grid_values,
    join_metrics, read_table, validate_landscape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutavault-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sieve = args.mutavault_root / "sieve"
    master = sieve / "paper_figures/interp4disc/appendix/full_landscape_data/landscape_scores_complete.csv.gz"
    root = sieve / "structure_confidence_controls/paper_figs/threshold_balanced_matched"
    pair_path = root / "data/ptm_ge_050_pairs.csv.gz"
    comparison_path = root / "data/original_paper_vs_ptm_matched_recall_1pct_plot_data.csv"
    matched_points_path = root / "data/all_strategy_recall_at_1pct_5pct.csv"
    full = validate_landscape(read_table(master))
    remote = full.loc[full.is_remote]
    result = match_ptm(remote)
    previous = read_table(pair_path)
    old_pairs = set(zip(previous.protein, previous.positive_sequence_sha256, previous.negative_sequence_sha256))
    new_pairs = set(zip(result.pairs.reference_id, result.pairs.positive_sequence_sha256, result.pairs.negative_sequence_sha256))
    if old_pairs != new_pairs:
        raise AssertionError(f"Pair identities changed: {len(old_pairs - new_pairs)} old, {len(new_pairs - old_pairs)} new")
    methods = {"sae": "SAE", "raw_esmc": "ESM-C", "hmm": "Pfam", "mmseqs2": "MMseqs2"}
    comparison = read_table(comparison_path)
    checked = 0
    for row in comparison.itertuples():
        for frame, suffix in ((remote, "before"), (result.cohort, "after")):
            target = frame.loc[frame.reference_id.eq(row.protein)]
            column = SCORE_COLUMNS[methods[row.method]]
            recall, fpr, threshold = grid_values(target.loc[target.structural_class.eq("analogue"), column],
                target.loc[target.structural_class.eq("negative"), column], [.01])
            for name, value in (("recall", recall[0]), ("observed_fpr", fpr[0]), ("threshold", threshold[0])):
                np.testing.assert_allclose(value, getattr(row, name + "_" + suffix), rtol=0, atol=1e-12)
            checked += 1
    matched_points = read_table(matched_points_path)
    for row in matched_points.loc[matched_points.strategy.eq("ptm_ge_050")].itertuples():
        target = result.cohort.loc[result.cohort.reference_id.eq(row.protein)]
        column = SCORE_COLUMNS[methods[row.method]]
        recall, fpr, threshold = grid_values(target.loc[target.structural_class.eq("analogue"), column],
            target.loc[target.structural_class.eq("negative"), column], [row.fpr_budget])
        for value, expected in ((recall[0], row.recall), (fpr[0], row.observed_fpr), (threshold[0], row.threshold)):
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)
        checked += 1
    metrics_path = sieve / "audit/results/full/sequence_metrics.parquet"
    summary_path = root / "paper_sampling_plots/iupred_vertical_violins/iupred_distribution_summary.csv"
    metrics = pd.read_parquet(metrics_path, columns=["protein_key", "sequence_sha256", "iupred_long_mean"])
    disorder = join_metrics(remote, metrics, ["iupred_long_mean"])
    summary = distribution_summary(disorder, "iupred_long_mean").sort_values(["reference_id", "structural_class"]).reset_index(drop=True)
    previous_summary = read_table(summary_path).rename(columns={"protein": "reference_id"})
    previous_summary["structural_class"] = previous_summary.structural_class.replace({"analog": "analogue"})
    previous_summary = previous_summary.sort_values(["reference_id", "structural_class"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(summary, previous_summary[summary.columns], check_exact=False, rtol=0, atol=1e-12)
    sources = [master, pair_path, comparison_path, matched_points_path, metrics_path, summary_path]
    report = {"status": "passed", "full_rows": len(full), "remote_rows": len(remote),
        "matched_rows": len(result.cohort), "exact_pair_identities": len(new_pairs),
        "recall_fpr_threshold_comparisons": checked, "iupred_class_distributions": len(summary),
        "tolerance": 1e-12, "bootstrap_intervals": "recomputed by portable engine; not a bitwise CI parity claim",
        "input_sha256": {str(p.relative_to(sieve)): sha256_file(p) for p in sources}}
    atomic_json(args.output, report)
    print(report["status"], f"{len(new_pairs):,} exact pairs; {checked} operating points; {len(summary)} IUPred distributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
