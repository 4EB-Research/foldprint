#!/usr/bin/env python3
"""Reproduce aggregate concordance results offline from external Harmonize scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from protein_sae_pipeline.concordance import aggregate_results, comparisons
from protein_sae_pipeline.io import atomic_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-predictions-sha256")
    parser.add_argument("--expected-cohort-sha256")
    parser.add_argument(
        "--write-private-figures",
        action="store_true",
        help="Also render per-design Figure AX into ignored/private output storage",
    )
    args = parser.parse_args()
    hashes = {"cohort": sha256_file(args.cohort), "predictions": sha256_file(args.predictions)}
    for name, actual in hashes.items():
        expected = getattr(args, f"expected_{name}_sha256")
        if expected and actual != expected:
            parser.error(f"{name} input checksum mismatch")
    # Input allowlists prevent unrelated private columns from propagating.
    cohort_columns = {
        "id",
        "family",
        "generator",
        "main",
        "existing",
        "selection_bin",
        "sequence_sha256",
    }
    score_columns = {
        "id",
        "model",
        "sample_id",
        "optimized_tm",
        "fixed_tm",
        "prediction_coverage",
        "reference_coverage",
        "normalization_length",
        "sequence_sha256",
    }
    cohort = pd.read_csv(
        args.cohort, usecols=lambda c: c in cohort_columns, float_precision="round_trip"
    )
    scores = pd.read_csv(
        args.predictions, usecols=lambda c: c in score_columns, float_precision="round_trip"
    )
    rows = comparisons(cohort, scores)
    tables = aggregate_results(rows)
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "concordance-manifest.json"
    manifest.unlink(missing_ok=True)
    files = []
    for name, table in tables.items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)
    if args.write_private_figures:
        from protein_sae_pipeline.figures.concordance import render_concordance

        render_concordance(rows, out / "private-figures")
        files.extend(sorted((out / "private-figures").glob("quadrants-*.*")))
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "input_sha256": hashes,
            "designs": int(cohort.id.nunique()),
            "main_designs": int(rows.loc[rows.main, "id"].nunique()),
            "predictions": len(scores),
            "threshold": 0.5,
            "primary_endpoint": "optimized_tm: optimized correspondence, full WT sequence length",
            "secondary_endpoint": "fixed_tm: input correspondence, full WT length, max directional score",
            "ensemble_sizes": {"esmfold": 1, "colabfold": 5, "esmfold2": 5},
            "selection": "Frozen cohort flags/bins; legacy RMSD selection retained; never rebalance after rescoring",
            "interpretation": "Targeted audit; no population, experimental accuracy, foldability or function estimate",
            "ranges": "Min-max of five predictions; not confidence intervals or independent AF2 replicates",
            "inference_submitted": False,
            "raw_structures_recomputed": False,
            "private_figures_written": args.write_private_figures,
            "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in files},
        },
    )
    print(
        f"Verified {len(scores):,} scores; wrote aggregate results for {cohort.id.nunique():,} designs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
