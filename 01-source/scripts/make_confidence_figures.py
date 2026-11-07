#!/usr/bin/env python3
"""Generate IUPred/pTM controls, recalibrated recall, and reviewer appendix figures.

Sequences and structures are inputs to the existing scoring pipeline. This
command starts from a complete scored landscape, not a prefiltered cohort.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from protein_sae_pipeline.cohorts import construct_cohorts
from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.confidence import match_ptm
from protein_sae_pipeline.io import atomic_json, sha256_file
from protein_sae_pipeline.figures.common import write_frame
from protein_sae_pipeline.figures.figure3 import SCORE_COLUMNS
from protein_sae_pipeline.figures.confidence_controls import render_confidence_controls, render_iupred
from protein_sae_pipeline.figures.reviewer_appendix import (
    landscape, render_score_appendix, render_rocs, render_recall_summaries,
)
from protein_sae_pipeline.reviewer_analysis import (
    PAPER_LABELS, Retrieval, auc, distribution_summary, evaluate_retrieval, grid_values,
    join_metrics, read_table, score_separation, validate_landscape,
)


def point_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reference, target in frame.groupby("reference_id", sort=True):
        for method, column in SCORE_COLUMNS.items():
            case = target.loc[target.structural_class.eq("analogue"), column].to_numpy()
            control = target.loc[target.structural_class.eq("negative"), column].to_numpy()
            recall, fpr, threshold = grid_values(case, control, [.01, .05])
            for i, budget in enumerate((.01, .05)):
                rows.append({"reference_id": reference, "method": method, "target_fpr": budget,
                    "recall": recall[i], "observed_fpr": fpr[i], "threshold": threshold[i],
                    "case_count": len(case), "control_count": len(control)})
    return pd.DataFrame(rows)


def export_retrieval(result: Retrieval, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, pd.DataFrame):
            write_frame(root / (field.name + ".csv.gz"), value)
    # Compact arrays retain shared replicate indices across methods/grid points.
    np.savez_compressed(root / "roc_bootstrap.npz", **result.bootstrap_roc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional frozen pipeline config: checks all expected cohort counts")
    parser.add_argument("--confidence-table", type=Path, help="Keyed table containing predicted_ptm, if absent from master")
    parser.add_argument("--iupred-table", type=Path, help="Keyed CSV(.gz)/Parquet with iupred_long_mean")
    parser.add_argument("--sections", nargs="+", choices=("controls", "iupred", "appendix", "retrieval"),
                        default=["controls", "iupred", "appendix", "retrieval"])
    parser.add_argument("--endpoint-replicates", type=int, default=2000)
    parser.add_argument("--roc-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png", "svg"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--expected-master-sha256", help="Optional byte-level binding to a frozen input table")
    args = parser.parse_args(argv)
    if min(args.endpoint_replicates, args.roc_replicates) < 1 or args.seed < 0:
        parser.error("Replicate counts must be positive and seed nonnegative")
    master_sha = sha256_file(args.master_table)
    if args.expected_master_sha256 and master_sha != args.expected_master_sha256:
        parser.error("Master-table checksum differs from --expected-master-sha256")
    full = validate_landscape(read_table(args.master_table))
    sections = set(args.sections)
    if args.confidence_table:
        full = join_metrics(full, read_table(args.confidence_table), ["predicted_ptm"])
    remote = full.loc[full.is_remote].copy().reset_index(drop=True)
    if remote.empty:
        parser.error("No designs satisfy WT identity <30%")
    config = load_config(args.config) if args.config else None
    if config:
        construct_cohorts(full, config)
        labels = {r: str(v["display_name"]) for r, v in config.raw["references"].items()}
    else:
        order = [r for r in PAPER_LABELS if r in set(full.reference_id)]
        order += sorted(set(full.reference_id) - set(order))
        labels = {r: PAPER_LABELS.get(r, r) for r in order}
    matched = None
    if sections & {"controls", "appendix"}:
        matched = match_ptm(remote)
    disorder = None
    if "iupred" in sections:
        if args.iupred_table is None:
            parser.error("IUPred section requires --iupred-table (score_iupred.py produces this table)")
        metrics = read_table(args.iupred_table)
        if "smoothing" in metrics and not metrics.smoothing.eq("medium").all():
            parser.error("The paper IUPred figure requires medium smoothing")
        disorder = join_metrics(remote, metrics, ["iupred_long_mean"])
    if "retrieval" in sections:
        for generator, frame in remote.groupby("generation_method", sort=True):
            for reference in labels:
                if set(frame.loc[frame.reference_id.eq(reference), "structural_class"]) != {"analogue", "negative"}:
                    parser.error(f"{reference}/{generator}: generator-specific retrieval requires both classes")
    # Validate inputs before producing any output. The final manifest is written
    # only after all requested analyses and renders succeed.
    out = args.output_root
    data = out / "confidence_data"
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "confidence_figure_manifest.json"
    manifest_path.unlink(missing_ok=True)
    counts = []
    cohorts = {"full_landscape": full, "remote": remote}
    if matched is not None:
        cohorts["matched"] = matched.cohort
        write_frame(data / "ptm_ge_050_pairs.csv.gz", matched.pairs)
        write_frame(data / "ptm_ge_050_cohort.csv.gz", matched.cohort)
        write_frame(data / "matching_quality.csv", matched.diagnostics)
        point_frames = [point_estimates(frame).assign(cohort=name) for name, frame in
                        (("remote", remote), ("matched", matched.cohort))]
        points = pd.concat(point_frames, ignore_index=True)
        write_frame(data / "recall_full_and_matched.csv", points)
        for name, frame in (("remote", remote), ("matched", matched.cohort)):
            write_frame(data / f"ptm_{name}_distribution_summary.csv", distribution_summary(frame, "predicted_ptm"))
        if "controls" in sections and not args.prepare_only:
            render_confidence_controls(remote, matched.cohort, points, labels=labels, output=out, formats=args.formats)
    for name, frame in cohorts.items():
        group = frame.groupby(["reference_id", "generation_method", "structural_class"], sort=True).size().rename("n").reset_index()
        group.insert(0, "cohort", name)
        counts.append(group)
    write_frame(data / "cohort_counts.csv", pd.concat(counts, ignore_index=True))
    if matched is not None:
        auc_rows = []
        for name, frame in (("remote", remote), ("matched", matched.cohort)):
            for reference, target in frame.groupby("reference_id", sort=True):
                auc_rows.append({"cohort": name, "reference_id": reference,
                    "ptm_auc": auc(target.loc[target.structural_class.eq("analogue"), "predicted_ptm"],
                                   target.loc[target.structural_class.eq("negative"), "predicted_ptm"])})
        write_frame(data / "ptm_auc.csv", pd.DataFrame(auc_rows))
    if disorder is not None:
        write_frame(data / "iupred_distribution_summary.csv", distribution_summary(disorder, "iupred_long_mean"))
        write_frame(data / "iupred_plot_data.csv.gz", disorder[["reference_id", "sequence_sha256", "generation_method",
            "structural_class", "iupred_long_mean"]])
        rows = [{"reference_id": reference, "auc_higher_disorder_predicts_analogue":
            auc(target.loc[target.structural_class.eq("analogue"), "iupred_long_mean"],
                target.loc[target.structural_class.eq("negative"), "iupred_long_mean"])}
            for reference, target in disorder.groupby("reference_id", sort=True)]
        write_frame(data / "iupred_auc.csv", pd.DataFrame(rows))
        if not args.prepare_only:
            render_iupred(disorder, labels=labels, output=out, formats=args.formats)
    settings = dict(seed=args.seed, endpoint_replicates=args.endpoint_replicates, roc_replicates=args.roc_replicates)
    if "appendix" in sections:
        write_frame(data / "score_separation_class_statistics.csv", score_separation(
            {"full_landscape": full, "matched": matched.cohort}))
        # Scatter cutoffs are calibrated on the displayed all-identity/full or
        # matched rows; they differ from the remote benchmark cutoffs.
        scatter_points = pd.concat([point_estimates(frame).assign(cohort=name) for name, frame in
            (("full_landscape", full), ("matched", matched.cohort))], ignore_index=True)
        write_frame(data / "scatter_fpr_thresholds.csv", scatter_points)
        pooled = {}
        for name, frame in (("Full (<30% identity)", remote), ("Matched subsample", matched.cohort)):
            print(f"Bootstrap: {name} ({len(frame):,} rows)", flush=True)
            result = evaluate_retrieval(frame, matched=name.startswith("Matched"), **settings)
            pooled[name] = result
            export_retrieval(result, data / ("matched_retrieval" if name.startswith("Matched") else "pooled_retrieval"))
        if not args.prepare_only:
            landscape(full, labels, out, args.formats)
            render_score_appendix(full, matched.cohort, labels, out, args.formats)
            render_rocs(pooled, labels, out, max_fpr=1, stem="appendix/full_and_ptm_matched_rocs", formats=args.formats)
    if "retrieval" in sections:
        by_generator = {}
        for index, (generator, frame) in enumerate(remote.groupby("generation_method", sort=True)):
            print(f"Bootstrap: {generator} ({len(frame):,} rows)", flush=True)
            result = evaluate_retrieval(frame, **settings)
            by_generator[generator] = result
            export_retrieval(result, data / f"generator_{index}")
        if not args.prepare_only:
            render_rocs(by_generator, labels, out, stem="figure3/figure3_roc_by_generator", formats=args.formats)
            render_recall_summaries(by_generator, labels, out, args.formats)
    inputs = {str(args.master_table): master_sha}
    for path in (args.config, args.confidence_table, args.iupred_table):
        if path is not None:
            inputs[str(path)] = sha256_file(path)
    # Enumerate only files produced by this invocation, excluding stale files
    # from other sections/formats in a reused output directory.
    names = ["cohort_counts.csv"]
    if matched is not None:
        names += ["ptm_ge_050_pairs.csv.gz", "ptm_ge_050_cohort.csv.gz", "matching_quality.csv",
                  "recall_full_and_matched.csv", "ptm_remote_distribution_summary.csv",
                  "ptm_matched_distribution_summary.csv", "ptm_auc.csv"]
    if disorder is not None:
        names += ["iupred_distribution_summary.csv", "iupred_plot_data.csv.gz", "iupred_auc.csv"]
    retrieval_dirs = []
    if "appendix" in sections:
        names += ["score_separation_class_statistics.csv", "scatter_fpr_thresholds.csv"]
        retrieval_dirs += ["pooled_retrieval", "matched_retrieval"]
    if "retrieval" in sections:
        retrieval_dirs += [f"generator_{i}" for i in range(remote.generation_method.nunique())]
    for directory in retrieval_dirs:
        names += [f"{directory}/{field.name}.csv.gz" for field in fields(Retrieval) if field.name != "bootstrap_roc"]
        names += [f"{directory}/roc_bootstrap.npz"]
    output_paths = [data / name for name in names]
    if not args.prepare_only:
        stems = []
        if "controls" in sections:
            stems += ["figure4/combined_ptm_recall_only"]
        if "iupred" in sections:
            stems += ["appendix/iupred_vertical_violins"]
        if "retrieval" in sections:
            stems += ["figure3/figure3_roc_by_generator", "figure3/figure3_recall_by_generator"]
        if "appendix" in sections:
            stems += ["appendix/six_protein_landscapes", "appendix/six_protein_hmm_sequence_identity_scatters",
                      "appendix/full_and_ptm_matched_rocs"]
            for method in ("sae", "esmc", "hmm", "mmseqs2"):
                stems += [f"appendix/six_protein_{method}_tm_scatters", f"appendix/full_{method}_tm_scatters",
                          f"appendix/matched_{method}_tm_scatters"]
            output_paths += [out / "appendix/score_distributions_full_and_matched.pdf"]
        output_paths += [out / f"{stem}.{extension}" for stem in stems for extension in args.formats]
    atomic_json(manifest_path, {
        "schema_version": 1, "complete": True, "sections": sorted(sections), "prepare_only": args.prepare_only,
        "dataset_id": config.dataset_id if config else None, "fictional": config.is_example if config else None,
        "inputs": inputs, "reference_labels": labels, "generation_order": sorted(remote.generation_method.unique()),
        "rows": {name: len(frame) for name, frame in cohorts.items()},
        "score_columns": SCORE_COLUMNS,
        "roc_cube_axes": ["replicate", "method in score_columns order", "fpr_grid"],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "code_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
            for path in [Path(__file__).resolve(),
                *sorted((Path(__file__).resolve().parents[1] / "src/protein_sae_pipeline").rglob("*.py"))]},
        "rules": {"remote": "positional_identity_fraction < 0.30", "analogue": "WT-normalized TM-score >= 0.50",
            "matching": "pTM >=0.50, within target; retain all negatives; distinct positives; minimum total absolute pTM distance",
            "matching_ties": "sort pTM then sequence SHA256; prefer earlier sorted positive on equal cost",
            "matching_caliper": None, "matching_uses_retrieval_scores": False,
            "retrieval": "strict score > cutoff; thresholds recalibrated within each displayed cohort/target/method",
            "scatter_scope": "full all-identity landscape versus matched; benchmark ROCs use <30% identity",
            "iupred": "whole-sequence long-mode mean; unmatched <30%-identity cohort; no confidence gate"},
        "bootstrap": {**settings, "full": "sequence rows within target/class/generator; same draw across methods and FPRs",
            "matched": "complete frozen pairs within target/ordered generator combination; matching not refitted",
            "interval": "pointwise 2.5/97.5 percentiles; conditional on observed targets/labels"},
        "outputs": {str(p.relative_to(out)): sha256_file(p) for p in sorted(set(output_paths))},
    })
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
