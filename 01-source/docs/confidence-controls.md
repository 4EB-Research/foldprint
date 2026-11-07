# IUPred and pTM confidence controls

`scripts/make_confidence_figures.py` adds the confidence controls and related
appendix panels in the November 2026 reviewer draft. It runs independently of
the existing Figures 1-3 entry points. It accepts the canonical master table
from `assemble_master.py` or the historical complete-landscape column names
listed below. There are no imports from mutavault and no absolute paths in the
analysis code. Cross-predictor agreement is outside this migration.

## Fictional example

`scripts/run_example.py` now runs these analyses after the original figures.
The committed `example/inputs/fictional_confidence.csv` and
`fictional_iupred.csv` contain invented scores, not IUPred or ESMFold output.
They exercise the plots without requiring a licensed predictor. Regenerate
them with `python scripts/generate_confidence_example.py --data-root example`.

Once the ordinary example master has been assembled, run only the new analyses:

```bash
uv run python scripts/make_confidence_figures.py \
  --config configs/example.yaml \
  --master-table outputs/example/tables/master.csv.gz \
  --confidence-table example/inputs/fictional_confidence.csv \
  --iupred-table example/inputs/fictional_iupred.csv \
  --output-root outputs/example \
  --endpoint-replicates 100 --roc-replicates 50
```

These small bootstrap counts are for exercising the software. The default
for analysis is 2,000 endpoint and 2,000 ROC replicates, as described in the
attached draft. The historical generator recall summaries used 10,000
endpoint replicates; use `--endpoint-replicates 10000` for that setting.
The manifest always records the actual counts and seed. The portable engine
defines a new deterministic stream with shared samples across methods and
FPRs; confidence interval endpoints are recomputed, not copied from old files.

## Calculate IUPred3 predictions

Obtain the official standalone IUPred3 distribution from its authors and pass
its directory explicitly. The licensed predictor and parameter files are not
included in this repository. The adapter is adapted from
`sieve/audit/iupred3_adapter.py`, including its package fingerprint definition,
per-residue outputs, and whole-sequence summaries.

```bash
uv run python scripts/score_iupred.py \
  --config CONFIG --data-root DATA --output-root OUT \
  --iupred-package /path/to/official/iupred3 --workers 4
```

The paper setting is **long mode, medium smoothing**. Optional `--modes long
short` computes both modes. This step scores all validated input sequences;
the plotting command applies the <30% identity restriction afterwards.

Outputs under `OUT/iupred/` are:

* `sequence_metrics.csv.gz`: one row per canonical candidate key, including
  `iupred_long_mean`, fraction of residues scoring strictly above 0.5, longest
  run, extrema, and counts outside the unit interval;
* `residue_profiles.jsonl.gz`: unmodified scores in sequence order;
* `official_notice.txt`: the official CLI attribution, also printed at runtime;
* `provenance.json`: settings, package file hashes/fingerprint and output hashes;
* `cache/`: independently hashed batches, reused on an identical resumed run.

Batch identities bind the sequences, predictor fingerprint, mode, smoothing
and adapter/workflow source hashes. A completed cache batch is published only
after its profile file and checksum have been written. Failed runs can be
restarted with the same command. Parallel workers have separate predictor
instances and process independent sequences. Smoothed residue scores are
**never clipped**: the official smoothing can overshoot 0-1. The plotted value
is a sequence mean, not the fraction of residues predicted to be disordered.

## Run on a complete scored landscape

```bash
uv run python scripts/make_confidence_figures.py \
  --master-table /path/to/complete_master.csv.gz \
  --iupred-table OUT/iupred/sequence_metrics.csv.gz \
  --output-root OUT
```

The master must include `reference_id`, `sequence_sha256`, `generation_method`,
`positional_identity_fraction`, `usalign_tm_score_wt_norm`, and the four score
columns (`sae_cosine_to_wt`, `esmc_cosine_to_wt`,
`pfam_full_sequence_bit_score`, `mmseqs2_bit_score`). pTM can be in the master
as `predicted_ptm`, or supplied through `--confidence-table` with the canonical
key and that column. The IUPred table uses the same key and
`iupred_long_mean`. CSV and gzipped CSV require only the standard dependencies;
reading a historical Parquet table additionally requires a pandas Parquet
engine such as PyArrow in the execution environment.

Historical aliases are accepted explicitly:

| Historical column | Canonical column |
| --- | --- |
| `protein_key` | `reference_id` |
| `origin` | `generation_method` |
| `hamming_sequence_identity` | `positional_identity_fraction` |
| `usalign_alignment_optimized_tm_score_wt_norm` | `usalign_tm_score_wt_norm` |
| `raw_esmc_layer60_cosine_to_wt` | `esmc_cosine_to_wt` |

Conflicting aliases, duplicate/missing keys, missing scores, nonfinite values
and incomplete supplemental coverage fail validation. The optional `--config`
enforces the existing frozen full/remote per-target counts and supplies display
names/order. `--expected-master-sha256` can bind the run to an audited input
file. Real sequences and per-design results remain external input/output data;
none are supplied by the public example.

Use `--sections controls iupred` to generate just Figure 4 and the disorder
appendix, with no retrieval bootstrap or score-scatter rendering. Individual
sections are also supported. `--prepare-only` writes data without images.
`--formats pdf png svg` controls individual figure exports. The eight-page
score-distribution review PDF is always produced when appendix rendering is
requested.

## Matching and uncertainty definitions

Both retrieval cohorts use exact WT positional identity **<0.30**. Structural
analogs have reference-normalized TM-score **>=0.50**. The matched cohort:

1. Applies the inclusive pTM **>=0.50** gate.
2. Retains every passing structural negative within each target.
3. Selects a distinct passing analog for every negative, minimizing the sum of
   absolute pTM differences using exact sorted dynamic programming.

There is no distance caliper, no replacement, and no generator restriction.
The sort order is pTM then sequence SHA256; equal-cost assignments prefer
earlier sorted positives. Retrieval scores never enter cohort selection.
An insufficient analog pool or a target with no passing negatives fails
explicitly. `matching_quality.csv` records retained counts, pair distances and
the same-generator fraction; `ptm_ge_050_pairs.csv.gz` retains the exact pairs.

Recall thresholds are recalibrated **separately within each target, method and
cohort**, using `floor(q*N_negatives)` allowed false positives, with retrieval
requiring **score > cutoff**. Ties remain negative. Every observed threshold is
retained for empirical ROC curves, including tied no-hit scores. The full-range
endpoint retrieves all rows. No interpolation is used across score ties.

The full and generator-specific benchmarks resample sequence rows within
target/class/generator. The matched bootstrap resamples **complete frozen
pairs**, stratified by target and the ordered analog-generator/negative-generator
combination. This preserves the actual matching analysis implementation.
The draft's generic sequence-bootstrap caption for the matched ROC should be
updated to describe pairs. Matching is held fixed, not refitted. Every draw is
shared across all four methods, 1%/5% endpoints and the ROC grid. Intervals are
pointwise 95% percentiles, conditional on the observed targets and labels.

## Outputs and figure scope

| Output | Cohort and analysis |
| --- | --- |
| `figure4/combined_ptm_recall_only.*` | Full remote vs pTM-thresholded matched distributions and recalibrated 1%-FPR recall |
| `appendix/iupred_vertical_violins.*` | Unmatched remote designs, no confidence gate; medians, IQRs and means |
| `figure3/figure3_roc_by_generator.*` | Separate low-FPR ROC rows by generator |
| `figure3/figure3_recall_by_generator.*` | Separate 1%/5% recall and bootstrap intervals by generator |
| `appendix/six_protein_landscapes.*` | Complete all-identity sequence/structure landscapes (A1) |
| `appendix/six_protein_{method}_tm_scatters.*` | Complete all-identity score/structure landscapes (A2-A5) |
| `appendix/full_*_tm_scatters.*`, `matched_*_tm_scatters.*` | Same full-sample axes/edges, with cohort-specific FPR cutoffs |
| `appendix/score_distributions_full_and_matched.pdf` | Eight method/cohort pages in full-then-matched order |
| `appendix/full_and_ptm_matched_rocs.*` | Full-range pooled remote and fixed-pair matched ROCs (A6) |
| `appendix/six_protein_hmm_sequence_identity_scatters.*` | Complete HMM/identity landscape; 30% cutoff is a reference line |

`confidence_data/` contains cohort membership/pairs, counts by
target/generator/class, pTM AUROCs and distribution summaries, IUPred
distributions/AUROCs, all plotted operating points, scatter cutoffs, and
standardized score separation. The separation statistic is
`(mean_analog - mean_negative) / sqrt((var_analog + var_negative)/2)`,
with unbiased sample variances and equal class weights. Zero variance gives an
undefined/empty D field, not an infinite value.

Retrieval subdirectories contain empirical curves, bands, endpoints, AUROCs,
endpoint bootstrap draws, paired differences relative to ESM-C, and compact ROC
replicate cubes. `confidence_figure_manifest.json` binds the inputs, rules,
settings and output hashes. It is written only after the requested work succeeds.

The full scatter cohort includes **all identities** (158,233 designs for the
paper). The unfiltered retrieval/ROC/IUPred cohort uses **<30% identity**
(66,653 designs). The saved matched cohort has 8,918 designs in 4,459 pairs.
These are different scopes; the renderer does not substitute one for another.
Full-scatter FPR cutoffs use the all-identity negatives actually displayed.

## Optional migration parity audit

```bash
python scripts/audit_confidence_migration.py \
  --mutavault-root /path/to/private/mutavault \
  --output outputs/migration_parity.json
```

This read-only audit checks all 4,459 pair identities, 96 saved operating points
(recall, FPR and threshold), and all 12 IUPred class distributions. It writes only
aggregate checks and source hashes. It requires the historical private data and
a Parquet engine and is not part of the public example. Bootstrap intervals are
recomputed with the portable engine; this audit does not claim bitwise CI parity.
`docs/confidence-source-map.json` records the original source paths and hashes
used for the migration.

The recorded aggregate checks are in `confidence-migration-validation.json`.
They cover pair identities, point estimates and IUPred summaries. The migration
also passed the 90-test offline suite, official-CLI checks in both IUPred modes
on the first fictional sequence, and 48-sequence parallel scoring with verified
cache reuse. Real-data rendering was checked with reduced bootstrap counts;
it is a layout/integration check, not a new final interval estimate.
