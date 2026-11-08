# Figure and plotted-data map

Figures are generated from validated analysis tables. The current paper uses the generator-stratified Figure 3 entry point described below; the older pooled command is retained for compatibility. The paper plotting entry points are:

| Paper figure | Command | Main inputs |
| --- | --- | --- |
| Figure 1 A-D | `scripts/make_figure_1.py` | Canonical master table and sparse representations |
| Figure 2 A-E | `scripts/make_figure_2.py` | Same inputs, configured examples, and WT structure; PyMOL for molecular panels |
| Figure 3 | `scripts/make_confidence_figures.py --sections retrieval` | Complete scored landscape; both classes split by generator |
| Figure 4 | `scripts/make_confidence_figures.py --sections controls` | Complete scored landscape and pTM metadata |
| Figures A1-A6 | `scripts/make_confidence_figures.py --sections appendix` | Complete scored landscape and pTM metadata |
| Figure Sx | `scripts/make_confidence_figures.py --sections iupred` | Complete scored landscape and keyed IUPred3 means |
| Figure AX | `scripts/analyze_concordance.py --write-private-figures` | Frozen cohort and complete externally scored predictions |

```bash
uv run python scripts/make_figure_1.py --config CONFIG --data-root DATA --output-root OUT
uv run python scripts/make_figure_2.py --config CONFIG --data-root DATA --output-root OUT
uv run python scripts/make_confidence_figures.py \
  --master-table MASTER --iupred-table IUPRED --output-root OUT \
  --sections retrieval controls appendix iupred
uv run python scripts/analyze_concordance.py \
  --cohort COHORT --predictions SCORES --output-root OUT/concordance \
  --write-private-figures
```

The Figure 1 and 2 wrappers read `OUT/tables/master.csv.gz` and write their
combined figures, standalone panels and plotted data into `OUT/figureN/`.
Use `scripts/make_figures.py --master-table MASTER --figures 1 2` with the same
config/data/output arguments when the master table lives elsewhere.
The confidence command requires `predicted_ptm` in the master table or a separate
`--confidence-table`; its formats and schemas are documented in
[confidence-controls.md](confidence-controls.md).

These commands analyze supplied inputs; they do not generate sequences or run
structure-prediction jobs. The bundled fictional example checks software
behavior and does not reproduce the paper's numerical results. Real figure
regeneration requires the corresponding managed-access inputs. Biohub Atlas
annotation labels and Figure 3 reference-thumbnail artwork are optional
manuscript additions and are not required for the plots.

## Figure 1

Figure 1 describes the focal-reference design landscape and its SAE representation.

Panel 1A plots sequence identity against WT-normalized TM-score across the complete focal-reference landscape. Panel 1B plots TM-score against SAE cosine similarity to WT for the same candidates. Panel 1C reports class medians from exactly the same full cohort as 1B. Panel 1D summarizes recurrence and cumulative contributions of WT SAE features in the remote focal-reference cohort. Panel B includes class-wise marginal relative densities.

The corresponding plotted-data files are:

* `figure1a_plot_data.csv.gz`
* `figure1b_plot_data.csv.gz`
* `figure1c_class_medians.csv`
* `figure1d_feature_summary.csv.gz` (also emitted under the historical `figure1c_feature_summary.csv.gz` alias)

## Figure 2

Figure 2 contains the gpD interpretability analyses.

Panel 2A shows WT activation traces for the five highest max-pooled SAE features together with their structural locations. Panel 2B compares selected feature traces across WT and configured candidate examples. Panel 2C shows the WT focal-position features and their prevalence across the remote cohort. Panel 2D shows the number of position-unique WT features across the sequence, and Panel 2E shows the structural context of the configured focal position.

The main plotted-data outputs are:

* `figure2a_wt_feature_traces.csv.gz`
* `figure2a_wt_pdb_residue_colouring.csv.gz`
* `figure2b_example_traces.csv.gz`
* `figure2b_examples.csv`
* `figure2c_wt_position_features.csv.gz`
* `figure2c_position_prevalence.csv`
* `figure2d_position_unique_plot_data.csv`

PyMOL can optionally be used for ray-traced structural renders:

```bash
python scripts/make_figure_2.py \
  --config CONFIG \
  --data-root DATA \
  --output-root OUT \
  --pymol-executable /path/to/pymol
```

If PyMOL is unavailable, the structural image slots are left blank; no 2D
schematic is substituted.

Additional audit files record the position-unique features and feature-selection statistics used in the interpretability analysis. External feature annotations are not included.

## Figure 3

The current paper Figure 3 compares retrieval separately within ESM3 and RFdiffusion/ProteinMPNN, splitting both classes by generator. Run `make_confidence_figures.py --sections retrieval`; output is `figure3/figure3_roc_by_generator.*`. Reference-structure thumbnails are external manuscript artwork.

The legacy `make_figure_3.py` output described next is pooled and does not match the revised main-text layout.

Panel 3A shows low-FPR ROC curves for SAE, ESM-C, Pfam, and MMseqs2 with paired bootstrap intervals. Panel 3B reports recall at 1% and 5% FPR with paired stratified intervals.

The main plotted-data outputs are:

* `figure3_roc_plot_data.csv.gz`
* `figure3_roc_bands.csv.gz`
* `figure3_operating_points.csv`
* `figure3_endpoint_intervals.csv`

Additional outputs contain the full bootstrap replicates and paired differences relative to ESM-C.

The bundled fictional example is intended only to test the analysis and plotting pipeline and should not be interpreted biologically.

## Confidence-control revision and appendix

`scripts/make_confidence_figures.py` is the entry point for the reviewer
revision's pTM/IUPred controls and related recall/appendix panels. It also
provides generator-specific Figure 3 curves and recall summaries; the original
Figure 3 command above retains its pooled output. The revised output files and
their exact cohort definitions are mapped in
[confidence-controls.md](confidence-controls.md#outputs-and-figure-scope).

The separate `score_iupred.py` command calculates long-mode, medium-smoothed
IUPred3 residue scores and whole-sequence means. pTM matching is computed from
the supplied confidence metadata, never from retrieval scores.

## Cross-predictor Figure AX

`analyze_concordance.py` produces aggregate results only by default. With private scored inputs and `--write-private-figures`, it additionally renders the two-by-two main-cohort WT-score panels (ESM3 above RFdiffusion/ProteinMPNN; ESMFold2 left, AF2 right). `optimized_tm` is the paper endpoint and `fixed_tm` is a separate sensitivity endpoint. Per-design images and inputs are not public release assets. The command defaults to aggregate tables unless the figure flag is supplied.

Required cohort columns are `id`, `family`, `generator`, `main`, `existing`,
and `selection_bin`. Required prediction columns are `id`, `model`, `sample_id`,
`optimized_tm`, `fixed_tm`, `prediction_coverage`, `reference_coverage`, and
`normalization_length`. Each design requires one `esmfold` score and five scores
each for `colabfold` and `esmfold2`. Missing or duplicate predictions fail
validation. The renderer uses the frozen main-cohort flag and the existing
`esm3` / `rfd-mpnn` and six-target vocabulary.

Concordance inputs must already use their documented full-WT-sequence
normalization and reference convention, including the PDZ3 Asp332 override and
excluded reference position. Plotting does not rescore coordinates or change
retrieval labels. Optional `--expected-cohort-sha256` and
`--expected-predictions-sha256` bind an invocation to frozen input tables.

The Figure A6 caption is present in the supplied draft but its image is absent.
The appendix command nevertheless supplies the full-range ROC figure at
`appendix/full_and_ptm_matched_rocs.*`.
