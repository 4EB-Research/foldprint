# Figure and plotted-data map

All main-text figures are generated from the validated analysis tables. The main entry points are:

```bash
python scripts/make_figure_1.py --config CONFIG --data-root DATA --output-root OUT
python scripts/make_figure_2.py --config CONFIG --data-root DATA --output-root OUT
python scripts/make_figure_3.py --config CONFIG --data-root DATA --output-root OUT
```

Each command writes the combined figure, standalone panels, and the underlying plotted data to `OUT/figureN/`.

## Figure 1

Figure 1 describes the focal-reference design landscape and its SAE representation.

Panel 1A plots sequence identity against WT-normalized TM-score across the complete focal-reference landscape. Panel 1B plots TM-score against SAE cosine similarity to WT for the same candidates. Panel 1C summarizes the recurrence and cumulative contribution of WT SAE features in the remote focal-reference cohort.

The corresponding plotted-data files are:

* `figure1a_plot_data.csv.gz`
* `figure1b_plot_data.csv.gz`
* `figure1c_feature_summary.csv.gz`

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

Figure 3 compares retrieval performance across the remote-design cohorts.

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
