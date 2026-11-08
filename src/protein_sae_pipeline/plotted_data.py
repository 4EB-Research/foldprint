"""Definitions for the plotted-data manifest produced with Figures 1--3."""

from __future__ import annotations


REQUIRED_PLOTTED_DATA_PATHS = frozenset(
    {
        "figure1/data/figure1a_plot_data.csv.gz",
        "figure1/data/figure1b_plot_data.csv.gz",
        "figure1/data/figure1c_feature_summary.csv.gz",
        "figure1/data/figure1c_candidate_score_audit.csv.gz",
        "figure2/data/figure2a_wt_feature_traces.csv.gz",
        "figure2/data/figure2a_wt_pdb_residue_colouring.csv.gz",
        "figure2/data/figure2b_example_traces.csv.gz",
        "figure2/data/figure2b_examples.csv",
        "figure2/data/figure2b_focal_residue_composition.csv",
        "figure2/data/figure2c_wt_position_features.csv.gz",
        "figure2/data/figure2c_position_prevalence.csv",
        "figure2/data/figure2_position_unique_features.csv",
        "figure2/data/figure2_feature_selection_audit.csv.gz",
        "figure3/data/figure3_roc_plot_data.csv.gz",
        "figure3/data/figure3_operating_points.csv",
        "figure3/data/figure3_endpoint_bootstrap_replicates.csv.gz",
        "figure3/data/figure3_endpoint_intervals.csv",
        "figure3/data/figure3_paired_difference_intervals.csv",
        "figure3/data/figure3_roc_bootstrap_replicates.csv.gz",
        "figure3/data/figure3_roc_bands.csv.gz",
    }
)
