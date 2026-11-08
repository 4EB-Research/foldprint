"""Figure 2: residue-level SAE interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..interpretability import (
    FeatureRanking,
    PositionFeatureAudit,
    position_feature_audit,
    position_recurrence,
    position_unique_empirical_p,
    position_unique_features,
    positional_activations,
    rank_wt_features,
)
from ..io import safe_relative_path
from ..keys import CandidateKey
from ..representations.sparse import SparseResidueMatrix, SparseVector
from ..representations.store import IndexedRepresentations
from ..schemas import ValidatedInputs
from ..structures.pdb import ResidueID, parse_pdb
from .common import (
    ANALOGUE_COLOR,
    FOCAL_COLOR,
    INK,
    MUTED,
    NEGATIVE_COLOR,
    configure_style,
    require_columns,
    require_unique_keys,
    save_figure,
    write_panel_bundle,
)


FIGURE_EXAMPLE_COLUMNS = {
    "panel_role",
    "reference_id",
    "sequence_sha256",
    "required_structural_class",
    "required_identity_regime",
}
@dataclass(frozen=True)
class Figure2Data:
    panel_a_traces: pd.DataFrame
    panel_a_structure: pd.DataFrame
    panel_b_traces: pd.DataFrame
    panel_b_examples: pd.DataFrame
    panel_b_residue_composition: pd.DataFrame
    panel_c_traces: pd.DataFrame
    panel_c_prevalence: pd.DataFrame
    position_unique_features: pd.DataFrame
    panel_d_position_unique: pd.DataFrame
    feature_selection_audit: pd.DataFrame
    reference_id: str
    focal_position: int
    sequence_length: int
    panel_b_window: tuple[int, int]
    panel_a_feature_ids: tuple[int, ...]
    panel_b_feature_id: int
    panel_c_feature_ids: tuple[int, ...]
    structural_panels_included: bool


def _project_ca_trace(coordinates: np.ndarray) -> np.ndarray:
    """Deterministic 2D PCA projection for a documented PDB-derived schematic."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("WT schematic requires at least two finite CA coordinates")
    if not np.isfinite(values).all():
        raise ValueError("WT CA coordinates contain NaN or infinity")
    centered = values - values.mean(axis=0)
    _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
    basis = right[:2].T
    if basis.shape[1] < 2:
        basis = np.column_stack((basis, np.zeros(3)))
    projected = centered @ basis
    for column in range(2):
        anchor = int(np.argmax(np.abs(projected[:, column])))
        if projected[anchor, column] < 0:
            projected[:, column] *= -1
    return projected


def _wt_structure_table(
    inputs: ValidatedInputs,
    reference_id: str,
    feature_ids: np.ndarray,
    wt_selected: np.ndarray,
    focal_position: int,
) -> pd.DataFrame:
    reference = inputs.references[reference_id]
    structure = parse_pdb(reference.structure_path, chain_id=reference.chain_id)
    positions = inputs.reference_positions[reference_id]
    observed = [item for item in positions if item.observed]
    residue_map = {
        residue.author_id: residue for residue in structure.residues
    }
    records: list[dict[str, object]] = []
    coordinates: list[np.ndarray] = []
    mapped_positions = []
    for item in observed:
        author_id = ResidueID(
            str(item.chain_id), int(item.author_residue_number), item.insertion_code
        )
        residue = residue_map.get(author_id)
        if residue is None or residue.one_letter != item.residue:
            raise ValueError(
                f"WT PDB/position map mismatch at {reference_id} position "
                f"{item.sequence_position}"
            )
        coordinates.append(residue.coordinate("CA"))
        mapped_positions.append((item, author_id))
    projected = _project_ca_trace(np.asarray(coordinates))
    for feature_column, feature_id in enumerate(feature_ids):
        for row_index, (item, author_id) in enumerate(mapped_positions):
            records.append(
                {
                    "feature_rank": feature_column + 1,
                    "feature_id": int(feature_id),
                    "sequence_position": int(item.sequence_position),
                    "wt_residue": str(item.residue),
                    "author_chain": author_id.chain_id,
                    "author_residue_number": author_id.residue_number,
                    "author_insertion_code": author_id.insertion_code,
                    "projected_x": float(projected[row_index, 0]),
                    "projected_y": float(projected[row_index, 1]),
                    "normalized_activation": float(
                        wt_selected[item.sequence_position - 1, feature_column]
                    ),
                    "pymol_color_value_0_1": float(
                        np.clip(
                            wt_selected[item.sequence_position - 1, feature_column],
                            0.0,
                            1.0,
                        )
                    ),
                    "pymol_selection": (
                        f"chain {author_id.chain_id} and resi "
                        f"{author_id.residue_number}{author_id.insertion_code}"
                    ),
                    "is_focal_position": item.sequence_position == focal_position,
                }
            )
    return pd.DataFrame(records)


def _position_unique_table(
    residues: SparseResidueMatrix,
    *,
    focal_position: int,
    activation_threshold: float,
) -> pd.DataFrame:
    result = position_unique_features(
        residues,
        activation_threshold=activation_threshold,
    )
    records: list[dict[str, object]] = []
    for position, count, feature_ids in zip(
        result.positions,
        result.counts,
        result.feature_ids_by_position,
        strict=True,
    ):
        records.append(
            {
                "sequence_position": int(position),
                "position_unique_feature_count": int(count),
                "position_unique_feature_ids": ";".join(
                    str(value) for value in sorted(int(item) for item in feature_ids)
                ),
                "count_descending_rank": int(
                    1 + np.count_nonzero(result.counts > count)
                ),
                "empirical_greater_equal_tail_p": position_unique_empirical_p(
                    result,
                    position_1based=int(position),
                ),
                "is_focal_position": int(position) == focal_position,
                "activation_threshold": result.activation_threshold,
                "total_wt_position_unique_features": result.total_unique_features,
            }
        )
    return pd.DataFrame(records)


def _feature_selection_table(
    ranking: FeatureRanking,
    audit: PositionFeatureAudit,
    panel_a_features: np.ndarray,
    panel_b_feature: int,
    panel_c_features: np.ndarray,
) -> pd.DataFrame:
    panel_a = {int(value) for value in panel_a_features}
    panel_c = {int(value) for value in panel_c_features}
    records: list[dict[str, object]] = []
    for pooled_rank, (feature_id, pooled_activation) in enumerate(
        zip(ranking.feature_ids, ranking.scores, strict=True),
        start=1,
    ):
        feature = int(feature_id)
        index = int(np.flatnonzero(audit.feature_ids == feature)[0])
        selected_a = feature in panel_a
        selected_b = feature == panel_b_feature
        selected_c = feature in panel_c
        records.append(
            {
                "feature_id": feature,
                "wt_pooled_rank": pooled_rank,
                "wt_pooled_normalized_activation": float(pooled_activation),
                "previous_position_activation": float(audit.previous_activation[index]),
                "focal_position_activation": float(audit.current_activation[index]),
                "next_position_activation": float(audit.next_activation[index]),
                "focal_activation_rank": int(audit.activation_rank[index]),
                "gain_from_previous": float(audit.gain_from_previous[index]),
                "gain_rank": int(audit.gain_rank[index]),
                "local_mean_contrast": float(audit.local_mean_contrast[index]),
                "local_mean_rank": int(audit.local_mean_rank[index]),
                "local_max_contrast": float(audit.local_max_contrast[index]),
                "local_max_rank": int(audit.local_max_rank[index]),
                "selected_panel_a": selected_a,
                "selected_panel_b": selected_b,
                "selected_panel_c": selected_c,
                "selected_any_panel": selected_a or selected_b or selected_c,
            }
        )
    return pd.DataFrame(records)


def _load_examples(
    master: pd.DataFrame,
    inputs: ValidatedInputs,
    config: PipelineConfig,
    reference_id: str,
) -> pd.DataFrame:
    path = safe_relative_path(
        inputs.data_root,
        config.paths["figure_examples"],
        label="figure examples path",
    )
    examples = pd.read_csv(path, dtype={"sequence_sha256": str})
    missing = FIGURE_EXAMPLE_COLUMNS - set(examples.columns)
    extra = set(examples.columns) - FIGURE_EXAMPLE_COLUMNS
    if missing or extra:
        raise ValueError(
            f"figure example fields differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    examples = examples.loc[examples["reference_id"].astype(str).eq(reference_id)].copy()
    if examples.empty or examples["panel_role"].duplicated().any():
        raise ValueError("Figure 2 requires unique configured examples for the focal reference")
    joined = examples.merge(
        master,
        on=["reference_id", "sequence_sha256"],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_master"),
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("A configured Figure 2 example is absent from the master table")
    for row in joined.itertuples(index=False):
        required_class = str(row.required_structural_class)
        if required_class not in {"analogue", "negative"}:
            raise ValueError("Figure example structural class must be analogue or negative")
        if str(row.structural_class) != required_class:
            raise ValueError(f"Figure example {row.panel_role} has the wrong structural class")
        regime = str(row.required_identity_regime)
        is_remote = bool(row.is_remote)
        if regime == "remote" and not is_remote:
            raise ValueError(f"Figure example {row.panel_role} is not remote")
        if regime == "nonremote" and is_remote:
            raise ValueError(f"Figure example {row.panel_role} is unexpectedly remote")
        if regime not in {"remote", "nonremote", "any"}:
            raise ValueError("Figure example identity regime must be remote, nonremote, or any")
    return joined.drop(columns=["_merge"])


def _row_sparse_vector(
    store: IndexedRepresentations,
    key: CandidateKey | None,
    reference_id: str,
    position_1based: int,
) -> SparseVector:
    residues = (
        store.wt_residues(reference_id)
        if key is None
        else store.candidate_residues(key)
    )
    position = position_1based - 1
    if not 0 <= position < residues.sequence_length:
        raise ValueError("focal position lies outside a residue profile")
    return SparseVector(
        np.asarray(residues.indices[position]),
        np.asarray(residues.values[position]),
        residues.size,
    )


def prepare_figure2(
    master: pd.DataFrame,
    store: IndexedRepresentations,
    inputs: ValidatedInputs,
    config: PipelineConfig,
    *,
    include_structural_panels: bool = True,
) -> Figure2Data:
    required = {
        "reference_id",
        "sequence_sha256",
        "manifest_row",
        "generation_method",
        "positional_identity_fraction",
        "usalign_tm_score_wt_norm",
        "sae_cosine_to_wt",
        "structural_class",
        "is_remote",
    }
    require_columns(master, required, label="master table")
    require_unique_keys(master, label="master table")
    reference_id = str(config.raw["interpretability"]["reference_id"])
    focal_position = int(config.raw["interpretability"]["focal_position"])
    reference = inputs.references[reference_id]
    if reference.figure_focal_position != focal_position:
        raise ValueError("Reference manifest and config disagree on the focal position")
    target = master.loc[master["reference_id"].astype(str).eq(reference_id)].copy()
    target = target.sort_values("manifest_row", kind="stable").reset_index(drop=True)
    remote = target.loc[target["is_remote"].astype(bool)].copy()
    expected_remote = int(
        config.expected_counts["by_reference"][reference_id]["remote_total"]
    )
    if len(remote) != expected_remote:
        raise ValueError(
            f"Figure 2 remote cohort has {len(remote)} rows; expected {expected_remote}"
        )
    profile_cohort = remote
    candidate_by_hash = {
        candidate.sequence_sha256: candidate
        for candidate in inputs.candidates
        if candidate.reference_id == reference_id
    }

    wt_profile = store.wt_profile(reference_id)
    wt_residues = store.wt_residues(reference_id)
    ranking = rank_wt_features(wt_profile)
    all_feature_ids = ranking.feature_ids.astype(np.int64)
    if len(all_feature_ids) < 5:
        raise ValueError("Figure 2 requires at least five active WT features")
    wt_selected_all = positional_activations(wt_residues, all_feature_ids)

    panel_a_features = all_feature_ids[:5]
    panel_a_values = wt_selected_all[:, :5]
    panel_a_records: list[dict[str, object]] = []
    for column, feature_id in enumerate(panel_a_features):
        for position, activation in enumerate(panel_a_values[:, column], start=1):
            panel_a_records.append(
                {
                    "feature_rank": column + 1,
                    "feature_id": int(feature_id),
                    "sequence_position": position,
                    "wt_residue": reference.sequence[position - 1],
                    "normalized_activation": float(activation),
                    "is_focal_position": position == focal_position,
                }
            )
    panel_a_traces = pd.DataFrame(panel_a_records)
    panel_a_structure = (
        _wt_structure_table(
            inputs,
            reference_id,
            panel_a_features,
            panel_a_values,
            focal_position,
        )
        if include_structural_panels
        else pd.DataFrame()
    )

    examples = _load_examples(master, inputs, config, reference_id)
    panel_b_feature = int(panel_a_features[0])
    panel_b_records: list[dict[str, object]] = []
    wt_trace = positional_activations(
        wt_residues, np.asarray([panel_b_feature], dtype=np.int64)
    )[:, 0]
    for position, activation in enumerate(wt_trace, start=1):
        panel_b_records.append(
            {
                "example_order": 0,
                "panel_role": "WT",
                "reference_id": reference_id,
                "sequence_sha256": reference.sequence_sha256,
                "structural_class": "WT",
                "positional_identity_fraction": 1.0,
                "usalign_tm_score_wt_norm": 1.0,
                "feature_id": panel_b_feature,
                "sequence_position": position,
                "normalized_activation": float(activation),
                "is_focal_position": position == focal_position,
                "residue": reference.sequence[position - 1],
                "wt_residue": reference.sequence[position - 1],
                "is_substitution": False,
            }
        )
    example_summary: list[dict[str, object]] = []
    for order, row in enumerate(examples.itertuples(index=False), start=1):
        key = CandidateKey(reference_id, str(row.sequence_sha256))
        candidate = candidate_by_hash.get(str(row.sequence_sha256))
        if candidate is None:
            raise ValueError("Figure 2B example is absent from validated candidates")
        trace = positional_activations(
            store.candidate_residues(key),
            np.asarray([panel_b_feature], dtype=np.int64),
        )[:, 0]
        for position, activation in enumerate(trace, start=1):
            panel_b_records.append(
                {
                    "example_order": order,
                    "panel_role": str(row.panel_role),
                    "reference_id": reference_id,
                    "sequence_sha256": str(row.sequence_sha256),
                    "structural_class": str(row.structural_class),
                    "positional_identity_fraction": float(
                        row.positional_identity_fraction
                    ),
                    "usalign_tm_score_wt_norm": float(row.usalign_tm_score_wt_norm),
                    "feature_id": panel_b_feature,
                    "sequence_position": position,
                    "normalized_activation": float(activation),
                    "is_focal_position": position == focal_position,
                    "residue": candidate.sequence[position - 1],
                    "wt_residue": reference.sequence[position - 1],
                    "is_substitution": (
                        candidate.sequence[position - 1] != reference.sequence[position - 1]
                    ),
                }
            )
        window_start = max(1, focal_position - 6)
        window_end = min(reference.sequence_length, focal_position + 6)
        window_matches = sum(
            candidate.sequence[index] == reference.sequence[index]
            for index in range(window_start - 1, window_end)
        )
        example_summary.append(
            {
                "example_order": order,
                "panel_role": str(row.panel_role),
                "reference_id": reference_id,
                "sequence_sha256": str(row.sequence_sha256),
                "structural_class": str(row.structural_class),
                "identity_regime": str(row.required_identity_regime),
                "positional_identity_fraction": float(row.positional_identity_fraction),
                "usalign_tm_score_wt_norm": float(row.usalign_tm_score_wt_norm),
                "local_identity_fraction": window_matches / (window_end - window_start + 1),
            }
        )
    panel_b_traces = pd.DataFrame(panel_b_records)
    panel_b_examples = pd.DataFrame(example_summary)
    composition_records: list[dict[str, object]] = []
    for structural_class in ("analogue", "negative"):
        class_keys = [
            str(value)
            for value in remote.loc[
                remote["structural_class"].eq(structural_class),
                "sequence_sha256",
            ]
        ]
        if not class_keys:
            raise ValueError(f"Figure 2B lacks structural class {structural_class}")
        residue_counts: dict[str, int] = {}
        for sequence_hash in class_keys:
            candidate = candidate_by_hash.get(sequence_hash)
            if candidate is None:
                raise ValueError("Figure 2B candidate is absent from validated inputs")
            residue = candidate.sequence[focal_position - 1]
            residue_counts[residue] = residue_counts.get(residue, 0) + 1
        for residue, count in sorted(
            residue_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            composition_records.append(
                {
                    "structural_class": structural_class,
                    "residue": residue,
                    "count": count,
                    "fraction": count / len(class_keys),
                    "cohort_count": len(class_keys),
                }
            )
    panel_b_composition = pd.DataFrame(composition_records)

    audit = position_feature_audit(
        wt_residues,
        all_feature_ids,
        position_1based=focal_position,
    )
    panel_c_columns = np.argsort(audit.activation_rank)[:4]
    panel_c_features = all_feature_ids[panel_c_columns]
    panel_c_records: list[dict[str, object]] = []
    for display_rank, column in enumerate(panel_c_columns, start=1):
        for position, activation in enumerate(
            wt_selected_all[:, column], start=1
        ):
            panel_c_records.append(
                {
                    "position_activation_rank": display_rank,
                    "feature_id": int(all_feature_ids[column]),
                    "sequence_position": position,
                    "wt_residue": reference.sequence[position - 1],
                    "normalized_activation": float(activation),
                    "is_focal_position": position == focal_position,
                }
            )
    panel_c_traces = pd.DataFrame(panel_c_records)
    prevalence_records: list[dict[str, object]] = []
    for structural_class in ("analogue", "negative"):
        class_rows = profile_cohort.loc[
            profile_cohort["structural_class"].eq(structural_class)
        ]
        residue_profiles = [
            store.candidate_residues(
                CandidateKey(reference_id, str(value))
            )
            for value in class_rows["sequence_sha256"]
        ]
        prevalence = position_recurrence(
            residue_profiles,
            panel_c_features,
            position_1based=focal_position,
        )
        for display_rank, feature_id, count, fraction, low, high in zip(
            range(1, len(panel_c_features) + 1),
            prevalence.feature_ids,
            prevalence.active_counts,
            prevalence.recurrence,
            prevalence.wilson_low,
            prevalence.wilson_high,
            strict=True,
        ):
            prevalence_records.append(
                {
                    "position_activation_rank": display_rank,
                    "feature_id": int(feature_id),
                    "structural_class": structural_class,
                    "active_candidate_count": int(count),
                    "candidate_count": prevalence.total,
                    "prevalence": float(fraction),
                    "wilson_low": float(low),
                    "wilson_high": float(high),
                    "cohort_scope": "focal_reference_remote",
                }
            )
    panel_c_prevalence = pd.DataFrame(prevalence_records)

    activation_threshold = float(
        config.raw["interpretability"]["feature_activation_threshold"]
    )
    unique_positions = _position_unique_table(
        wt_residues,
        focal_position=focal_position,
        activation_threshold=activation_threshold,
    )
    panel_d_position_unique = unique_positions.copy()
    selection_audit = _feature_selection_table(
        ranking,
        audit,
        panel_a_features,
        panel_b_feature,
        panel_c_features,
    )

    return Figure2Data(
        panel_a_traces=panel_a_traces,
        panel_a_structure=panel_a_structure,
        panel_b_traces=panel_b_traces,
        panel_b_examples=panel_b_examples,
        panel_b_residue_composition=panel_b_composition,
        panel_c_traces=panel_c_traces,
        panel_c_prevalence=panel_c_prevalence,
        position_unique_features=unique_positions,
        panel_d_position_unique=panel_d_position_unique,
        feature_selection_audit=selection_audit,
        reference_id=reference_id,
        focal_position=focal_position,
        sequence_length=reference.sequence_length,
        panel_b_window=(
            max(1, focal_position - 6),
            min(reference.sequence_length, focal_position + 6),
        ),
        panel_a_feature_ids=tuple(int(value) for value in panel_a_features),
        panel_b_feature_id=panel_b_feature,
        panel_c_feature_ids=tuple(int(value) for value in panel_c_features),
        structural_panels_included=include_structural_panels,
    )


def export_figure2(
    data: Figure2Data,
    output_root: Path,
    config: PipelineConfig,
    *,
    pymol_available: bool = True,
) -> None:
    tables = {
        "figure2a_wt_feature_traces.csv.gz": data.panel_a_traces,
        "figure2b_example_traces.csv.gz": data.panel_b_traces,
        "figure2b_examples.csv": data.panel_b_examples,
        "figure2b_focal_residue_composition.csv": data.panel_b_residue_composition,
        "figure2c_wt_position_features.csv.gz": data.panel_c_traces,
        "figure2c_position_prevalence.csv": data.panel_c_prevalence,
        "figure2_position_unique_features.csv": data.position_unique_features,
        "figure2d_position_unique_plot_data.csv": data.panel_d_position_unique,
        "figure2_feature_selection_audit.csv.gz": data.feature_selection_audit,
    }
    if data.structural_panels_included:
        tables["figure2a_wt_pdb_residue_colouring.csv.gz"] = data.panel_a_structure
    write_panel_bundle(
        output_root / "figure2" / "data",
        tables,
        metadata={
            "schema_version": 1,
            "dataset_id": config.dataset_id,
            "reference_id": data.reference_id,
            "focal_position_1based": data.focal_position,
            "panel_a_feature_ids": list(data.panel_a_feature_ids),
            "panel_b_feature_id": data.panel_b_feature_id,
            "panel_c_feature_ids": list(data.panel_c_feature_ids),
            "structural_panels_included": data.structural_panels_included,
            "wt_structural_inset": (
                {
                    "source": "configured experimental WT PDB",
                    "representation": (
                        "same-run code-generated PyMOL ray-traced cartoons"
                        if pymol_available
                        else "blank when PyMOL renders are unavailable"
                    ),
                    "activation_mapping": "normalized residue activation on line segments",
                    "activation_scale": [0.0, 1.0],
                    "values_above_one": (
                        "retained in plotted-data tables; saturated in rendering"
                    ),
                    "camera_claim": (
                        "fixed code-defined PyMOL camera"
                        if pymol_available
                        else "not applicable; structural image left blank"
                    ),
                    "pymol_handoff": (
                        "figure2a_wt_pdb_residue_colouring.csv.gz contains feature ID, "
                        "author residue mapping, PyMOL selection, and clipped colour value"
                    ),
                }
                if data.structural_panels_included
                else {"status": "omitted by structure-free rendering mode"}
            ),
            "feature_selection": {
                "selection_inputs": (
                    "WT pooled activation and focal/previous/next numerical activations"
                ),
                "panel_a": "five highest WT max-pooled activations",
                "panel_b": "highest WT max-pooled activation",
                "panel_c": "four highest focal-position WT activations",
            },
            "main_figure_panels": {
                "A": (
                    "WT full-length feature traces and PyMOL feature maps"
                    if data.structural_panels_included and pymol_available
                    else "WT full-length feature traces; structural image slots blank"
                    if data.structural_panels_included
                    else "WT full-length feature traces; molecular cartoons omitted"
                ),
                "B": (
                    f"feature {data.panel_b_feature_id} residues "
                    f"{data.panel_b_window[0]}-{data.panel_b_window[1]} in configured examples"
                ),
                "C": (
                    "focal WT features and prevalence recomputed from the complete "
                    "focal-reference remote cohort"
                ),
                "D": "WT position-unique active-feature counts",
                "E": (
                    f"focal protein residue-{data.focal_position} structural close-up"
                    if data.structural_panels_included and pymol_available
                    else "structural image blank because PyMOL is unavailable"
                    if data.structural_panels_included
                    else "omitted by structure-free rendering mode"
                ),
            },
            "position_unique_features": {
                "definition": "active at one WT residue and no other WT residue",
                "activity": (
                    "normalized residue activation strictly greater than configured threshold"
                ),
                "rank": "1 + number of WT positions with a strictly larger count",
                "empirical_p": (
                    "greater-equal tail over all other WT positions with plus-one correction"
                ),
            },
        },
        metadata_name="figure2_metadata.json",
    )


def _draw_ca_schematic(axis: plt.Axes, frame: pd.DataFrame) -> None:
    frame = frame.sort_values("sequence_position")
    points = frame[["projected_x", "projected_y"]].to_numpy(float)
    positions = frame["sequence_position"].to_numpy(int)
    activations = frame["normalized_activation"].to_numpy(float)
    segments = []
    segment_values = []
    for index in range(len(points) - 1):
        if positions[index + 1] != positions[index] + 1:
            continue
        segments.append([points[index], points[index + 1]])
        segment_values.append(max(activations[index], activations[index + 1]))
    collection = mcollections.LineCollection(
        segments,
        cmap=mcolors.LinearSegmentedColormap.from_list(
            "neutral_to_activation", ["#F5EDED", "#D6AAAA", "#A85C5C"]
        ),
        norm=plt.Normalize(0, 1.0),
        linewidths=2.8,
    )
    collection.set_array(np.asarray(segment_values))
    axis.add_collection(collection)
    focal = frame.loc[frame["is_focal_position"].astype(bool)]
    if not focal.empty:
        axis.scatter(
            focal["projected_x"], focal["projected_y"], s=11, color=FOCAL_COLOR, zorder=3
        )
    x_span = max(float(np.ptp(points[:, 0])), 1.0)
    y_span = max(float(np.ptp(points[:, 1])), 1.0)
    axis.set_xlim(
        float(points[:, 0].min()) - 0.06 * x_span,
        float(points[:, 0].max()) + 0.06 * x_span,
    )
    axis.set_ylim(
        float(points[:, 1].min()) - 0.06 * y_span,
        float(points[:, 1].max()) + 0.06 * y_span,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def _draw_panel_a(
    container: plt.Figure | plt.SubFigure,
    data: Figure2Data,
) -> None:
    grid = container.add_gridspec(
        5,
        2,
        width_ratios=[3.2, 1.0],
        hspace=0.40,
        wspace=0.08,
        top=0.84,
        bottom=0.06,
    )
    for row, feature_id in enumerate(data.panel_a_feature_ids):
        trace = data.panel_a_traces.loc[data.panel_a_traces["feature_id"].eq(feature_id)]
        rendered_activation = np.clip(
            trace["normalized_activation"].to_numpy(float), 0.0, 1.0
        )
        axis = container.add_subplot(grid[row, 0])
        axis.plot(
            trace["sequence_position"], rendered_activation, color=INK, linewidth=1.0
        )
        axis.fill_between(
            trace["sequence_position"], 0, rendered_activation, color=INK, alpha=1.0
        )
        axis.axvline(data.focal_position, color=FOCAL_COLOR, linewidth=0.7, alpha=0.7)
        axis.set_xlim(1, data.sequence_length)
        axis.set_ylim(0, 1.0)
        axis.set_yticks([0, 1.0])
        axis.set_title(f"Feature {feature_id}", loc="left", fontsize=7.0, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        if row < 4:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Residue position")
        schematic = container.add_subplot(grid[row, 1])
        _draw_ca_schematic(
            schematic,
            data.panel_a_structure.loc[
                data.panel_a_structure["feature_id"].eq(feature_id)
            ],
        )
    container.suptitle(
        f"A  WT sparse composition switches at profile position {data.focal_position}",
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontweight="bold",
    )


def _draw_panel_b(axis: plt.Axes, data: Figure2Data) -> None:
    style = {"WT": INK, "analogue": ANALOGUE_COLOR, "negative": NEGATIVE_COLOR}
    role_labels = {
        "WT": "WT focal protein",
        "remote_analogue": "Remote analogue",
        "remote_negative": "Remote negative",
    }
    for (_order, role), group in data.panel_b_traces.groupby(
        ["example_order", "panel_role"], sort=True, observed=True
    ):
        structural_class = str(group.iloc[0]["structural_class"])
        axis.plot(
            group["sequence_position"],
            group["normalized_activation"],
            linewidth=1.1,
            color=style.get(structural_class, MUTED),
            label=role_labels.get(str(role), str(role).replace("_", " ").title()),
        )
    axis.axvline(data.focal_position, color=FOCAL_COLOR, linewidth=0.8)
    axis.set(xlabel="Residue position", ylabel="Normalized activation")
    axis.set_xlim(1, data.sequence_length)
    axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=6.5, loc="upper left")
    axis.set_title(
        f"B  Feature {data.panel_b_feature_id} follows structural state\n"
        "rather than local sequence identity",
        loc="left",
        fontweight="bold",
    )


def _draw_panel_c(container: plt.Figure | plt.SubFigure, data: Figure2Data) -> None:
    grid = container.add_gridspec(
        4,
        2,
        width_ratios=[3.0, 1.2],
        hspace=0.40,
        wspace=0.18,
        top=0.84,
        bottom=0.08,
    )
    common_y_max = max(
        1.0, float(data.panel_c_traces["normalized_activation"].max())
    )
    for row, feature_id in enumerate(data.panel_c_feature_ids):
        trace = data.panel_c_traces.loc[data.panel_c_traces["feature_id"].eq(feature_id)]
        axis = container.add_subplot(grid[row, 0])
        axis.plot(
            trace["sequence_position"],
            trace["normalized_activation"],
            color=INK,
            linewidth=1.0,
        )
        focal = trace.loc[trace["is_focal_position"].astype(bool)]
        axis.scatter(
            focal["sequence_position"], focal["normalized_activation"], color=FOCAL_COLOR, s=10
        )
        axis.set_xlim(1, data.sequence_length)
        axis.set_ylim(0, common_y_max)
        axis.set_yticks([0, common_y_max])
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(f"Feature {feature_id}", loc="left", fontsize=7.0, fontweight="bold")
        if row < 3:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Residue position")
        prevalence = data.panel_c_prevalence.loc[
            data.panel_c_prevalence["feature_id"].eq(feature_id)
        ].set_index("structural_class")
        bars = container.add_subplot(grid[row, 1])
        bars.bar(
            [0, 1],
            [prevalence.loc["analogue", "prevalence"], prevalence.loc["negative", "prevalence"]],
            color=[ANALOGUE_COLOR, NEGATIVE_COLOR],
            width=0.62,
        )
        bars.set_ylim(0, 1)
        bars.spines[["top", "right"]].set_visible(False)
        if row < 3:
            bars.set_xticks([])
        else:
            bars.set_xticks([0, 1], ["Analogue", "Negative"], rotation=25, ha="right")
    container.suptitle(
        f"C  Top four features by absolute WT activation at residue {data.focal_position}",
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontweight="bold",
    )


def render_figure2(data: Figure2Data, output_root: Path) -> None:
    from .figure2_layout import render_figure2_layout

    render_figure2_layout(data, output_root)
