#!/usr/bin/env python3
"""Prepare plotted-data tables and render Figures 1--3 from canonical inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil

import pandas as pd

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.master import validate_master_against_inputs
from protein_sae_pipeline.figures.figure1 import (
    export_figure1,
    prepare_figure1,
    render_figure1,
)
from protein_sae_pipeline.figures.figure2 import (
    export_figure2,
    prepare_figure2,
    render_figure2,
)
from protein_sae_pipeline.figures.figure3 import (
    export_figure3,
    prepare_figure3,
    render_figure3,
)
from protein_sae_pipeline.figures.pymol_ribbons import render_wt_structure_panels
from protein_sae_pipeline.representations.store import load_indexed_representations
from protein_sae_pipeline.validation import load_and_validate_inputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate figures and plotted-data tables from a validated "
            "master score table and sparse representation store."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--master-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("1", "2", "3"),
        default=("1", "2", "3"),
        help="Figure numbers to generate (default: all).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write plotted-data tables and metadata without rendering images.",
    )
    parser.add_argument(
        "--pymol-executable",
        default="pymol",
        help="PyMOL executable used to ray-trace manuscript Figure 2 structures.",
    )
    parser.add_argument(
        "--skip-full-structure-file-check",
        action="store_true",
        help=(
            "Validate the complete candidate-structure manifest without stat-ing every "
            "candidate PDB. Any candidate PDB used by Figure 2 is still opened."
        ),
    )
    parser.add_argument(
        "--skip-figure2-structures",
        action="store_true",
        help=(
            "Render Figure 2A traces and panels 2B-2D without molecular cartoons, "
            "panel 2E, candidate PDB validation, or PyMOL."
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    inputs = load_and_validate_inputs(
        config,
        args.data_root,
        validate_candidate_structure_files=not (
            args.skip_full_structure_file_check or args.skip_figure2_structures
        ),
    )
    master = pd.read_csv(args.master_table, low_memory=False)
    validate_master_against_inputs(master, inputs, config)
    selected = set(args.figures)

    store = None
    if selected & {"1", "2"}:
        chunk_path = args.data_root / config.paths["representations"]
        index_path = chunk_path.with_name("representation_index.csv")
        store = load_indexed_representations(
            chunk_path,
            index_path,
            inputs,
            config=config,
        )

    if "1" in selected:
        assert store is not None
        print("Preparing Figure 1 tables...", flush=True)
        figure1 = prepare_figure1(master, store, config)
        export_figure1(figure1, args.output_root, config)
        if not args.prepare_only:
            print("Rendering Figure 1...", flush=True)
            render_figure1(figure1, args.output_root)
        print(args.output_root / "figure1")

    if "2" in selected:
        assert store is not None
        stage = (
            "Preparing numerical Figure 2 panels without structures..."
            if args.skip_figure2_structures
            else "Preparing Figure 2 panels..."
        )
        print(stage, flush=True)
        figure2 = prepare_figure2(
            master,
            store,
            inputs,
            config,
            include_structural_panels=not args.skip_figure2_structures,
        )
        pymol_available = shutil.which(args.pymol_executable) is not None
        export_figure2(
            figure2,
            args.output_root,
            config,
            pymol_available=not args.skip_figure2_structures
            and pymol_available,
        )
        if not args.prepare_only:
            if not args.skip_figure2_structures and pymol_available:
                print("Ray-tracing Figure 2 structures with PyMOL...", flush=True)
                reference = inputs.references[figure2.reference_id]
                geometry = (
                    {"packing_positions": (5, 8, 33)} if config.is_example else {}
                )
                render_wt_structure_panels(
                    reference.structure_path,
                    figure2.panel_a_structure,
                    args.output_root / "figure2" / "structure_renders",
                    focal_position=figure2.focal_position,
                    pymol_executable=args.pymol_executable,
                    **geometry,
                )
            elif not args.skip_figure2_structures:
                print(
                    "PyMOL was not found; Figure 2 structural panels will be left blank.",
                    flush=True,
                )
            print("Rendering Figure 2...", flush=True)
            render_figure2(figure2, args.output_root)
        print(args.output_root / "figure2")

    if "3" in selected:
        print("Preparing Figure 3 bootstrap tables...", flush=True)
        figure3 = prepare_figure3(master, config)
        export_figure3(figure3, args.output_root, config)
        if not args.prepare_only:
            print("Rendering Figure 3...", flush=True)
            render_figure3(figure3, args.output_root, config)
        print(args.output_root / "figure3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
