#!/usr/bin/env python3
"""Ray-trace all Figure 2 WT structure panels from the declared PDB."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_sae_pipeline.figures.pymol_ribbons import render_wt_structure_panels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt-pdb", type=Path, required=True)
    parser.add_argument("--colouring-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pymol-executable", default="pymol")
    parser.add_argument("--focal-position", type=int, default=20)
    args = parser.parse_args()
    renders = render_wt_structure_panels(
        args.wt_pdb,
        args.colouring_table,
        args.output_dir,
        focal_position=args.focal_position,
        pymol_executable=args.pymol_executable,
    )
    for output in (*renders.feature_images, renders.focal_structure_image, renders.manifest):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
