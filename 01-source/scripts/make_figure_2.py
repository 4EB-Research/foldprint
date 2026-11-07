#!/usr/bin/env python3
"""Generate Figure 2 and its plotted-data tables from the canonical master table."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_figures import main as make_figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pymol-executable", default="pymol")
    parser.add_argument("--skip-figure2-structures", action="store_true")
    parser.add_argument("--skip-full-structure-file-check", action="store_true")
    args = parser.parse_args()
    arguments = [
        "--config",
        str(args.config),
        "--data-root",
        str(args.data_root),
        "--master-table",
        str(args.output_root / "tables" / "master.csv.gz"),
        "--output-root",
        str(args.output_root),
        "--figures",
        "2",
        "--pymol-executable",
        args.pymol_executable,
    ]
    if args.prepare_only:
        arguments.append("--prepare-only")
    if args.skip_figure2_structures:
        arguments.append("--skip-figure2-structures")
    if args.skip_full_structure_file_check:
        arguments.append("--skip-full-structure-file-check")
    return make_figures(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
