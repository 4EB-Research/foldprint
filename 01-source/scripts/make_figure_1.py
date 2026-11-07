#!/usr/bin/env python3
"""Generate Figure 1 and its plotted-data tables from the canonical master table."""

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
        "1",
    ]
    if args.prepare_only:
        arguments.append("--prepare-only")
    return make_figures(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
