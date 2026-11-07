#!/usr/bin/env python3
"""Run the complete fictional example through the analysis pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]


def _run(script: str, common: list[str], *arguments: str, environment: dict[str, str]) -> None:
    command = [sys.executable, str(REPOSITORY / "scripts" / script), *common, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--usalign-executable", default="USalign")
    parser.add_argument("--hmmsearch-executable", default="hmmsearch")
    parser.add_argument("--mmseqs-executable", default="mmseqs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid the optional live Biohub path and use committed synthetic representations.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    config = args.config.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    common = [
        "--config",
        str(config),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
    ]
    environment = os.environ.copy()
    source = str(REPOSITORY / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    if args.offline:
        environment.pop("BIOHUB_API_TOKEN", None)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["MPLCONFIGDIR"] = str(output_root / ".matplotlib-cache")

    print("The example is fictional; its numerical results are illustrative only.", flush=True)
    _run("validate_inputs.py", common, environment=environment)
    _run("score_identity.py", common, environment=environment)
    _run(
        "score_usalign.py",
        common,
        "--executable",
        args.usalign_executable,
        "--workers",
        str(args.workers),
        environment=environment,
    )
    _run(
        "score_pfam.py",
        common,
        "--executable",
        args.hmmsearch_executable,
        "--cpus",
        str(args.workers),
        environment=environment,
    )
    _run(
        "score_mmseqs.py",
        common,
        "--executable",
        args.mmseqs_executable,
        "--threads",
        str(args.workers),
        environment=environment,
    )
    _run("score_representations.py", common, environment=environment)
    _run("assemble_master.py", common, environment=environment)
    _run("build_cohorts.py", common, environment=environment)
    for script in ("make_figure_1.py", "make_figure_2.py", "make_figure_3.py"):
        _run(script, common, environment=environment)

    _run(
        "make_confidence_figures.py",
        ["--config", str(config), "--master-table", str(output_root / "tables/master.csv.gz"),
         "--output-root", str(output_root),
         "--confidence-table", str(data_root / "inputs/fictional_confidence.csv"),
         "--iupred-table", str(data_root / "inputs/fictional_iupred.csv"),
         "--endpoint-replicates", "100", "--roc-replicates", "50"],
        environment=environment,
    )

    print(f"Complete fictional example: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
