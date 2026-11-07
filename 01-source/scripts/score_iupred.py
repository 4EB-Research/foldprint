#!/usr/bin/env python3
"""Score validated candidate sequences with a user-installed official IUPred3 package."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path

from protein_sae_pipeline.config import load_config
from protein_sae_pipeline.io import atomic_json, sha256_file
from protein_sae_pipeline.figures.common import write_frame
from protein_sae_pipeline.scoring.iupred import load_library, official_cli, package_provenance, predict
from protein_sae_pipeline.scoring import iupred as adapter
from protein_sae_pipeline.validation import load_and_validate_inputs
import pandas as pd


LIBRARY = None


def initialize_worker(package: str) -> None:
    global LIBRARY
    LIBRARY = load_library(Path(package))


def compute_batch(records, modes, smoothing, fingerprint):
    results = []
    for reference_id, sequence_id, sequence in records:
        for mode in modes:
            result = predict(LIBRARY, sequence, mode, smoothing)
            result.update(reference_id=reference_id, sequence_id=sequence_id,
                          package_fingerprint=fingerprint)
            results.append(result)
    return results


def cache_valid(path: Path, signature: str) -> bool:
    marker = path.with_suffix(".json")
    try:
        value = json.loads(marker.read_text())
        return value["signature"] == signature and value["sha256"] == sha256_file(path)
    except (OSError, ValueError, KeyError):
        return False


def cache_batch(path: Path, signature: str, results) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt") as stream:
        for result in results:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {"signature": signature, "sha256": sha256_file(path), "rows": len(results)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iupred-package", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("long", "short"), default=["long"])
    parser.add_argument("--smoothing", choices=("no", "medium", "strong"), default="medium")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    if min(args.workers, args.batch_size) < 1:
        parser.error("--workers and --batch-size must be positive")
    config = load_config(args.config)
    inputs = load_and_validate_inputs(config, args.data_root, validate_candidate_structure_files=False)
    library = load_library(args.iupred_package)
    provenance = package_provenance(args.iupred_package)
    destination = args.output_root / "iupred"
    destination.mkdir(parents=True, exist_ok=True)
    # Retain and print the official runtime attribution instead of suppressing it.
    first = inputs.candidates[0]
    notice = official_cli(args.iupred_package, first.sequence, args.modes[0], args.smoothing)
    headers = "\n".join(line for line in notice.splitlines() if line.startswith("#")) + "\n"
    print(headers, end="")
    (destination / "official_notice.txt").write_text(headers)
    # Check the first sequence against the independently invoked official CLI
    # for every requested mode, accounting only for its four-decimal rounding.
    for mode in args.modes:
        text = notice if mode == args.modes[0] else official_cli(args.iupred_package, first.sequence, mode, args.smoothing)
        rows = [line.split() for line in text.splitlines() if line and not line.startswith("#")]
        direct = predict(library, first.sequence, mode, args.smoothing)["scores"]
        if ([int(row[0]) for row in rows] != list(range(1, len(first.sequence) + 1))
            or "".join(row[1] for row in rows) != first.sequence
            or any(abs(float(row[2]) - value) > .000050001 for row, value in zip(rows, direct))):
            raise RuntimeError("Adapter predictions differ from official IUPred3 CLI")
    profiles = destination / "residue_profiles.jsonl.gz"
    complete_marker = destination / "provenance.json"
    complete_marker.unlink(missing_ok=True)
    cache = destination / "cache"
    cache.mkdir(exist_ok=True)
    batches, pending = [], []
    for start in range(0, len(inputs.candidates), args.batch_size):
        candidates = inputs.candidates[start:start + args.batch_size]
        records = [(c.reference_id, c.sequence_id, c.sequence) for c in candidates]
        definition = {"records": records, "modes": args.modes, "smoothing": args.smoothing,
                      "fingerprint": provenance["package_fingerprint"],
                      "adapter": sha256_file(Path(adapter.__file__)), "workflow": sha256_file(Path(__file__))}
        signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        path = cache / (signature + ".jsonl.gz")
        batches.append(path)
        if not cache_valid(path, signature):
            pending.append((path, signature, records))
    print(f"IUPred3: {len(batches) - len(pending)}/{len(batches)} validated batches cached", flush=True)
    fingerprint = provenance["package_fingerprint"]
    if args.workers == 1:
        initialize_worker(str(args.iupred_package))
        for path, signature, records in pending:
            cache_batch(path, signature, compute_batch(records, args.modes, args.smoothing, fingerprint))
    else:
        # Keep only two jobs per worker in flight, limiting result memory.
        jobs = iter(pending)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(str(args.iupred_package),)) as pool:
            active = {}
            def submit_next():
                item = next(jobs, None)
                if item is None:
                    return
                path, signature, records = item
                future = pool.submit(compute_batch, records, args.modes, args.smoothing, fingerprint)
                active[future] = (path, signature)
            for _ in range(2 * args.workers):
                submit_next()
            while active:
                future = next(as_completed(active))
                path, signature = active.pop(future)
                cache_batch(path, signature, future.result())
                submit_next()
    temporary = profiles.with_name(profiles.name + ".tmp")
    summaries = {}
    try:
        with gzip.open(temporary, "wt") as stream:
            for path in batches:
                with gzip.open(path, "rt") as batch:
                    for line in batch:
                        result = json.loads(line)
                        stream.write(line)
                        key = (result["reference_id"], result["sequence_sha256"])
                        row = summaries.setdefault(key, {"reference_id": key[0], "sequence_sha256": key[1],
                            "sequence_length": result["sequence_length"], "smoothing": args.smoothing,
                            "package_fingerprint": fingerprint})
                        for name, value in result["summary"].items():
                            row[f"iupred_{result['mode']}_{name}"] = value
        temporary.replace(profiles)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if set(summaries) != {(c.reference_id, c.sequence_sha256) for c in inputs.candidates}:
        raise RuntimeError("IUPred summaries do not exactly cover validated candidates")
    metric_path = destination / "sequence_metrics.csv.gz"
    write_frame(metric_path, pd.DataFrame(summaries.values()))
    atomic_json(complete_marker, {
        "dataset_id": config.dataset_id, "rows": len(summaries), "predictor": provenance,
        "modes": args.modes, "smoothing": args.smoothing, "clipping": "none",
        "summary": "whole-sequence arithmetic mean; disordered residues use score > 0.5",
        "complete": True,
        "input_sha256": {str(args.config): sha256_file(args.config),
            config.paths["candidates"]: sha256_file(args.data_root / config.paths["candidates"])},
        "output_sha256": {metric_path.name: sha256_file(metric_path), profiles.name: sha256_file(profiles)},
    })
    print(destination / "sequence_metrics.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
