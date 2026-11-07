"""Strict scientific configuration loader.

Paths and process counts may be supplied by scripts. Scientific choices live
only in the validated YAML file so a run cannot silently change the analysis
endpoint from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "example",
        "generation_methods",
        "references",
        "paths",
        "thresholds",
        "models",
        "native_tools",
        "bootstrap",
        "interpretability",
        "expected_counts",
    }
)

ESMC_FIELDS = {
    "api_alias",
    "weights_repository",
    "weights_revision",
    "sdk_version",
    "sdk_commit",
    "transformers_version",
    "transformers_commit",
    "hidden_state_index",
    "embedding_dimension",
    "special_tokens",
    "pooling",
    "cosine_accumulation",
}
SAE_FIELDS = {
    "api_alias",
    "repository",
    "revision",
    "hidden_state_index",
    "codebook_size",
    "top_k",
    "request_normalize_features",
    "normalization",
    "pooling",
    "cosine_accumulation",
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _require_exact(mapping: Mapping[str, Any], *, required: set[str], label: str) -> None:
    missing = required - set(mapping)
    extra = set(mapping) - required
    if missing or extra:
        raise ValueError(f"{label} fields differ: missing={sorted(missing)}, extra={sorted(extra)}")


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.raw["dataset_id"])

    @property
    def is_example(self) -> bool:
        return bool(self.raw["example"])

    @property
    def generation_methods(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["generation_methods"])

    @property
    def identity_threshold(self) -> float:
        return float(self.raw["thresholds"]["remote_identity_fraction"])

    @property
    def tm_threshold(self) -> float:
        return float(self.raw["thresholds"]["structural_analogue_tm"])

    @property
    def paths(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.raw["paths"].items()}

    @property
    def bootstrap(self) -> dict[str, Any]:
        return dict(self.raw["bootstrap"])

    @property
    def expected_counts(self) -> dict[str, Any]:
        return dict(self.raw["expected_counts"])

    def model(self, name: str) -> dict[str, Any]:
        return dict(self.raw["models"][name])

    def native_tool(self, name: str) -> dict[str, Any]:
        return dict(self.raw["native_tools"][name])


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = _mapping(payload, "configuration")
    _require_exact(raw, required=set(TOP_LEVEL_FIELDS), label="configuration")
    if int(raw["schema_version"]) != 1:
        raise ValueError("Only configuration schema_version 1 is supported")
    if not isinstance(raw["dataset_id"], str) or not raw["dataset_id"].strip():
        raise ValueError("dataset_id must be a nonempty string")
    methods = raw["generation_methods"]
    if not isinstance(methods, list) or len(methods) < 1 or len(set(methods)) != len(methods):
        raise ValueError("generation_methods must be a nonempty unique list")

    thresholds = _mapping(raw["thresholds"], "thresholds")
    _require_exact(
        thresholds,
        required={"remote_identity_fraction", "structural_analogue_tm", "fpr_endpoints"},
        label="thresholds",
    )
    if not 0 < float(thresholds["remote_identity_fraction"]) < 1:
        raise ValueError("remote_identity_fraction must be in (0,1)")
    if not 0 < float(thresholds["structural_analogue_tm"]) <= 1:
        raise ValueError("structural_analogue_tm must be in (0,1]")
    endpoints = [float(value) for value in thresholds["fpr_endpoints"]]
    if endpoints != [0.01, 0.05]:
        raise ValueError("The frozen pipeline requires FPR endpoints [0.01, 0.05]")

    bootstrap = _mapping(raw["bootstrap"], "bootstrap")
    _require_exact(
        bootstrap,
        required={
            "seed",
            "endpoint_replicates",
            "roc_replicates",
            "unit_test_replicates",
            "interval_percentiles",
            "roc_grid_points",
            "strata",
        },
        label="bootstrap",
    )
    if tuple(bootstrap["strata"]) != (
        "reference_id",
        "structural_class",
        "generation_method",
    ):
        raise ValueError("bootstrap strata differ from the configured definition")
    endpoint_replicates = int(bootstrap["endpoint_replicates"])
    roc_replicates = int(bootstrap["roc_replicates"])
    if endpoint_replicates <= 0 or not 0 < roc_replicates <= endpoint_replicates:
        raise ValueError("bootstrap replicate counts are invalid")
    if [float(x) for x in bootstrap["interval_percentiles"]] != [2.5, 97.5]:
        raise ValueError("bootstrap interval_percentiles must be [2.5, 97.5]")
    if int(bootstrap["seed"]) < 0:
        raise ValueError("bootstrap seed must be nonnegative")
    if int(bootstrap["roc_grid_points"]) != 101:
        raise ValueError("bootstrap ROC grid differs from the configured definition")
    if int(bootstrap["unit_test_replicates"]) != 16:
        raise ValueError("unit_test_replicates must be 16")

    paths = _mapping(raw["paths"], "paths")
    _require_exact(
        paths,
        required={
            "candidates",
            "references",
            "reference_positions",
            "structures",
            "representations",
            "figure_examples",
        },
        label="paths",
    )
    references = _mapping(raw["references"], "references")
    if not references:
        raise ValueError("references must be nonempty")
    for key, value in references.items():
        item = _mapping(value, f"references.{key}")
        _require_exact(
            item,
            required={
                "display_name",
                "expected_full",
                "focal_position",
                "pfam_accession",
            },
            label=f"references.{key}",
        )
        accession = item["pfam_accession"]
        if not isinstance(accession, str) or re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]*\.[1-9][0-9]*", accession
        ) is None:
            raise ValueError(
                f"references.{key}.pfam_accession must be a versioned accession"
            )

    required_model_names = {"esmc", "sae"}
    models = _mapping(raw["models"], "models")
    if set(models) != required_model_names:
        raise ValueError(f"models must contain exactly {sorted(required_model_names)}")
    for name in required_model_names:
        if not isinstance(models[name], Mapping):
            raise ValueError(f"models.{name} must be a mapping")
    esmc = _mapping(models["esmc"], "models.esmc")
    sae = _mapping(models["sae"], "models.sae")
    _require_exact(esmc, required=ESMC_FIELDS, label="models.esmc")
    _require_exact(sae, required=SAE_FIELDS, label="models.sae")
    esmc_frozen = {
        "api_alias": "esmc-6b-2024-12",
        "weights_repository": "biohub/ESMC-6B",
        "weights_revision": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
        "sdk_version": "3.3.0",
        "sdk_commit": "67838dc8ac76f4145613e6cb36c5f3d758542f7c",
        "transformers_version": "4.57.6",
        "transformers_commit": "ef32577f55da19a4989cd7b22e004dc43a4998cb",
        "hidden_state_index": 60,
        "embedding_dimension": 2560,
        "special_tokens": ["BOS", "EOS"],
        "pooling": "fp32_arithmetic_mean_over_residues",
        "cosine_accumulation": "float64",
    }
    if any(esmc.get(key) != value for key, value in esmc_frozen.items()):
        raise ValueError("models.esmc differs from the supported model and pooling definition")
    sae_frozen = {
        "api_alias": "esmc-6b-2024-12-sae-layer60-k64-codebook16384",
        "repository": "biohub/ESMC-6B-sae-layer60-k64-codebook16384",
        "revision": "99752fe6e4d25fbb26f887db2d9225e7a577da73",
        "hidden_state_index": 60,
        "codebook_size": 16384,
        "top_k": 64,
        "request_normalize_features": False,
        "normalization": "raw_divide_maximum_then_multiply_idf_fp32",
        "pooling": "featurewise_max_over_normalized_residue_activations",
        "cosine_accumulation": "float64",
    }
    if any(sae.get(key) != value for key, value in sae_frozen.items()):
        raise ValueError("models.sae differs from the supported model and normalization")

    tools = _mapping(raw["native_tools"], "native_tools")
    if set(tools) != {"usalign", "hmmer", "mmseqs2"}:
        raise ValueError("native_tools must contain exactly usalign, hmmer and mmseqs2")
    for name, item in tools.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"native_tools.{name} must be a mapping")
    usalign = _mapping(tools["usalign"], "native_tools.usalign")
    hmmer = _mapping(tools["hmmer"], "native_tools.hmmer")
    mmseqs = _mapping(tools["mmseqs2"], "native_tools.mmseqs2")
    _require_exact(
        usalign,
        required={
            "executable", "version", "source_commit", "source_archive_sha256",
            "arguments", "structure_1", "structure_2", "primary_normalization",
        },
        label="native_tools.usalign",
    )
    _require_exact(
        hmmer,
        required={"executable", "version", "arguments", "score", "no_hit"},
        label="native_tools.hmmer",
    )
    _require_exact(
        mmseqs,
        required={
            "executable", "release", "commit", "evalue", "sensitivity",
            "max_sequences", "format",
        },
        label="native_tools.mmseqs2",
    )
    if any(
        usalign.get(key) != value
        for key, value in {
            "version": "20260819",
            "source_commit": "fa4376bd99fa17a123d05d7ea47cf6574c80d64f",
            "source_archive_sha256": "ee5a2ac8904a3d526a3888d7897a16594de0507bf4b167ec8c0c72b2202776d8",
            "arguments": ["-mol", "prot"],
            "structure_1": "candidate",
            "structure_2": "wild_type",
            "primary_normalization": "structure_2_resolved_ca_length",
        }.items()
    ):
        raise ValueError("native_tools.usalign differs from the frozen protocol")
    if any(
        hmmer.get(key) != value
        for key, value in {
            "version": "3.4",
            "arguments": [
                "--max", "--noali", "-T", "-1000", "--domT", "-1000",
                "--incT", "-1000", "--incdomT", "-1000",
            ],
            "score": "tblout_full_sequence_bit_score_field_6",
            "no_hit": "per_reference_full_landscape_minimum_finite_minus_one_bit",
        }.items()
    ):
        raise ValueError("native_tools.hmmer differs from the frozen hmmsearch protocol")
    if any(
        mmseqs.get(key) != value
        for key, value in {
            "release": "18-8cc5c",
            "commit": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
            "evalue": 1.0,
            "sensitivity": 7.5,
            "max_sequences": 2000,
            "format": "query,target,evalue,bits,pident,alnlen,qcov,tcov",
        }.items()
    ):
        raise ValueError("native_tools.mmseqs2 differs from the frozen protocol")

    interpretation = _mapping(raw["interpretability"], "interpretability")
    _require_exact(
        interpretation,
        required={
            "reference_id",
            "focal_position",
            "feature_activation_threshold",
        },
        label="interpretability",
    )

    if not isinstance(raw["expected_counts"], Mapping):
        raise ValueError("expected_counts must be a mapping")
    counts = _mapping(raw["expected_counts"], "expected_counts")
    _require_exact(
        counts,
        required={"full_total", "remote_total", "remote_analogues", "remote_negatives", "by_reference"},
        label="expected_counts",
    )
    if int(counts["remote_total"]) != int(counts["remote_analogues"]) + int(counts["remote_negatives"]):
        raise ValueError("global remote class counts do not sum to remote_total")
    by_reference = _mapping(counts["by_reference"], "expected_counts.by_reference")
    if set(by_reference) != set(references):
        raise ValueError("expected_counts.by_reference keys differ from configured references")
    totals = {"full": 0, "remote_total": 0, "remote_analogues": 0, "remote_negatives": 0}
    for reference_id, value in by_reference.items():
        row = _mapping(value, f"expected_counts.by_reference.{reference_id}")
        _require_exact(
            row,
            required={"full", "remote_analogues", "remote_negatives", "remote_total"},
            label=f"expected_counts.by_reference.{reference_id}",
        )
        if int(row["remote_total"]) != int(row["remote_analogues"]) + int(row["remote_negatives"]):
            raise ValueError(f"remote counts do not sum for {reference_id}")
        if int(row["full"]) != int(references[reference_id]["expected_full"]):
            raise ValueError(f"expected full count differs for {reference_id}")
        for key in totals:
            totals[key] += int(row[key])
    if totals["full"] != int(counts["full_total"]) or any(
        totals[key] != int(counts[key])
        for key in ("remote_total", "remote_analogues", "remote_negatives")
    ):
        raise ValueError("per-reference expected counts do not sum to global counts")
    focal_reference = str(interpretation["reference_id"])
    if focal_reference not in references or references[focal_reference]["focal_position"] != 20:
        raise ValueError("interpretability reference must be a configured focal-position-20 target")
    return PipelineConfig(config_path.resolve(), raw)
