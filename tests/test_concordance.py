"""Boundary-value score tests; no fabricated coordinates or model predictions."""

import pandas as pd
import pytest

from protein_sae_pipeline.concordance import aggregate_results, comparisons


def score_case():
    cohort = pd.DataFrame(
        [
            {
                "id": "boundary",
                "family": "test",
                "generator": "test",
                "main": True,
                "existing": False,
                "selection_bin": 2,
            }
        ]
    )
    values = {"esmfold": [0.49], "colabfold": [0.5] * 5, "esmfold2": [0.4, 0.49, 0.5, 0.6, 0.7]}
    scores = pd.DataFrame(
        [
            {
                "id": "boundary",
                "model": m,
                "sample_id": i,
                "optimized_tm": v,
                "fixed_tm": v,
                "prediction_coverage": 1.0,
                "reference_coverage": 0.9,
                "normalization_length": 10,
            }
            for m, vals in values.items()
            for i, v in enumerate(vals, 1)
        ]
    )
    return cohort, scores


def test_inclusive_boundary_stability_crossing_and_deduplication():
    c, s = score_case()
    rows = comparisons(c, s)
    assert rows.median_flip.all()
    assert rows[rows.method.eq("colabfold")].stable_flip.all()
    assert rows[rows.method.eq("esmfold2")].crossing.all()
    assert not rows[rows.method.eq("esmfold2")].stable_flip.any()
    tables = aggregate_results(rows)
    assert tables["unique-design-summary"].any_median_disagreement.eq(1).all()
    for table in tables.values():
        assert not {"id", "sequence", "sequence_sha256", "x", "y"} & set(table)


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "extra_model", "coverage", "normalization", "boolean", "nan"]
)
def test_invalid_or_incomplete_scores_fail_closed(damage):
    c, s = score_case()
    if damage == "missing":
        s = s.iloc[:-1]
    if damage == "duplicate":
        s = pd.concat([s, s.iloc[:1]])
    if damage == "extra_model":
        s.loc[0, "model"] = "substitute"
    if damage == "coverage":
        s.loc[0, "prediction_coverage"] = 0.9
    if damage == "normalization":
        s.loc[0, "normalization_length"] = 11
    if damage == "boolean":
        c["main"] = "unknown"
    if damage == "nan":
        s.loc[0, "fixed_tm"] = float("nan")
    with pytest.raises(ValueError):
        comparisons(c, s)
