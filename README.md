# Foldprint

This repository implements the scoring, cohort, statistical, interpretability, and plotting pipeline used in the accompanying paper.

It does not publicly reproduce the paper's numerical results because the real candidate sequences and structures are withheld for reasons described in the Responsible Use Statement. Instead, the repository includes a small fictional dataset that exercises the same analysis pipeline. Its outputs have no biological or performance interpretation. `configs/example.yaml` provides the corresponding fictional example configuration.

## What is included

The repository contains:

* sequence identity, US-align, Pfam, MMseqs2, ESM-C, and SAE scoring;
* fixed cohort construction and low-FPR retrieval evaluation;
* paired stratified bootstrap analysis;
* IUPred3 scoring, pTM confidence matching, and matched-cohort re-evaluation;
* the interpretability analyses used in Figures 1–2;
* main and confidence-control/appendix figures and their plotted-data outputs; and
* a fictional dataset for testing the complete public pipeline.

Generation workflows and production folding are outside the scope of this repository. Candidate structures are treated as inputs; the folding settings used for the private paper cohort are documented in `docs/methods.md`.

## Fictional example data

Everything under `example/` is fictional and exists only to exercise the public pipeline. It includes two fictional references, 48 fictional candidates, synthetic structures, profile fixtures, and synthetic representation arrays.

None of these files correspond to paper-cohort proteins or model outputs. Generated figures and tables are written to `outputs/` and are not committed.

## Quick start

```bash
uv sync --frozen --extra test

uv run python scripts/run_example.py \
  --config configs/example.yaml \
  --data-root example \
  --output-root outputs/example \
  --offline

uv run pytest -m "not live"
```

The workflow requires HMMER 3.4, MMseqs2 release 18-8cc5c, and US-align version 20260819.

The offline example requires no Biohub token or network connection.

## Figures

`scripts/make_figure_1.py` and `scripts/make_figure_2.py` generate the interpretability figures. Figure 1 now includes panels A-D, with panel C calculated directly from the same complete cohort as panel B. The original `make_figure_3.py` command is a **pooled legacy analysis**, not the generator-stratified Figure 3 in the current paper. The current Figure 3 uses `scripts/make_confidence_figures.py --sections retrieval`. The [figure-to-command map](docs/figure-map.md) also covers every appendix plot and the offline concordance command `scripts/analyze_concordance.py`.

`scripts/make_confidence_figures.py` adds Figure 4, IUPred violins,
generator-specific retrieval summaries, full/matched score distributions and
full-range ROCs. The fictional example runs this workflow as well.

See [the confidence-control workflow](docs/confidence-controls.md) for commands,
input formats, matching rules and the correspondence to the reviewer draft.
Real IUPred scoring uses `scripts/score_iupred.py` with a separately installed
official IUPred3 package; the offline example uses clearly fictional scores.

See `docs/figure-map.md` for the panel-level mapping between analyses, inputs, and outputs.

## Optional live Biohub scoring

Optional live ESM-C and SAE scoring is available through `scripts/score_biohub.py`. It requires the `biohub` dependency extra, a Biohub API token, and the private SAE normalization asset.

The normalization vectors used for the paper analysis are not distributed with this repository.

## Reproducibility and release limits

The fictional example demonstrates that the public pipeline runs end to end; it does not reproduce the paper's biological results.


## Documentation

* `docs/methods.md` — computational methods and scoring definitions
* `docs/data-formats.md` — input tables and representation formats
* `docs/figure-map.md` — figure panels and plotted-data outputs

## License

This repository is released under the MIT License. Third-party tools, models, and assets retain their own licenses and citation requirements; see `THIRD_PARTY_NOTICES.md`.
