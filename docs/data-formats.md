# Data formats

The confidence-control revision additionally accepts keyed confidence and
IUPred summary tables. See [confidence-controls.md](confidence-controls.md)
for their schema, historical aliases, outputs and complete-cohort requirements.

All manifest paths are relative to `--data-root`. Protein sequences are normalized to uppercase, with whitespace removed, and must contain only the 20 canonical amino-acid letters.

## Candidate identity

Scientific joins use a sequence-derived candidate key rather than `sequence_id`:

```text
sequence_sha256 = SHA256(normalized_sequence encoded as ASCII)
candidate_key = (reference_id, sequence_sha256)
```

`sequence_id` is retained only as display/provenance metadata. The same sequence may therefore be evaluated against multiple references, but duplicate sequences within a reference are rejected.

## Input tables

### `candidates.csv`

Contains the candidate sequence, its `reference_id`, `sequence_id`, and configured `generation_method`. Candidate sequences must have the same length as the corresponding WT.

### `references.csv`

Defines each WT reference and contains:

```text
reference_id
reference_sequence_id
sequence
structure_path
chain_id
pfam_accession
pfam_profile_path
figure_focal_position
```

The supplied Pfam HMM must match the configured versioned `pfam_accession`.

### `reference_positions.csv`

Maps one-based WT sequence positions to PDB residue numbering and records whether coordinates are present. This allows unresolved WT residues and PDB numbering offsets to be handled explicitly.

### `structures.csv`

Maps each candidate key to its predicted structure and chain. Candidate structures must contain the complete candidate sequence in coordinate order. WT structures may omit positions marked unobserved in `reference_positions.csv`.

Structure parsing conventions, including alternate locations and modified residues, are described in `docs/methods.md`.

### `figure_examples.csv`

Freezes the individual candidates displayed in Figure 2. Figure generation verifies their candidate keys and analysis classes rather than selecting replacement examples dynamically.

## Representation data

`landscape_representations.npz` stores the precomputed layer-60 ESM-C and SAE representations used by the analysis. It contains:

* mean-pooled 2560-dimensional ESM-C embeddings;
* sparse residue-level SAE activations;
* max-pooled SAE feature profiles; and
* the indices and offsets required to associate these arrays with individual sequences.

`representation_index.csv` maps representation rows to `reference_id`, sequence digest, and role (`candidate` or `wt_reference`). It contains every candidate and one WT representation per reference.

A companion `landscape_representations.json` records dataset identity, dimensions, checksums, and the scientific configuration used to generate the representations. Non-example data must originate from the live Biohub pipeline; synthetic tensors are accepted only for explicitly marked example configurations.

## Scorer outputs

Each scorer emits one row per candidate key together with a provenance sidecar. The master table is assembled only after confirming that scorer outputs contain the same candidate set.

The resulting table retains the structural and retrieval scores required for the analyses, with class labels derived from the configured thresholds.

Private Pfam recomputation additionally uses `pfam_censor_floors.csv`, which stores the full-sequence and best-domain censoring floors for each reference.

## Analysis cohorts

The four predefined cohorts are:

```text
full_landscape
focal_reference_full
focal_reference_remote
all_references_remote
```
