# Computational methods

This repository implements the analysis pipeline used in the paper. The real candidate sequences and structures are withheld for biosecurity reasons, so the public example uses fictional data to exercise the same analysis code. Its numerical results have no biological interpretation.

## Candidate identity

Sequences are normalized to uppercase canonical amino acids. Scientific joins use the key

```text
(reference_id, SHA256(sequence))
```

rather than `sequence_id`, which is retained only as display metadata. Candidate and cognate WT sequences must have equal length.

Full input and representation schemas are described in `docs/data-formats.md`.

## Sequence identity

Sequence identity is calculated as an ungapped, full-length positional comparison with the cognate WT:

```text
positional identity = identical positions / sequence length
```

Candidates with positional identity below 0.30 are considered remote.

## Structures

Candidate and WT PDB files are inputs to the repository. Candidate structures used in the paper were generated with `facebook/esmfold_v1` using FP32 inference and trunk chunk size 128.

Candidate structures must cover the full candidate sequence. WT structures may omit residues explicitly marked unobserved in `reference_positions.csv`. Only the configured protein chain and first PDB model are used.

## Structural similarity

Structural similarity is calculated with US-align using the candidate as Structure 1 and the WT as Structure 2:

```text
USalign candidate_ca.pdb wt_ca.pdb -mol prot
```

The primary structural score is the TM-score normalized by the resolved WT length. Designs with TM-score ≥ 0.50 are classified as structural analogues and those with TM-score < 0.50 as structural negatives.

Both TM-score normalization directions and standard alignment diagnostics are retained in the output.

### Local structural comparison

## ESM-C score

ESM-C scoring uses `esmc-6b-2024-12` hidden state 60. BOS and EOS tokens are removed and residue embeddings are mean-pooled to produce a 2,560-dimensional sequence representation.

The ESM-C score is the cosine similarity between each candidate representation and the corresponding WT representation.

## SAE score

The SAE analysis uses `esmc-6b-2024-12-sae-layer60-k64-codebook16384` at ESM-C layer 60.

After BOS/EOS removal, residue-level feature activations are normalized using the featurewise maximum and IDF values associated with the pretrained SAE. Each residue retains 64 active features. Sequence-level SAE profiles are obtained by taking the maximum activation of each feature across residues.

The SAE score is the cosine similarity between the candidate and WT max-pooled feature profiles.

The normalization vectors required for the paper analysis are private inputs and are not distributed in this repository.

## Pfam score

Each candidate is scored against one versioned cognate Pfam profile using HMMER 3.4 `hmmsearch`:

```text
hmmsearch --cpu N --max --noali \
  -T -1000 --domT -1000 --incT -1000 --incdomT -1000 \
  --tblout scores.tblout target_profile.hmm candidates.fasta
```

The retrieval score is the full-sequence bit score from `tblout`. Candidates absent from the output are assigned a reference-specific censoring floor derived from the complete landscape.

The cognate Pfam accession for each paper reference was fixed before evaluation.

## MMseqs2 score

MMseqs2 release 18-8cc5c compares each candidate only with its cognate WT:

```text
mmseqs easy-search candidates.fasta cognate_wt.fasta output.tsv temp_dir \
  -e 1.0 -s 7.5 --max-seqs 2000 --threads N \
  --format-output query,target,evalue,bits,pident,alnlen,qcov,tcov
```

The retrieval score is the MMseqs2 bit score. Candidates with no hit receive a score of 0.

## Analysis cohorts

Four fixed cohorts are used:

* `full_landscape`: all configured candidates
* `gpd_full`: complete focal-reference landscape
* `gpd_remote`: focal-reference candidates with <30% positional identity
* `six_target_remote`: remote candidates across all six references

## Retrieval evaluation

All retrieval scores are oriented so that larger values indicate greater similarity to WT.

For empirical recall at a target false-positive rate, thresholds are chosen conservatively so that the observed false-positive rate does not exceed the specified budget. The same procedure is used for the 1% and 5% operating points and for the 0–5% FPR ROC curves.

Confidence intervals are obtained by stratified bootstrap resampling within:

```text
(reference_id, structural_class, generation_method)
```

The same bootstrap samples are reused across retrieval methods, allowing paired comparisons. The paper analysis uses 10,000 replicates for operating points and 2,000 for ROC intervals.

## Interpretability analyses

WT SAE features are ranked by their max-pooled WT activation.

Feature recurrence is the fraction of candidate profiles in which a WT feature is active. For candidate profile `c` and WT profile `w`, the contribution of feature `j` to their cosine similarity is

```text
c[j] * w[j] / (norm(c) * norm(w))
```

and cumulative contributions are evaluated in WT feature-rank order.

Residue-level traces use the normalized SAE activations directly. A position-unique WT feature is defined as a feature active at one WT residue and nowhere else in the WT sequence.

The Figure 2 focal-position analyses additionally compare feature activation at the focal residue with its immediate neighbours and with the local structural RMSD defined above.

Displayed candidate examples are fixed by candidate key rather than selected dynamically.

## Public and private reproducibility

The public repository reproduces the paper's analysis pipeline but not its withheld numerical results.

Internally, release checks include both:

1. running the full downstream analysis against the private stored scores; and
2. recomputing all major scoring methods on a representative private subset and comparing them with the canonical records.
