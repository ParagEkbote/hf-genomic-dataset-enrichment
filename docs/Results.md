# Results and Outputs

This page describes the outputs produced by the enrichment pipeline and the analysis code that consumes them. Pipeline execution is documented in [Pipeline.md](Pipeline.md).

## Materialized Pipeline Outputs

| Output | Producer | Contents | Default location |
|---|---|---|---|
| `carbon_cpu_enriched_sequences` | CPU enrichment asset | Source fields plus sequence and quality features | `.dagster_hf_storage/carbon_cpu_enriched_sequences` |
| `carbon_pilot_corpus` | Sampling asset | Deterministic, stratified subset of CPU-enriched rows | `.dagster_hf_storage/carbon_pilot_corpus` |
| `carbon_tokenized_corpus` | Tokenization asset | `token_ids`, `token_mask`, `token_length`, and identity columns | `.dagster_hf_storage/carbon_tokenized_corpus` |
| `carbon_embeddings` | GPU enrichment asset | Embeddings and embedding norms | `.dagster_hf_storage/carbon_embeddings` |
| `carbon_likelihood_stats` | GPU enrichment asset | Log-probability, perplexity, and token-level likelihood statistics | `.dagster_hf_storage/carbon_likelihood_stats` |

Outputs are written as sharded Parquet data unless a downstream resource explicitly loads them into another analytical store.

## Shared Identity Contract

Derived outputs join back to source records using:

```text
(record_id, start, end)
```

The tokenized corpus contains:

- `record_id`
- `start`
- `end`
- `token_ids`
- `token_mask`
- `token_length`

Embedding outputs contain the identity columns, `embedding`, and `embedding_norm`. Likelihood outputs contain the identity columns plus:

- `mean_log_prob`
- `sum_log_prob`
- `perplexity`
- `supervised_position_count`
- `min_token_logprob`
- `argmin_position`
- `per_token_logprob_std`

The composite identity must be used for joins. Row position is not a stable identity across stages.

## Analysis Code

The analysis code lives under `carbon_enrichment/results/`.

### Derived Analysis

| Module | Responsibility |
|---|---|
| `derived/distributions.py` | Computes distributional summaries over sequence and enrichment data |
| `derived/likelihood_distributions.py` | Joins likelihood statistics to sequence-derived features and analyzes likelihood distributions |
| `derived/embedding_features.py` | Loads embedding outputs into LanceDB for vector analysis |
| `derived/case_study.py` | Produces the Phase 4.5 anomaly and information-gain analysis |
| `derived/provenance.py` | Builds provenance information from the Faceberg catalog |
| `derived/script.py` | Runs taxonomy-rank analysis over the shared cohort |

These modules are analytical consumers of materialized datasets. They are not Dagster assets in the current definitions.

### Notebook and Visualization Helpers

The `results/notebook/` directory contains reusable analysis helpers for CPU and GPU enrichment results, taxonomy indexing, and taxonomy lookups.

The `results/visualization/` directory contains phase-specific plotting modules:

- `phase2.py` for CPU feature and quality validation;
- `phase3.py`, `phase3_1.py`, and `phase3_2.py` for distribution analysis;
- `phase4.py` for retrieval analysis;
- `phase4_5.py` for the case-study analysis.

These modules should be treated as analysis tooling rather than pipeline stages.

## Metrics Directory

Generated and checked-in analytical outputs live under `metrics/derived/`.

| Directory | Purpose | Representative outputs |
|---|---|---|
| `common_cohort/` | Defines and analyzes the shared cohort used across data products | Taxonomy and sample-size summaries |
| `phase2_csv/` | Validates CPU-enriched biological and sequence features | GC content, sequence length, taxonomy composition, stop-codon checks |
| `phase3/distributions/` | Summarizes sequence and likelihood distributions | Cohort summaries, length deciles, conditional distributions, fitted models |
| `phase4/` | Evaluates embedding retrieval | KNN neighbors and LanceDB metrics |
| `phase4/level3/` | Examines biological and taxonomic relationships in nearest neighbors | Taxonomic consistency, shared taxonomy depth, biological correlations |

Metrics are derived artifacts, not raw pipeline inputs. Their generating analysis parameters should be recorded with the output when reproducibility matters.

## Catalog and Provenance

Catalog metadata is stored under `carbon_enrichment/carbon-catalog/` and is consumed by the Faceberg resource. It describes table structure, lineage, and provenance for catalog-managed outputs including CPU-enriched, sampled, tokenized, embedding, and likelihood datasets.

The intended lineage is:

```text
pretraining_split
      |
      v
cpu_enriched
      |
      v
sampled_cpu
      |
      v
tokenized
   +--+--+
   |     |
   v     v
embeddings  likelihood_stats
```

`results/derived/provenance.py` can use this catalog metadata to build a run-level provenance manifest. The raw streaming source remains an external Hugging Face input rather than a copied catalog table.

## Interpretation and Reproduction

When interpreting a result, record:

1. The source asset or dataset revision.
2. The validation level and sampling fraction.
3. The pilot sampling seed.
4. The model and tokenizer revisions for GPU-derived results.
5. The shared cohort key definition.
6. The analysis module and output phase.

CPU and GPU outputs should not be joined by row order. Use the composite biological key and verify key uniqueness or duplicate handling before aggregation.

## Current Scope

The results modules and visualization scripts are analytical tooling and are still evolving. The materialized asset contracts in `carbon_enrichment/schema.py` are the stable reference for tokenized, embedding, and likelihood column names.
