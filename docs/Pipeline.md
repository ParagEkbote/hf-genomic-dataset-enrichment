# Pipeline Guide

This page describes how the Carbon enrichment pipeline executes. The higher-level module ownership map is in [Architecture.md](Architecture.md).

**Related documentation:** [Architecture](Architecture.md) · [Results and Outputs](Results.md) · [README](../README.md)

## Data Flow

```text
Hugging Face Carbon corpus
          |
          v
carbon_cpu_enriched_sequences
          |
          v
carbon_pilot_corpus
          |
          v
carbon_tokenized_corpus
          |
          v
carbon_gpu_enrichment
       +--+--+
       |     |
       v     v
carbon_embeddings   carbon_likelihood_stats
```

The canonical identity of a biological sequence interval is the composite key:

```text
(record_id, start, end)
```

This key is preserved through the tokenized, embedding, and likelihood outputs.

## Dagster Jobs

| Job | Assets | Purpose |
|---|---|---|
| `carbon_cpu_job` | `carbon_cpu_enriched_sequences` | Stream, validate, normalize, and enrich the source corpus |
| `carbon_tokenize_job` | `carbon_tokenized_corpus` | Tokenize the sampled corpus in Carbon DNA mode |
| `carbon_inference_job` | `carbon_embeddings`, `carbon_likelihood_stats` | Run the GPU forward pass and write both derived outputs |
| `carbon_gpu_job` | Sampling, tokenization, embeddings, likelihood | Run the complete GPU path |

All jobs use Dagster's in-process executor. The CPU and tokenization assets manage their own bounded process pools, and the GPU asset manages model inference internally. Keeping Dagster in-process avoids nested worker-process and IPC conflicts.

## Assets

### CPU Enrichment

[`carbon_cpu_enriched_sequences`](../carbon-enrichment/carbon_enrichment/assets/cpu/streaming.py) reads the Carbon dataset as a streaming Hugging Face `IterableDataset`. It processes bounded Arrow record batches and writes Parquet shards.

The CPU stage performs these operations in order:

1. Validate the source schema on the first batch.
2. Normalize string, taxonomy, and nucleotide fields.
3. Accumulate validation and quality statistics.
4. Compute sequence-derived features such as length, GC content, GC skew, entropy, k-mer frequencies, coding status, and strand-normalized sequence.
5. Write the enriched batches to Parquet shards.

The complete source corpus is not materialized as a Dataset, DataFrame, or unbounded Python collection.

### Pilot Sampling

[`carbon_pilot_corpus`](../carbon-enrichment/carbon_enrichment/assets/gpu/sampling.py) reads the CPU-enriched corpus and writes a deterministic pilot subset. Each row receives an inclusion decision based on a SHA-256 hash of its seed and `record_id`.

Sampling is stratified using:

- a raw sequence-length bucket proxy;
- coding-region status;
- strand;
- taxonomy domain.

The sampler records the bounded input population and sampled population by stratum so representativeness drift can be reviewed after the run.

### Tokenization

[`carbon_tokenized_corpus`](../carbon-enrichment/carbon_enrichment/assets/gpu/tokenize_and_tag.py) converts the pilot corpus into Carbon model inputs.

The stage:

1. Filters rows according to the upstream canonical-DNA quality contract.
2. Uppercases and truncates sequences to the native context limit.
3. Aligns sequences to a multiple of six bases.
4. Wraps each sequence in `<dna>...</dna>` tags.
5. Runs the Carbon hybrid 6-mer tokenizer.
6. Writes `token_ids`, `token_mask`, and `token_length` with the biological composite key.

A probe sequence is checked before real work begins to ensure the tokenizer is actually operating in DNA 6-mer mode rather than silently falling back to BPE mode.

### GPU Enrichment

[`carbon_gpu_enrichment`](../carbon-enrichment/carbon_enrichment/assets/gpu/embeddings.py) consumes the tokenized corpus and performs one Carbon model forward pass per batch. It writes two materialized assets:

- `carbon_embeddings`: pooled hidden-state embeddings and embedding norms;
- `carbon_likelihood_stats`: log-probability, perplexity, and related token-level statistics.

The asset uses token-length buckets, bounded GPU token budgets, bfloat16 inference, and checkpointed/sharded output. OOM retries and runtime statistics are recorded as materialization metadata.

## Resources

| Resource | Module | Responsibility |
|---|---|---|
| `hf_resource` | [`resources/hf_client.py`](../carbon-enrichment/carbon_enrichment/resources/hf_client.py) | Loads the source Hugging Face dataset through `dagster-hf-datasets` |
| `carbon` | [`resources/carbon.py`](../carbon-enrichment/carbon_enrichment/resources/carbon.py) | Lazily loads the pinned Carbon model and hybrid tokenizer |
| `LanceDBResource` | [`resources/lancedb.py`](../carbon-enrichment/carbon_enrichment/resources/lancedb.py) | Stores and searches embedding vectors |
| `ClickHouseResource` | [`resources/clickhouse.py`](../carbon-enrichment/carbon_enrichment/resources/clickhouse.py) | Runs analytical queries at scale |
| DuckDB helpers | [`resources/duckdb.py`](../carbon-enrichment/carbon_enrichment/resources/duckdb.py) | Supports local Arrow/Parquet analysis and sampling checks |
| Faceberg helpers | [`resources/faceberg.py`](../carbon-enrichment/carbon_enrichment/resources/faceberg.py) | Reads catalog-managed tables and lineage metadata |

The Carbon model resource is lazy: importing definitions or loading configuration does not load the 3B-parameter model. Model and tokenizer revisions are exposed for provenance metadata.

## Configuration

Runtime configuration is defined by [`CarbonPipelineConfig`](../carbon-enrichment/carbon_enrichment/config.py). Development run configurations live in [`carbon_enrichment/config/`](../carbon-enrichment/carbon_enrichment/config/):

| File | Asset | Main purpose |
|---|---|---|
| [`dev-cpu.yaml`](../carbon-enrichment/carbon_enrichment/config/dev-cpu.yaml) | `carbon_cpu_enriched_sequences` | CPU validation level, streaming, batch and shard settings |
| [`dev-sampling.yaml`](../carbon-enrichment/carbon_enrichment/config/dev-sampling.yaml) | `carbon_pilot_corpus` | Sampling fraction, seed, input, and pilot output settings |
| [`dev-tokenize.yaml`](../carbon-enrichment/carbon_enrichment/config/dev-tokenize.yaml) | `carbon_tokenized_corpus` | Pilot input, tokenized output, shard, and worker settings |
| [`dev-gpu.yaml`](../carbon-enrichment/carbon_enrichment/config/dev-gpu.yaml) | `carbon_gpu_enrichment` | Tokenized input, GPU outputs, batch, and token budget settings |

The sampling and tokenization configurations must use the same `pilot_output_dir`.

## Storage and Execution Boundaries

By default, intermediate outputs are written below `.dagster_hf_storage/`:

- `carbon_cpu_enriched_sequences`
- `carbon_pilot_corpus`
- `carbon_tokenized_corpus`
- `carbon_embeddings`
- `carbon_likelihood_stats`

The pipeline is designed for bounded memory, resumable shard production, and explicit separation between CPU preprocessing, tokenization, and GPU inference. Running the full assets requires access to the Hugging Face dataset and, for model inference, a compatible GPU environment.

## CI Boundary

The repository CI checks source quality, Python compilation, YAML configuration structure, and Dagster definition imports. It does not download the full corpus, load the Carbon model, run GPU inference, or execute the production assets.
