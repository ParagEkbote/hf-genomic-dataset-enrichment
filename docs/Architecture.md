# Architecture

Module and resource ownership for `carbon-enrichment`. This is a map of where things live in the code, not a run guide.

## Pipeline wiring (`definitions.py`)

```
Hugging Face Hub
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
        |
   +----+----+
   |         |
   v         v
carbon_embeddings   carbon_likelihood_stats
                            |
                            v
                    carbon_likelihood_summary
```

`definitions.py` is the single Dagster wiring point: it imports the assets below, groups them, defines jobs, and registers resources. It does not contain pipeline logic itself.

**Asset groups**
- `CPU_ASSETS` — `carbon_cpu_enriched_sequences`
- `GPU_PIPELINE_ASSETS` — `carbon_pilot_corpus`, `carbon_tokenized_corpus`, `carbon_gpu_enrichment`

**Jobs** (each a scoped asset selection, all on `in_process_executor` — nested multiprocessing between Dagster's step workers and the assets' own `ProcessPoolExecutor`/GPU pools would otherwise deadlock)
| Job | Assets |
|---|---|
| `carbon_cpu_job` | `carbon_cpu_enriched_sequences` |
| `carbon_tokenize_job` | `carbon_tokenized_corpus` |
| `carbon_inference_job` | `carbon_embeddings`, `carbon_likelihood_stats` |
| `carbon_gpu_job` | full GPU path: sampling → tokenize → embeddings + likelihood |

**Registered resources:** `hf_resource` (`create_huggingface_resource()`), `carbon` (`CarbonModelResource()`).

## Assets — CPU (`assets/cpu/`)

| Module | Owns |
|---|---|
| `ingest.py` | Builds the lazy HF `IterableDataset` for the selected Carbon pretraining-corpus validation tier. Split selection (`"train"`) and row-count truncation (`.take(n)`) are separate steps because `IterableDataset` has no length to slice against. The stream is never materialized. |
| `streaming.py` | Orchestrates the CPU stage: pulls bounded `pa.RecordBatch`es from the ingest stream, runs schema validation once (first batch, main process), fans batches out to a worker pool for normalize → validate → enrich, and writes Parquet shards from the main process. `pa.RecordBatch` is the canonical transport type throughout — no Dataset, DataFrame, or unbounded Python collection is ever materialized. |
| `normalization.py` | Deterministic, batch-oriented normalization via `pyarrow.compute` column kernels (not per-row Python): whitespace-strip string fields, uppercase nucleotide sequences, normalize taxonomy/categorical whitespace. Preserves all input columns, never drops records, never infers biology, never calls model inference. |
| `enrichment.py` | Adds derived features — base composition, GC content/skew, Shannon entropy, 3-mer frequency vectors — as vectorized NumPy array operations over the whole batch (Arrow string column → fixed-width byte array → 2D uint8 array → broadcast math), not a per-row loop. Strand reverse-complement is the one deliberately non-vectorized piece (ragged per-row substrings don't vectorize cleanly). No Dataset materialization, no Parquet I/O. |
| `validation.py` | Incremental validation and quality-stat accumulation over bounded batches. Never calls `Dataset.to_pandas()`, `.filter()`, or `.map()`; never materializes the full corpus; raw taxonomy/sequence data is never truncated. `ValidationStats` is mutated in place per worker process — instances must not be shared across processes; `merge_validation_stats()` combines per-worker results in the parent process. |

## Assets — GPU (`assets/gpu/`)

| Module | Owns |
|---|---|
| `sampling.py` | Stratified sampling from `carbon_cpu_enriched_sequences` (~32M rows) down to `carbon_pilot_corpus` (~25% subset). Inclusion is decided per row via a deterministic hash of `record_id`, with a per-row inclusion probability identical across strata — this is proportional stratified sampling in one streaming pass (no reservoir, no second pass to learn stratum sizes first). Stratifies on `sequence_length` (raw bp, already available from CPU enrichment) rather than token length, since token length isn't known until tokenization runs. |
| `tokenize_and_tag.py` | Standalone stage between CPU-enriched Parquet and GPU enrichment: wraps sequences (`<dna>{seq}</dna>`), filters to canonical uppercase ACGT (else `<oov>`), truncates to a multiple of 6, tokenizes with the Carbon hybrid 6-mer tokenizer, writes checkpointed token-ID shards keyed by `record_id`. Kept separate so a tokenization bug never forces re-running CPU enrichment and a GPU-stage bug never forces re-tokenizing. Written Arrow-native end to end (`pyarrow.compute` over whole columns); only the tokenizer call itself touches non-Arrow Python `str`. Includes a fail-fast check (`_assert_dna_mode_active`) against a known probe sequence, since a missing `<dna>` tag silently switches the tokenizer to BPE/English mode instead of erroring. |
| `embeddings.py` | GPU enrichment forward pass: produces `carbon_embeddings` and feeds `carbon_likelihood_stats`. Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at import time to guard against VRAM fragmentation above ~70GB occupancy. Writes via the same `ParquetShardWriter` used by the CPU streaming stage. |

## Resources (`resources/`)

| Module | Owns |
|---|---|
| `carbon.py` | `CarbonModelResource` — the single project-level source of the Carbon-3B model and its hybrid 6-mer tokenizer. Resolves the pinned `HuggingFaceBio/Carbon-3B` revision once per process (avoiding version skew between assets that would otherwise each call `from_pretrained` independently), loads the tokenizer with `trust_remote_code=True`, loads the model in bfloat16 (its native training precision, not a throughput choice), wires FlashAttention-2 via the HF Kernels Hub (not pip `flash-attn`), and puts the model in `.eval()` mode. Exposes `model_checkpoint`, `tokenizer_revision`, and the FA2 kernel revision for the provenance manifest. Lazy-loads both tokenizer and model so config-only access never triggers a 3B-parameter weight load. |
| `hf_client.py` | Thin project-specific configuration wrapper around `dagster_hf_datasets.HuggingFaceResource` (cache location, single project-level resource instance). Delegates all actual dataset I/O to that library rather than implementing a second client. Consumed by `assets/cpu/ingest.py` via Dagster dependency injection. |
| `clickhouse.py` | `ClickHouseResource`/`ClickHouseConfig` — local ClickHouse process management and query execution (binary path, thread count, temp directory defaults). Backs the analytical-scale queries described in the architecture overview and the Phase 4.5 case study. |
| `duckdb.py` | Connection helper plus pipeline-level constants used for local/ad-hoc validation — e.g. `PRETRAINING_SPLIT_ROW_COUNT`, `CPU_ENRICHED_EXPECTED_FRACTION`, and the length-bucket labels used when checking stratified-sampling representativeness against `faceberg.PIPELINE_TABLES`. |
| `faceberg.py` | The single read path for catalog-managed lineage datasets (`cpu_enriched`, `sampled_cpu`, `tokenized`, `likelihood_stats`, `embeddings`) spanning provenance through retrieval evaluation. Maps existing HF datasets to Iceberg table metadata without copying data. The raw pretraining corpus/split are intentionally *not* cataloged here — they remain streaming inputs. Lineage: `pretraining_split → cpu_enriched → sampled_cpu → tokenized → {likelihood_stats, embeddings}`. |
| `lancedb.py` | `LanceDBConfig`/`LanceDBResource` — local (or S3) LanceDB table for `carbon_embeddings`, with configurable distance metric (default cosine) and retry handling for OS-level file-lock contention. |

## Results (`results/`) (STILL WIP)

| Module | Owns |
|---|---|
| `derived/case_study.py` | The Phase 4.5 analysis script — reads `carbon-pilot-corpus-dedup`, `carbon-likelihood-stats`, `carbon-embeddings` directly from HF via ClickHouse, joins on a common cohort key set, and writes `results/case_study/phase45/*`. Owns the outlier/z-score constants (`DEFAULT_OUTLIER_Z = 3.0`, `DEFAULT_MIN_TAXON_N = 20`, biological sanity ranges for GC/entropy/perplexity, recurrent embedding dimensions). |
| `derived/distributions.py` | ClickHouse-backed distributional statistics over sequence data, using Numba-accelerated kernels on re-chunked batches (`BATCH_SIZE = 50_000`) for consistent shapes across ClickHouse's own Arrow-stream block size. |
| `derived/likelihood_distributions.py` | Joins likelihood/perplexity output back to sequence-derived features (`sequence_length`, `gc_content`, `entropy`, `is_coding_region`, via alias resolution) on `(record_id, start, end)`. |
| `derived/embedding_features.py` | Loads embedding Parquet output and pushes it into LanceDB via `LanceDBResource`/`LanceDBConfig`. |
| `derived/provenance.py` | Builds the run provenance manifest from the Faceberg catalog (`get_catalog`, `catalog_managed_tables`). |
| `derived/script.py` | Standalone CLI entry point for taxonomy-rank analysis over `AINovice2005/carbon-pilot-corpus-dedup`, joined on `record_id, start, end`; supports the taxonomy ranks from `domain` through subordinate levels. |
| `notebook/results.py`, `visualization/*.py` | Present but currently empty — not yet implemented. |

## Shared contracts

- **`schema.py`** — the data-contract module, deliberately independent of Dagster. Defines HF dataset identifiers, validation tiers, the expected raw schema, allowed categorical/token values, valid gene-boundary pairs, the IUPAC nucleotide alphabet, taxonomy/coordinate expectations, and GPU output contracts (embedding/likelihood columns, token-mask semantics). Explicitly lossless — must not impose stricter constraints than the raw dataset actually guarantees.
- **`config.py`** — runtime configuration only (`CarbonPipelineConfig`, a single `dagster.Config` object covering the full pipeline, deliberately not split per-asset). Data contracts live in `schema.py`; Dagster wiring lives in `definitions.py` — this module is neither.

## Catalog metadata (`carbon-catalog/`)

Iceberg-style metadata (`faceberg.yml`, `lineage.yml`, `lineage_report.csv`, `provenance.json`) plus per-table Avro metadata for `cpu_enriched_sequences`, `embeddings`, `likelihood_stats`, `pilot_corpus_dedup`, and `tokenized_corpus`. This is what `resources/faceberg.py` reads from — table structure and lineage, not row data.

## Metrics output map (`metrics/derived/`)

| Path | Corresponds to |
|---|---|
| `common_cohort/common_cohort_taxonomy.csv` | The shared cohort key set used to join CPU/likelihood/embeddings sources for downstream analysis (Phase 4.5 and others). |
| `phase2_csv/` | Feature-validation outputs: Fickett-proxy validation, GC skew vs. coding status / taxonomy class, length–GC distribution, range sanity checks, stop-codon validation, taxonomy cardinality. |
| `phase3/distributions/` | Cohort-level distribution analysis: `layer1_cohort_summary`, `layer1_length_deciles`, `layer2_length_conditional`, `layer2_length_model`, `layer3_record_metrics`, plus `analysis_config.json` for the run parameters. |
| `phase4/` | Retrieval evaluation: `knn_neighbors.csv` (LanceDB nearest-neighbor results) and `lancedb_metrics.csv`. |

`results/case_study/phase45/` (produced by `derived/case_study.py`, not checked into `metrics/`) is the anomaly-scoring and information-gain output described in the main architecture reference doc. (STILL WIP)