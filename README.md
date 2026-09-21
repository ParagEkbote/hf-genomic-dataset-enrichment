# hf-genomic-dataset-enrichment

Data Enrichment pipeline for the [carbon-pretraining-corpus](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus) dataset.

## Lineage of Datasets

Explore the interactive diagram and their schemas:

[View the Interactive Pipeline Diagram](https://htmlpreview.github.io/?https://github.com/ParagEkbote/hf-genomic-dataset-enrichment/blob/main/docs/carbon_pipeline_with_schemas.html)


- [`carbon-cpu-enriched-sequences`](https://huggingface.co/datasets/AINovice2005/carbon-cpu-enriched-sequences) — output of the CPU enrichment stage; adds sequence-derived features (length, GC content, coding status, strand, taxonomy) to the pretraining split. Source for stratified sampling into the GPU input corpus.
- [`carbon-cpu-enriched-sequences-sampled`](https://huggingface.co/datasets/AINovice2005/carbon-cpu-enriched-sequences-sampled) — A deduplicated sample upstream of `carbon-cpu-enriched-sequences` (3.22M records). Each row is a gene/sequence record with boundary/framing tokens, biological annotation (gene type, strand, taxonomy lineage, topology), raw and strand-normalized sequence, and derived features (GC content, GC skew, Shannon entropy, k-mer frequency vector, coding-region flag, QC flag).
- [`carbon-likelihood-stats`](https://huggingface.co/datasets/AINovice2005/carbon-likelihood-stats) — per-record likelihood/perplexity statistics produced by running the Carbon-3B model over the sampled CPU population under a fixed token budget.
- [`carbon-embeddings`](https://huggingface.co/datasets/AINovice2005/carbon-embeddings) — per-record embedding vectors produced by the same GPU enrichment pass, indexed into LanceDB for nearest-neighbor and similarity search.

CPU sampling controls representativeness of the population entering GPU enrichment. GPU token budgeting controls compute cost of the model pass itself.

## Start Here

| Need | Documentation |
|---|---|
| Understand module ownership and system boundaries | [Architecture](docs/Architecture.md) |
| Follow asset execution, resources, and configuration | [Pipeline Guide](docs/Pipeline.md) |
| Find materialized outputs, metrics, and provenance | [Results and Outputs](docs/Results.md) |

### Core References

- [Dagster definitions](carbon-enrichment/carbon_enrichment/definitions.py) — asset graph, jobs, and registered resources.
- [Runtime configuration](carbon-enrichment/carbon_enrichment/config.py) — shared `CarbonPipelineConfig` fields and defaults.
- [Data contracts](carbon-enrichment/carbon_enrichment/schema.py) — source schema, join keys, and derived output columns.
- [Development run configurations](carbon-enrichment/carbon_enrichment/config/) — CPU, sampling, tokenization, and GPU YAML files.

## Overview

```text
Hugging Face datasets
        |
        v
     Dagster
        |
        v
dagster-hf-datasets
        |
        |  enrichment
        |  taxonomy / metadata
        |  derived records
        v
   Enriched data
        |
        +----------+----------+----------+
        |          |          |          |
        v          v          v          v
   Faceberg     DuckDB   ClickHouse   LanceDB
   --------     ------    ----------   -------
   Table &      Ad-hoc    SQL analytics Embeddings
   dataset      local     Cohorts       Vector search
   inspection   queries   Taxonomy      Nearest neighbors
                          Metadata      Similar sequences
                          Data quality  Neighborhoods
```

## Resource Responsibilities

| System                  | Primary use                           |
| ----------------------- | ------------------------------------- |
| **Faceberg**            | Dataset and table inspection          |
| **dagster-hf-datasets** | Dataset enrichment and transformation |
| **DuckDB**              | Local/ad-hoc analytical queries over Parquet/Arrow outputs |
| **ClickHouse**          | Analytical and metadata queries at scale |
| **LanceDB**             | Embedding storage and vector search   |

## Questions supported

| Question type                        | Faceberg | DuckDB | ClickHouse | LanceDB |
| ------------------------------------ | :------: | :----: | :--------: | :-----: |
| What tables exist?                   |   **●**  |        |            |         |
| What is in the corpus?               |          |  **●** |    **●**   |         |
| What is the distribution of species? |          |  **●** |    **●**   |         |
| What is the data quality?            |          |  **●** |    **●**   |         |
| What sequences are similar?          |          |        |            |  **●**  |
| Find exact metadata matches          |          |  **●** |    **●**   |         |
| Hybrid semantic + metadata retrieval |          |        |    **●**   |  **●**  |
| Compare retrieval approaches         |          |        |    **●**   |  **●**  |
| Explain a retrieved neighborhood     |          |        |    **●**   |  **●**  |

The table describes where each operation is performed.

## Enrichment

The enrichment stage operates on the source dataset and adds structured information needed for downstream analysis.

Typical fields include:

* `record_id`
* `start`
* `end`
* sequence information
* taxonomy
* derived metadata
* enrichment or analysis outputs

The enrichment process is designed to preserve the relationship between derived records and their source records.

## DuckDB

DuckDB is used for local, ad-hoc analytical work directly against Parquet/Arrow outputs of the pipeline, without standing up or querying ClickHouse.

It is used for operations such as:

* joining CPU-enriched, likelihood-stats, and embeddings outputs on a common cohort key set
* one-off aggregation and statistics during pipeline development
* validating stratified sampling representativeness (stratum proportions in the sampled population vs. the full CPU-enriched corpus)
* quick data-quality checks before promoting queries to ClickHouse

## ClickHouse

ClickHouse is the analytical store for the enriched dataset at scale.

It is used for operations such as:

* taxonomy distributions
* species and higher-rank summaries
* metadata filtering
* exact record matching
* cohort construction
* aggregation of enrichment measurements
* data-quality checks
* comparison of derived results

This avoids scanning the full source corpus for every analytical query.

## LanceDB

LanceDB stores the embedding representation of sequence records.

It is used for:

* nearest-neighbor search
* sequence similarity
* retrieval of local neighborhoods
* combining vector results with structured metadata from ClickHouse

The vector index is therefore complementary to the analytical representation in ClickHouse.

## Documentation

- [Architecture](docs/Architecture.md) — module ownership, pipeline wiring, and system boundaries.
- [Pipeline Guide](docs/Pipeline.md) — asset execution, resources, configuration, and storage.
- [Results and Outputs](docs/Results.md) — materialized outputs, analysis modules, metrics, and provenance.
