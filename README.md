# hf-genomic-dataset-enrichment


Enrichment pipeline for the [carbon-pretraining-corpus](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus) .
## Overview


```text
Hugging Face datasets
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
        +--------------------+
        |                    |
        v                    v
   ClickHouse             LanceDB
   ------------            -------
   SQL analytics           Embeddings
   Cohorts                 Vector search
   Taxonomy                Nearest neighbors
   Metadata                Similar sequences
   Data quality            Neighborhoods
```

## Storage responsibilities

| System                  | Primary use                           |
| ----------------------- | ------------------------------------- |
| **Faceberg**            | Dataset and table inspection          |
| **dagster-hf-datasets** | Dataset enrichment and transformation |
| **ClickHouse**          | Analytical and metadata queries       |
| **LanceDB**             | Embedding storage and vector search   |

## Questions supported

| Question type                        | Faceberg | ClickHouse | LanceDB |
| ------------------------------------ | :------: | :--------: | :-----: |
| What tables exist?                   |   **●**  |            |         |
| What is in the corpus?               |          |    **●**   |         |
| What is the distribution of species? |          |    **●**   |         |
| What is the data quality?            |          |    **●**   |         |
| What sequences are similar?          |          |            |  **●**  |
| Find exact metadata matches          |          |    **●**   |         |
| Hybrid semantic + metadata retrieval |          |    **●**   |  **●**  |
| Compare retrieval approaches         |          |    **●**   |  **●**  |
| Explain a retrieved neighborhood     |          |    **●**   |  **●**  |
| Answer multi-step research questions |          |    **●**   |  **●**  |

The table describes where each operation is performed, rather than assigning every system a general-purpose role.

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

## ClickHouse

ClickHouse is the analytical store for the enriched dataset.

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




