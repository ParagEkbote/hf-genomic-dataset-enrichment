"""
Dagster definitions for the Carbon enrichment pipeline.

Pipeline:

    Hugging Face Hub
            │
            ▼
    create_carbon_stream()
            │
            ▼
       bounded batches
            │
            ├── schema validation
            ├── normalization
            ├── validation
            ├── NumPy enrichment
            │
            ▼
      Parquet shards
            │
            ▼
    carbon_cpu_enriched_sequences
            │
            ▼
    carbon_tokenized_corpus
            │
            ├── token_ids
            ├── token_mask
            └── token_length
            │
            ▼
    carbon_gpu_enrichment
            │
            ├── carbon_embeddings
            └── carbon_likelihood_stats

The CPU pipeline uses bounded streaming batches and does not materialize
the complete Carbon corpus as a Hugging Face Dataset or pandas DataFrame.

The GPU pipeline consumes the checkpointed tokenized corpus and performs
a single model forward pass per batch to produce both embeddings and
likelihood statistics.

The Hugging Face integration is provided by dagster-hf-datasets:

- HuggingFaceResource: dataset loading and Hub interaction
"""

import dagster as dg

from carbon_enrichment.assets.cpu.streaming import (
    carbon_cpu_enriched_sequences,
)
from carbon_enrichment.assets.gpu.embeddings import (
    carbon_gpu_enrichment,
)
from carbon_enrichment.assets.gpu.tokenize_and_tag import (
    carbon_tokenized_corpus,
)
from carbon_enrichment.resources.hf_client import (
    create_huggingface_resource,
)
from carbon_enrichment.assets.gpu.likelihood import (
    carbon_likelihood_summary,
)
from carbon_enrichment.assets.gpu.sampling import (
    carbon_pilot_corpus,
)


CPU_ASSETS = [
    carbon_cpu_enriched_sequences,
]


GPU_ASSETS = [
    carbon_tokenized_corpus,
    carbon_gpu_enrichment,
    carbon_likelihood_summary, 
    carbon_pilot_corpus,
]


defs = dg.Definitions(
    assets=[
        *CPU_ASSETS,
        *GPU_ASSETS,
    ],
    resources={
        "hf_resource": create_huggingface_resource(),
    },
)