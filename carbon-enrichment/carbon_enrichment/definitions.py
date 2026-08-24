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
    carbon_pilot_corpus
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
                        │
                        ▼
                carbon_likelihood_summary

The CPU pipeline uses bounded streaming batches and does not materialize
the complete Carbon corpus as a Hugging Face Dataset or pandas DataFrame.

The pilot corpus is a stratified subset of the CPU-enriched corpus
(design doc #22.5), used to bound GPU cost before scaling to the full
corpus.

The GPU pipeline consumes the checkpointed tokenized corpus and performs
a single model forward pass per batch to produce both embeddings and
likelihood statistics.

carbon_likelihood_summary is a CPU/analysis-stage asset that validates
and summarizes the likelihood stats produced by the GPU pass -- it does
not run a model forward pass itself (#14).

The Hugging Face integration is provided by dagster-hf-datasets:

- HuggingFaceResource: dataset loading and Hub interaction
"""

import dagster as dg

from carbon_enrichment.assets.cpu.streaming import (
    carbon_cpu_enriched_sequences,
)
from carbon_enrichment.resources.carbon import (
    CarbonModelResource,
)
from carbon_enrichment.assets.gpu.embeddings import (
    carbon_gpu_enrichment,
)
from carbon_enrichment.assets.derived.likelihood_embedding_features import (
    carbon_likelihood_summary,
)
from carbon_enrichment.assets.gpu.sampling import (
    carbon_pilot_corpus,
)
from carbon_enrichment.assets.gpu.tokenize_and_tag import (
    carbon_tokenized_corpus,
)
from carbon_enrichment.resources.hf_client import (
    create_huggingface_resource,
)

CPU_ASSETS = [
    carbon_cpu_enriched_sequences,
    carbon_pilot_corpus,
    carbon_tokenized_corpus,
]

GPU_ASSETS = [
    carbon_gpu_enrichment,
]

ANALYSIS_ASSETS = [
    carbon_likelihood_summary,
]


defs = dg.Definitions(
    assets=[
        *CPU_ASSETS,
        *GPU_ASSETS,
        *ANALYSIS_ASSETS,
    ],
    resources={
        "hf_resource": create_huggingface_resource(),
        "carbon": CarbonModelResource(),
    },
)