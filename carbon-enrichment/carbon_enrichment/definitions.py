"""
Dagster definitions for the Carbon enrichment pipeline.

CPU pipeline:

    Hugging Face Hub
            │
            ▼
    create_carbon_stream()
            │
            ▼
       bounded batches
            │
            ├── schema validation
            │
            ├── normalization
            │
            ├── validation
            │
            ├── NumPy enrichment
            │
            ▼
      Parquet shards
            │
            ▼
    carbon_cpu_enriched_sequences

The CPU pipeline uses bounded streaming batches and does not materialize
the complete Carbon corpus as a Hugging Face Dataset or pandas DataFrame.

The Hugging Face integration is provided by dagster-hf-datasets:

- HuggingFaceResource: dataset loading and Hub interaction

The GPU and publishing layers will be added as their corresponding
milestones become active.
"""

import dagster as dg

from carbon_enrichment.assets.cpu.streaming import (
    carbon_cpu_enriched_sequences,
)
from carbon_enrichment.resources.hf_client import (
    create_huggingface_resource,
)

CPU_ASSETS = [
    carbon_cpu_enriched_sequences,
]


defs = dg.Definitions(
    assets=CPU_ASSETS,
    resources={
        "hf_resource": create_huggingface_resource(),
    },
)
