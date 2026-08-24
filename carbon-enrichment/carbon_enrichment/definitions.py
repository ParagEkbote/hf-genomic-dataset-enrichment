"""
Dagster definitions for the Carbon enrichment pipeline.

Pipeline:

    Hugging Face Hub
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
            ▼
    carbon_gpu_enrichment
            ├── carbon_embeddings
            └── carbon_likelihood_stats
                        │
                        ▼
                carbon_likelihood_summary
"""

import dagster as dg

from carbon_enrichment.assets.cpu.streaming import (
    carbon_cpu_enriched_sequences,
)
from carbon_enrichment.assets.derived.likelihood_embedding_features import (
    carbon_likelihood_summary,
)
from carbon_enrichment.assets.gpu.embeddings import (
    carbon_gpu_enrichment,
)
from carbon_enrichment.assets.gpu.sampling import (
    carbon_pilot_corpus,
)
from carbon_enrichment.assets.gpu.tokenize_and_tag import (
    carbon_tokenized_corpus,
)
from carbon_enrichment.resources.carbon import (
    CarbonModelResource,
)
from carbon_enrichment.resources.hf_client import (
    create_huggingface_resource,
)


CPU_ASSETS = [
    carbon_cpu_enriched_sequences,
]

GPU_ASSETS = [
    carbon_pilot_corpus,
    carbon_tokenized_corpus,
    carbon_gpu_enrichment,
]

ANALYSIS_ASSETS = [
    carbon_likelihood_summary,
]


# ============================================================================
# Jobs
# ============================================================================

carbon_cpu_job = dg.define_asset_job(
    name="carbon_cpu_job",
    selection=dg.AssetSelection.keys(
        "carbon_cpu_enriched_sequences",
    ),
)

carbon_gpu_job = dg.define_asset_job(
    name="carbon_gpu_job",
    selection=dg.AssetSelection.keys(
        "carbon_pilot_corpus",
        "carbon_tokenized_corpus",
        "carbon_embeddings",
        "carbon_likelihood_stats",
    ),
)

carbon_analysis_job = dg.define_asset_job(
    name="carbon_analysis_job",
    selection=dg.AssetSelection.keys(
        "carbon_likelihood_summary",
    ),
)


# ============================================================================
# Definitions
# ============================================================================

defs = dg.Definitions(
    assets=[
        *CPU_ASSETS,
        *GPU_ASSETS,
        *ANALYSIS_ASSETS,
    ],
    jobs=[
        carbon_cpu_job,
        carbon_gpu_job,
        carbon_analysis_job,
    ],
    resources={
        "hf_resource": create_huggingface_resource(),
        "carbon": CarbonModelResource(),
    },
)