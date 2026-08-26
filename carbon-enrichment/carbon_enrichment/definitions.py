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

import os

# 1. Disable HF Rust threadpool cloning to prevent deadlock on fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Safeguard against CUDA virtual memory address fragmentation at high VRAM (>70GB)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# ============================================================================
# Asset Groupings
# ============================================================================

CPU_ASSETS = [
    carbon_cpu_enriched_sequences,
]

GPU_PIPELINE_ASSETS = [
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

# Standalone CPU sequence enrichment
carbon_cpu_job = dg.define_asset_job(
    name="carbon_cpu_job",
    selection=dg.AssetSelection.keys("carbon_cpu_enriched_sequences"),
    executor_def=dg.in_process_executor,
)

# Tokenize only (internal ProcessPoolExecutor manages its own worker forks)
carbon_tokenize_job = dg.define_asset_job(
    name="carbon_tokenize_job",
    selection=dg.AssetSelection.keys(
        "carbon_tokenized_corpus",
    ),
    executor_def=dg.in_process_executor,
)

# GPU Enrichment only (single forward pass for embeddings + likelihood)
carbon_inference_job = dg.define_asset_job(
    name="carbon_inference_job",
    selection=dg.AssetSelection.keys(
        "carbon_embeddings",
        "carbon_likelihood_stats",
    ),
    executor_def=dg.in_process_executor,
)

# Full End-to-End GPU pipeline
# Using in_process_executor to prevent nested multiprocessing / IPC pipe deadlocks
# between Dagster's step workers and the asset's internal ProcessPoolExecutor.
carbon_gpu_job = dg.define_asset_job(
    name="carbon_gpu_job",
    selection=dg.AssetSelection.keys(
        "carbon_pilot_corpus",
        "carbon_tokenized_corpus",
        "carbon_embeddings",
        "carbon_likelihood_stats",
    ),
    executor_def=dg.in_process_executor,
)

# Post-processing analytics & distribution summaries
carbon_analysis_job = dg.define_asset_job(
    name="carbon_analysis_job",
    selection=dg.AssetSelection.keys("carbon_likelihood_summary"),
    executor_def=dg.in_process_executor,
)


# ============================================================================
# Definitions
# ============================================================================

defs = dg.Definitions(
    assets=[
        *CPU_ASSETS,
        *GPU_PIPELINE_ASSETS,
        *ANALYSIS_ASSETS,
    ],
    jobs=[
        carbon_cpu_job,
        carbon_tokenize_job,
        carbon_inference_job,
        carbon_gpu_job,
        carbon_analysis_job,
    ],
    resources={
        "hf_resource": create_huggingface_resource(),
        "carbon": CarbonModelResource(),
    },
)
