"""
Runtime configuration for the Carbon enrichment pipeline.

Data contracts belong in schema.py.
Dagster asset/resource wiring belongs in definitions.py.
"""

from pathlib import Path
from typing import Literal

import dagster as dg

from carbon_enrichment.schema import DEFAULT_VALIDATION_LEVEL

# ============================================================================
# Complete Carbon pipeline configuration
# ============================================================================


class CarbonPipelineConfig(dg.Config):
    """
    Runtime configuration for the complete Carbon enrichment pipeline.

    A single Dagster Config object is used deliberately. Parameters annotated
    with separate dagster.Config classes are otherwise interpreted as asset
    inputs rather than configuration by Dagster.
    """

    # ------------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------------

    validation_level: Literal["dev", "integration", "auth"] = DEFAULT_VALIDATION_LEVEL

    # The CPU pipeline is streaming-only.
    streaming: bool = True

    # ------------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------------

    # Number of rows held in memory for one processing batch.
    batch_size: int = 1_000

    # Number of processed rows written to one Parquet shard.
    rows_per_shard: int = 250_000

    # ------------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------------

    compression: str = "zstd"

    # CPU-enriched source corpus.
    output_dir: str = ".dagster_hf_storage/carbon_cpu_enriched_sequences"

    # Output of the tokenize_and_tag stage (design doc #14.5) — deliberately
    # a separate directory from output_dir, not a subdirectory keyed off it,
    # so a tokenization-stage schema change or bug never requires touching
    # or re-running the CPU-enriched Parquet shards, and vice versa.
    tokenized_output_dir: str = ".dagster_hf_storage/carbon_tokenized_corpus"

    # Output of the single-pass GPU embedding stage.
    #
    # Kept independent from tokenized_output_dir so embedding representation
    # changes do not require re-tokenizing the corpus.
    embeddings_output_dir: str = ".dagster_hf_storage/carbon_embeddings"

    # Output of the single-pass GPU likelihood stage.
    #
    # Although embeddings and likelihood statistics are produced by the same
    # model forward pass, they remain separate materialized assets.
    likelihood_output_dir: str = ".dagster_hf_storage/carbon_likelihood_stats"

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    enable_progress: bool = True

    cpu_workers: int | None = None


# ============================================================================
# Local storage configuration
# ============================================================================


class StorageConfig(dg.Config):
    """Local storage configuration for development runs."""

    hf_cache_dir: str = ".hf_cache"

    dagster_storage_dir: str = ".dagster_hf_storage"

    @property
    def hf_cache_path(self) -> Path:
        return Path(self.hf_cache_dir)

    @property
    def dagster_storage_path(self) -> Path:
        return Path(self.dagster_storage_dir)


# ============================================================================
# Project defaults
# ============================================================================


DEFAULT_STREAMING: bool = True

DEFAULT_BATCH_SIZE: int = 1_000

DEFAULT_ROWS_PER_SHARD: int = 250_000

DEFAULT_PARQUET_COMPRESSION: str = "zstd"

DEFAULT_OUTPUT_DIR: str = ".dagster_hf_storage/carbon_cpu_enriched_sequences"

DEFAULT_TOKENIZED_OUTPUT_DIR: str = ".dagster_hf_storage/carbon_tokenized_corpus"

DEFAULT_EMBEDDINGS_OUTPUT_DIR: str = ".dagster_hf_storage/carbon_embeddings"

DEFAULT_LIKELIHOOD_OUTPUT_DIR: str = ".dagster_hf_storage/carbon_likelihood_stats"
