"""
Runtime configuration for the Carbon enrichment pipeline.

Data contracts belong in schema.py.
Fixed implementation constants belong in constants.py.
Dagster asset/resource wiring belongs in definitions.py.
"""

from pathlib import Path

import dagster as dg

from carbon_enrichment.schema import DEFAULT_VALIDATION_LEVEL


# ============================================================================
# Complete CPU pipeline configuration
# ============================================================================


class CarbonPipelineConfig(dg.Config):
    """
    Runtime configuration for the complete streaming Carbon CPU pipeline.

    A single Dagster Config object is used deliberately. Parameters annotated
    with separate dagster.Config classes are otherwise interpreted as asset
    inputs rather than configuration by Dagster.
    """

    # ------------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------------

    validation_level: str = DEFAULT_VALIDATION_LEVEL

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

    output_dir: str = (
        ".dagster_hf_storage/carbon_cpu_enriched_sequences"
    )

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    enable_progress: bool = True


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

DEFAULT_OUTPUT_DIR: str = (
    ".dagster_hf_storage/carbon_cpu_enriched_sequences"
)