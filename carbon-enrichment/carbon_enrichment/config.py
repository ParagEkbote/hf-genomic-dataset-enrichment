"""
Runtime configuration for the Carbon enrichment pipeline.

Data contracts belong in schema.py.
Fixed implementation constants belong in constants.py.
Dagster asset/resource wiring belongs in definitions.py.
"""

from pathlib import Path

import dagster as dg

from carbon_enrichment.schema import DEFAULT_VALIDATION_LEVEL


class IngestionConfig(dg.Config):
    """Runtime configuration for Carbon dataset ingestion."""

    validation_level: str = DEFAULT_VALIDATION_LEVEL


class CPUProcessingConfig(dg.Config):
    """Runtime configuration for CPU-side Dataset processing."""

    map_batch_size: int = 1_000

    # Keep this at 1 initially.
    #
    # Increase only after benchmarking the CPU enrichment stages. Hugging Face
    # Dataset.map() supports multiprocessing, but introducing it before the
    # baseline is measured makes debugging and reproducibility harder.
    num_proc: int = 1

    enable_progress: bool = True


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

DEFAULT_MAP_BATCH_SIZE: int = 1_000
DEFAULT_NUM_PROC: int = 1