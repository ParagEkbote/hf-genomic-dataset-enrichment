"""
Dagster definitions for the Carbon enrichment pipeline.

M1 currently exposes the CPU ingestion and validation layer:

    Hugging Face Hub
            │
            ▼
    carbon_raw_sequences
            │
            ▼
       asset checks

The Hugging Face integration is provided by dagster-hf-datasets:

- HuggingFaceResource: dataset loading and Hub interaction
- HFParquetIOManager: persistence of materialized HF Dataset objects

The GPU and publishing layers will be added to this Definitions object as
the corresponding milestones become active.
"""

from __future__ import annotations

import dagster as dg

from dagster_hf_datasets import (
    HFParquetIOManager,
    HuggingFaceResource,
)

from carbon_enrichment.assets.cpu.ingest import carbon_raw_sequences
from carbon_enrichment.assets.cpu.validation import (
    check_coordinates,
    check_gene_boundary_pairing,
    check_raw_schema,
    check_required_field_completeness,
    check_sequence_alphabet,
    check_taxonomy_format,
    check_token_vocabularies,
)


# ============================================================================
# M1 Assets
# ============================================================================

CPU_ASSETS = [
    carbon_raw_sequences,
]


# ============================================================================
# M1 Asset Checks
# ============================================================================

CPU_ASSET_CHECKS = [
    check_raw_schema,
    check_token_vocabularies,
    check_gene_boundary_pairing,
    check_sequence_alphabet,
    check_coordinates,
    check_taxonomy_format,
    check_required_field_completeness,
]


# ============================================================================
# Dagster Definitions
# ============================================================================

defs = dg.Definitions(
    assets=CPU_ASSETS,
    asset_checks=CPU_ASSET_CHECKS,
    resources={
        # The key MUST match the resource parameter used by
        # carbon_raw_sequences(..., hf_resource: HuggingFaceResource).
        "hf_resource": HuggingFaceResource(
            cache_dir=".hf_cache",
        ),

        # The key MUST match io_manager_key="hf_parquet_io_manager"
        # on carbon_raw_sequences.
        "hf_parquet_io_manager": HFParquetIOManager(
            base_dir=".dagster_hf_storage",
        ),
    },
)