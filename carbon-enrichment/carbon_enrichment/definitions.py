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

import dagster as dg
from dagster_hf_datasets.io_manager import HFParquetIOManager

from carbon_enrichment.assets.cpu.enrichment import (
    carbon_cpu_enriched_sequences,
)
from carbon_enrichment.assets.cpu.ingest import carbon_raw_sequences
from carbon_enrichment.assets.cpu.normalization import (
    carbon_normalized_sequences,
)
from carbon_enrichment.assets.cpu.validation import (
    check_coordinates,
    check_gene_boundary_pairing,
    check_raw_schema,
    check_required_field_completeness,
    check_sequence_alphabet,
    check_taxonomy_format,
    check_token_vocabularies,
)
from carbon_enrichment.resources.hf_client import (
    create_huggingface_resource,
)


CPU_ASSETS = [
    carbon_raw_sequences,
    carbon_normalized_sequences,
    carbon_cpu_enriched_sequences,
]


CPU_ASSET_CHECKS = [
    check_raw_schema,
    check_token_vocabularies,
    check_gene_boundary_pairing,
    check_sequence_alphabet,
    check_coordinates,
    check_taxonomy_format,
    check_required_field_completeness,
]


defs = dg.Definitions(
    assets=CPU_ASSETS,
    asset_checks=CPU_ASSET_CHECKS,
    resources={
        "hf_resource": create_huggingface_resource(),
        "hf_parquet_io_manager": HFParquetIOManager(
            base_dir=".dagster_hf_storage",
        ),
    },
)