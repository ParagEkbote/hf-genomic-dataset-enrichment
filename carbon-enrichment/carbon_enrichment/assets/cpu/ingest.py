"""
M1 — Data foundation assets.

Exit criteria (per revised plan): validated dataset at 1% (integration level).

Design notes:
  - `hf_dataset_asset` from dagster_hf_datasets fixes `split` at decoration
    time, which doesn't fit the 3-tier validation workflow (dev/integration/
    auth need to be selectable per run). So M1 uses a plain `@asset` backed
    directly by `HuggingFaceResource`, with the split string resolved from
    run config via `VALIDATION_LEVEL_SPLITS`.
  - Non-streaming loads are used throughout (HF split-slicing handles the
    row selection server/cache-side), which keeps `HFParquetIOManager`
    able to persist the result via `save_to_disk()` at every level,
    including dev — useful for fast iteration without re-downloading.
"""

from __future__ import annotations

import dagster as dg
from dagster_hf_datasets import HuggingFaceResource

from carbon_enrichment.schema import (
    DEFAULT_VALIDATION_LEVEL,
    HF_DATASET_CONFIG,
    HF_DATASET_PATH,
    VALIDATION_LEVEL_SPLITS,
)


class IngestionConfig(dg.Config):
    """Run-time config selecting which validation tier to ingest."""

    validation_level: str = DEFAULT_VALIDATION_LEVEL  # "dev" | "integration" | "auth"


@dg.asset(
    group_name="ingestion",
    io_manager_key="hf_parquet_io_manager",
    compute_kind="huggingface",
    description=(
        "Raw Carbon pretraining corpus rows (eukaryote_generator config), "
        "sliced to the requested validation tier via HF split-slicing."
    ),
)
def carbon_raw_sequences(
    context: dg.AssetExecutionContext,
    config: IngestionConfig,
    hf_resource: HuggingFaceResource,
):
    """Level 1/2/3 raw ingestion asset for the Carbon corpus.

    validation_level -> split:
      dev          -> train[:100]
      integration  -> train[:1%]
      auth         -> train[:25%]
    """
    level = config.validation_level
    if level not in VALIDATION_LEVEL_SPLITS:
        raise ValueError(
            f"Unknown validation_level={level!r}. "
            f"Expected one of {sorted(VALIDATION_LEVEL_SPLITS)}."
        )
    split = VALIDATION_LEVEL_SPLITS[level]

    context.log.info(
        f"Loading {HF_DATASET_PATH} [{HF_DATASET_CONFIG}] split={split!r} "
        f"(validation_level={level!r})"
    )

    dataset = hf_resource.load_dataset(
        path=HF_DATASET_PATH,
        config=HF_DATASET_CONFIG,
        split=split,
        streaming=False,
    )

    num_rows = hf_resource.get_num_rows(dataset)
    fingerprint = hf_resource.get_fingerprint(dataset)

    context.add_output_metadata(
        {
            "validation_level": level,
            "split": split,
            "num_rows": num_rows,
            "fingerprint": fingerprint if fingerprint else "n/a",
            "columns": dg.MetadataValue.md(
                "\n".join(f"- `{c}`" for c in dataset.column_names)
            ),
        }
    )

    return dataset