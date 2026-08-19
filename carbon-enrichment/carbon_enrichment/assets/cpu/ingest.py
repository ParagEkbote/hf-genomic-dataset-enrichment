"""
M1 — Carbon raw dataset ingestion.

The asset loads a selected validation tier from the Carbon pretraining
corpus using HuggingFaceResource from dagster-hf-datasets.

Validation tiers are selected at runtime:

    dev         -> train[:100]
    integration -> train[:1%]
    auth        -> train[:25%]

The raw dataset is intentionally preserved without transformation.
Validation is performed by asset checks in `validation.py`.
"""

import dagster as dg
from dagster_hf_datasets import HuggingFaceResource

from carbon_enrichment.config import IngestionConfig
from carbon_enrichment.schema import (
    HF_DATASET_CONFIG,
    HF_DATASET_PATH,
    VALIDATION_LEVEL_SPLITS,
)


@dg.asset(
    group_name="ingestion",
    io_manager_key="hf_parquet_io_manager",
    compute_kind="huggingface",
    description=(
        "Raw Carbon pretraining corpus rows from the "
        "eukaryote_generator configuration, sliced to the requested "
        "validation tier."
    ),
)
def carbon_raw_sequences(
    context: dg.AssetExecutionContext,
    config: IngestionConfig,
    hf_resource: HuggingFaceResource,
):
    """Load the Carbon corpus at the requested validation level."""

    level = config.validation_level

    if level not in VALIDATION_LEVEL_SPLITS:
        raise ValueError(
            f"Unknown validation_level={level!r}. "
            f"Expected one of {sorted(VALIDATION_LEVEL_SPLITS)}."
        )

    split = VALIDATION_LEVEL_SPLITS[level]

    context.log.info(
        f"Loading {HF_DATASET_PATH} "
        f"[{HF_DATASET_CONFIG}] "
        f"split={split!r} "
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
            "fingerprint": fingerprint or "n/a",
            "columns": dg.MetadataValue.md(
                "\n".join(
                    f"- `{column}`"
                    for column in dataset.column_names
                )
            ),
        }
    )

    return dataset