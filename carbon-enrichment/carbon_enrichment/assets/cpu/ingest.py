"""
M2 — Carbon raw dataset streaming.

Creates a lazy Hugging Face IterableDataset for the selected Carbon
pretraining corpus validation tier.

The stream is consumed by the streaming CPU pipeline and is never
materialized as an intermediate Hugging Face Dataset.
"""

from dagster_hf_datasets import HuggingFaceResource
from datasets import IterableDataset

from carbon_enrichment.config import CarbonPipelineConfig
from carbon_enrichment.schema import (
    HF_DATASET_CONFIG,
    HF_DATASET_PATH,
    HF_DATASET_SPLIT,
    VALIDATION_LEVEL_ROWS,
)


def create_carbon_stream(
    config: CarbonPipelineConfig,
    hf_resource: HuggingFaceResource,
) -> IterableDataset:
    """
    Create a lazy streaming view of the Carbon corpus.

    Streaming is mandatory. The returned IterableDataset is consumed
    incrementally by the CPU streaming processor.

    Split selection and stream truncation are handled as two separate
    steps: the named HF split ("train") is always opened in full as an
    IterableDataset, then `.take(n)` truncates it to the row count for
    the requested validation tier. Split-slicing syntax (e.g.
    "train[:100]") is not supported when streaming=True, since an
    IterableDataset has no length to slice against.
    """

    level = config.validation_level

    if level not in VALIDATION_LEVEL_ROWS:
        raise ValueError(
            f"Unknown or unmeasured validation_level={level!r}. "
            f"Expected one of {sorted(VALIDATION_LEVEL_ROWS)}."
        )

    if not config.streaming:
        raise ValueError(
            "The Carbon CPU pipeline requires streaming=True. "
            "Materialized Dataset ingestion is not supported."
        )

    dataset = hf_resource.load_dataset(
        path=HF_DATASET_PATH,
        config=HF_DATASET_CONFIG,
        split=HF_DATASET_SPLIT,
        streaming=True,
    )

    row_limit = VALIDATION_LEVEL_ROWS[level]

    return dataset.take(row_limit)
