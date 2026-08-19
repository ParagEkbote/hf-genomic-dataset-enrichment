"""
M2 — Carbon raw dataset streaming.

Creates a lazy Hugging Face IterableDataset for the selected Carbon
pretraining corpus validation tier.

The stream is consumed by the streaming CPU pipeline and is never
materialized as an intermediate Hugging Face Dataset.
"""

from datasets import IterableDataset
from dagster_hf_datasets import HuggingFaceResource

from carbon_enrichment.config import CarbonPipelineConfig
from carbon_enrichment.schema import (
    HF_DATASET_CONFIG,
    HF_DATASET_PATH,
    VALIDATION_LEVEL_SPLITS,
)


def create_carbon_stream(
    config: CarbonPipelineConfig,
    hf_resource: HuggingFaceResource,
) -> IterableDataset:
    """
    Create a lazy streaming view of the Carbon corpus.

    Streaming is mandatory. The returned IterableDataset is consumed
    incrementally by the CPU streaming processor.
    """

    level = config.validation_level

    if level not in VALIDATION_LEVEL_SPLITS:
        raise ValueError(
            f"Unknown validation_level={level!r}. "
            f"Expected one of {sorted(VALIDATION_LEVEL_SPLITS)}."
        )

    if not config.streaming:
        raise ValueError(
            "The Carbon CPU pipeline requires streaming=True. "
            "Materialized Dataset ingestion is not supported."
        )

    split = VALIDATION_LEVEL_SPLITS[level]

    return hf_resource.load_dataset(
        path=HF_DATASET_PATH,
        config=HF_DATASET_CONFIG,
        split=split,
        streaming=True,
    )