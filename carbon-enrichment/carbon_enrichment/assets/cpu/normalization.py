"""
CPU normalization asset for the Carbon enrichment pipeline.

Purpose
-------
Create a deterministic, normalized representation of the validated Carbon
dataset without modifying the raw ingestion asset.

Normalization is intentionally conservative at this stage:

- strip surrounding whitespace from string fields;
- normalize nucleotide sequences to uppercase;
- normalize taxonomy component whitespace;
- normalize categorical/token fields by stripping surrounding whitespace;
- preserve the original raw columns;
- do not remove records;
- do not infer biological meaning;
- do not perform model inference.

The normalized asset is the input to the subsequent CPU enrichment stage.

Pipeline position
-----------------
    carbon_raw_sequences
            │
            ▼
    validation checks
            │
            ▼
    carbon_normalized_sequences
            │
            ▼
    CPU framework enrichment
"""

from __future__ import annotations

from typing import Any

import dagster as dg
from datasets import Dataset


# ============================================================================
# Columns that contain string/token values
# ============================================================================

_TOKEN_COLUMNS = (
    "begin_of_sequence",
    "end_of_sequence",
    "begin_of_gene",
    "end_of_gene",
    "gene_type",
    "species_type",
    "strand",
    "molecule_type",
    "topology",
    "record_id",
)


# ============================================================================
# Normalization helpers
# ============================================================================


def _strip_string(value: Any) -> Any:
    """Strip surrounding whitespace while preserving null values."""

    if value is None:
        return None

    if isinstance(value, str):
        return value.strip()

    return value


def _normalize_sequence(value: Any) -> Any:
    """Normalize a nucleotide sequence to stripped uppercase text."""

    if value is None:
        return None

    if isinstance(value, str):
        return value.strip().upper()

    return value


def _normalize_taxonomy(value: Any) -> Any:
    """Normalize taxonomy while preserving its semicolon-delimited structure.

    Each taxonomy component is stripped independently so that:

        'Eukaryota; Mammalia; Homo'

    becomes:

        'Eukaryota;Mammalia;Homo'

    No taxonomy levels are added, removed, reordered, or inferred.
    """

    if value is None:
        return None

    if not isinstance(value, str):
        return value

    parts = [
        part.strip()
        for part in value.strip().split(";")
    ]

    return ";".join(parts)


def _normalize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Apply deterministic normalization to a Dataset batch."""

    normalized: dict[str, list[Any]] = {}

    for column in _TOKEN_COLUMNS:
        if column in batch:
            normalized[column] = [
                _strip_string(value)
                for value in batch[column]
            ]

    if "sequence" in batch:
        normalized["sequence"] = [
            _normalize_sequence(value)
            for value in batch["sequence"]
        ]

    if "taxonomy" in batch:
        normalized["taxonomy"] = [
            _normalize_taxonomy(value)
            for value in batch["taxonomy"]
        ]

    return normalized


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    group_name="cpu_normalization",
    io_manager_key="hf_parquet_io_manager",
    compute_kind="cpu",
    description=(
        "Deterministically normalized Carbon sequences and metadata. "
        "The raw ingestion asset is preserved unchanged."
    ),
)
def carbon_normalized_sequences(
    context: dg.AssetExecutionContext,
    carbon_raw_sequences: Dataset,
) -> Dataset:
    """Normalize the validated raw Carbon dataset."""

    input_rows = carbon_raw_sequences.num_rows

    context.log.info(
        f"Normalizing {input_rows:,} Carbon rows."
    )

    normalized = carbon_raw_sequences.map(
        _normalize_batch,
        batched=True,
        batch_size=1_000,
        desc="Normalizing Carbon dataset",
    )

    output_rows = normalized.num_rows

    if output_rows != input_rows:
        raise RuntimeError(
            "Normalization changed the number of rows: "
            f"input={input_rows}, output={output_rows}"
        )

    context.add_output_metadata(
        {
            "input_rows": input_rows,
            "output_rows": output_rows,
            "rows_preserved": output_rows == input_rows,
            "columns": dg.MetadataValue.md(
                "\n".join(
                    f"- `{column}`"
                    for column in normalized.column_names
                )
            ),
            "normalization_operations": dg.MetadataValue.md(
                "\n".join(
                    [
                        "- strip surrounding whitespace from string/token fields",
                        "- uppercase nucleotide sequences",
                        "- strip whitespace around taxonomy levels",
                        "- preserve raw columns",
                        "- preserve row count",
                        "- no records filtered",
                    ]
                )
            ),
        }
    )

    return normalized