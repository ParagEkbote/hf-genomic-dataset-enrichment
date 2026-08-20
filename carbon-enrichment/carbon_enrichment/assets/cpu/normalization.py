"""
M2 — Carbon CPU normalization.

This module contains the deterministic, batch-oriented normalization kernel.

Normalization is intentionally conservative:

- strip surrounding whitespace from string fields;
- normalize nucleotide sequences to uppercase;
- normalize taxonomy component whitespace;
- normalize categorical/token fields by stripping surrounding whitespace;
- preserve all input columns;
- do not remove records;
- do not infer biological meaning;
- do not perform model inference.

The module does not materialize a Hugging Face Dataset and does not call
Dataset.map(). Streaming/orchestration is handled by streaming.py.
"""

from typing import Any


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
    """Normalize taxonomy while preserving semicolon-delimited structure."""

    if value is None:
        return None

    if not isinstance(value, str):
        return value

    parts = [
        part.strip()
        for part in value.strip().split(";")
    ]

    return ";".join(parts)


# ============================================================================
# Batch normalization
# ============================================================================


def normalize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    normalized = {column: list(values) for column, values in batch.items()}

    n = len(next(iter(batch.values()))) if batch else 0
    token_cols_present = [c for c in _TOKEN_COLUMNS if c in normalized]
    has_sequence = "sequence" in normalized
    has_taxonomy = "taxonomy" in normalized

    for i in range(n):
        for col in token_cols_present:
            normalized[col][i] = _strip_string(normalized[col][i])
        if has_sequence:
            normalized["sequence"][i] = _normalize_sequence(normalized["sequence"][i])
        if has_taxonomy:
            normalized["taxonomy"][i] = _normalize_taxonomy(normalized["taxonomy"][i])

    return normalized