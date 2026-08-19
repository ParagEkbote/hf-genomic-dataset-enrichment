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


def normalize_batch(
    batch: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Apply deterministic normalization to one bounded batch.

    All input columns are preserved. Only the explicitly defined string,
    sequence, and taxonomy columns are transformed.

    Parameters
    ----------
    batch:
        Column-oriented batch represented as ``dict[str, list[Any]]``.

    Returns
    -------
    dict[str, list[Any]]
        Normalized batch with the same number of rows.
    """

    # Preserve every input column first. This is important because the
    # original implementation only returned normalized columns, which would
    # discard untouched numeric/metadata columns when used outside Dataset.map.
    normalized = {
        column: list(values)
        for column, values in batch.items()
    }

    for column in _TOKEN_COLUMNS:
        if column in normalized:
            normalized[column] = [
                _strip_string(value)
                for value in normalized[column]
            ]

    if "sequence" in normalized:
        normalized["sequence"] = [
            _normalize_sequence(value)
            for value in normalized["sequence"]
        ]

    if "taxonomy" in normalized:
        normalized["taxonomy"] = [
            _normalize_taxonomy(value)
            for value in normalized["taxonomy"]
        ]

    return normalized