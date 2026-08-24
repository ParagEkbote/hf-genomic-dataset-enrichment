"""
M2 — Streaming validation and quality checks for the Carbon pipeline.

Validation is performed incrementally on bounded batches.

This module:

- never calls Dataset.to_pandas();
- never calls Dataset.filter();
- never calls Dataset.map();
- never materializes the complete Carbon corpus;
- performs the blocking schema check separately;
- accumulates diagnostic quality statistics across batches.

The raw taxonomy and biological sequence information are never truncated.

NOTE ON PARALLEL EXECUTION
---------------------------
`ValidationStats.update()` mutates instance state in place. When batches are
validated inside worker processes (see streaming.py), each worker must use
its own local ValidationStats instance — do not share one instance across
processes. Use `merge_validation_stats()` in the parent process to combine
the per-worker results back into a single cumulative ValidationStats.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import dagster as dg

from carbon_enrichment.schema import (
    EXPECTED_COLUMNS,
    GENE_BOUNDARY_PAIRS,
    IUPAC_NUCLEOTIDE_CHARS,
    KNOWN_TAXONOMY_ROOTS,
    REQUIRED_FIELDS,
    VALID_SEQUENCE_TOKENS,
)

# ============================================================================
# Validation statistics
# ============================================================================


@dataclass
class ValidationStats:
    """Incremental validation counters for one streaming execution."""

    rows_checked: int = 0

    # ------------------------------------------------------------------------
    # Token/category validation
    # ------------------------------------------------------------------------

    token_violations: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------------
    # Gene boundaries
    # ------------------------------------------------------------------------

    boundary_pair_distribution: dict[str, int] = field(default_factory=dict)

    mismatched_boundary_pairs: int = 0

    # ------------------------------------------------------------------------
    # Sequence validation
    # ------------------------------------------------------------------------

    empty_sequences: int = 0

    clean_acgt_only: int = 0

    invalid_alphabet_rows: int = 0

    min_sequence_length: int | None = None

    max_sequence_length: int | None = None

    total_sequence_length: int = 0

    # ------------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------------

    negative_coordinate_rows: int = 0

    start_gte_end_rows: int = 0

    # ------------------------------------------------------------------------
    # Taxonomy
    # ------------------------------------------------------------------------

    empty_taxonomy: int = 0

    unknown_root_domain_rows: int = 0

    thin_lineage_rows: int = 0

    min_taxonomy_depth: int | None = None

    max_taxonomy_depth: int | None = None

    total_taxonomy_depth: int = 0

    # ------------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------------

    null_counts_by_column: dict[str, int] = field(default_factory=dict)

    # =========================================================================
    # Update
    # =========================================================================

    def update(
        self,
        batch: Mapping[str, list[Any]],
    ) -> None:
        """Validate one bounded batch and update cumulative counters."""

        rows = _batch_length(batch)

        if rows == 0:
            return

        self.rows_checked += rows

        _validate_tokens(batch, self)
        _validate_boundaries(batch, self)
        _validate_sequences(batch, self)
        _validate_coordinates(batch, self)
        _validate_taxonomy(batch, self)
        _validate_required_fields(batch, self)


# ============================================================================
# Cross-process merge
# ============================================================================


def merge_validation_stats(
    results: list[ValidationStats],
) -> ValidationStats:
    """
    Combine per-worker ValidationStats instances into one cumulative result.

    Each worker process must accumulate into its own local ValidationStats
    (via `.update()`), since dataclass mutation does not cross process
    boundaries. This function performs the parent-process reduction step,
    equivalent in effect to having run `.update()` sequentially on every
    batch in a single process.
    """

    merged = ValidationStats()

    for r in results:
        merged.rows_checked += r.rows_checked

        merged.empty_sequences += r.empty_sequences
        merged.clean_acgt_only += r.clean_acgt_only
        merged.invalid_alphabet_rows += r.invalid_alphabet_rows
        merged.total_sequence_length += r.total_sequence_length

        merged.negative_coordinate_rows += r.negative_coordinate_rows
        merged.start_gte_end_rows += r.start_gte_end_rows

        merged.empty_taxonomy += r.empty_taxonomy
        merged.unknown_root_domain_rows += r.unknown_root_domain_rows
        merged.thin_lineage_rows += r.thin_lineage_rows
        merged.total_taxonomy_depth += r.total_taxonomy_depth

        merged.mismatched_boundary_pairs += r.mismatched_boundary_pairs

        for key, value in r.token_violations.items():
            merged.token_violations[key] = merged.token_violations.get(key, 0) + value

        for key, value in r.boundary_pair_distribution.items():
            merged.boundary_pair_distribution[key] = (
                merged.boundary_pair_distribution.get(key, 0) + value
            )

        for key, value in r.null_counts_by_column.items():
            merged.null_counts_by_column[key] = (
                merged.null_counts_by_column.get(key, 0) + value
            )

        merged.min_sequence_length = _merge_min(
            merged.min_sequence_length, r.min_sequence_length
        )
        merged.max_sequence_length = _merge_max(
            merged.max_sequence_length, r.max_sequence_length
        )

        merged.min_taxonomy_depth = _merge_min(
            merged.min_taxonomy_depth, r.min_taxonomy_depth
        )
        merged.max_taxonomy_depth = _merge_max(
            merged.max_taxonomy_depth, r.max_taxonomy_depth
        )

    return merged


def _merge_min(
    a: int | None,
    b: int | None,
) -> int | None:

    if a is None:
        return b

    if b is None:
        return a

    return min(a, b)


def _merge_max(
    a: int | None,
    b: int | None,
) -> int | None:

    if a is None:
        return b

    if b is None:
        return a

    return max(a, b)


# ============================================================================
# Schema validation
# ============================================================================


def validate_schema(
    batch: Mapping[str, Any],
) -> None:
    """
    Perform the blocking raw-schema check.

    This checks the column contract only. It is intentionally called on the
    first streamed batch rather than on every batch.
    """

    actual_columns = set(batch)
    expected_columns = set(EXPECTED_COLUMNS)

    missing = expected_columns - actual_columns
    unexpected = actual_columns - expected_columns

    if missing or unexpected:
        raise ValueError(
            "Schema mismatch: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


# ============================================================================
# Batch validation
# ============================================================================


def validate_batch(
    batch: Mapping[str, list[Any]],
    stats: ValidationStats | None = None,
) -> ValidationStats:
    """
    Validate one normalized batch and update cumulative statistics.

    Schema validation is deliberately separate and should be performed once
    on the first raw batch using validate_schema().

    When called from a worker process, pass a fresh, worker-local
    ValidationStats (or leave `stats=None` to create one) — never share a
    single instance across processes. Combine worker results afterward with
    `merge_validation_stats()`.
    """

    if stats is None:
        stats = ValidationStats()

    stats.update(batch)

    return stats


# ============================================================================
# Metadata
# ============================================================================


def validation_metadata(
    stats: ValidationStats,
) -> dict[str, Any]:
    """Convert cumulative validation counters into Dagster metadata."""

    total = stats.rows_checked

    non_empty_sequences = total - stats.empty_sequences

    mean_sequence_length = (
        stats.total_sequence_length / non_empty_sequences
        if non_empty_sequences
        else None
    )

    clean_pct = (
        100.0 * stats.clean_acgt_only / max(non_empty_sequences, 1)
        if non_empty_sequences
        else 0.0
    )

    mean_taxonomy_depth = stats.total_taxonomy_depth / total if total else None

    return {
        "rows_checked": total,
        "violations_by_column": dict(stats.token_violations),
        "boundary_pair_distribution": dict(stats.boundary_pair_distribution),
        "mismatched_pair_count": (stats.mismatched_boundary_pairs),
        "empty_sequences": (stats.empty_sequences),
        "clean_acgt_only": (stats.clean_acgt_only),
        "clean_acgt_pct": round(
            clean_pct,
            2,
        ),
        "invalid_alphabet_rows": (stats.invalid_alphabet_rows),
        "min_sequence_length": (stats.min_sequence_length),
        "max_sequence_length": (stats.max_sequence_length),
        "mean_sequence_length": (
            round(mean_sequence_length, 1) if mean_sequence_length is not None else None
        ),
        "negative_coordinate_rows": (stats.negative_coordinate_rows),
        "start_gte_end_rows": (stats.start_gte_end_rows),
        "empty_taxonomy": (stats.empty_taxonomy),
        "unknown_root_domain_rows": (stats.unknown_root_domain_rows),
        "thin_lineage_rows": (stats.thin_lineage_rows),
        "min_taxonomy_depth": (stats.min_taxonomy_depth),
        "max_taxonomy_depth": (stats.max_taxonomy_depth),
        "mean_taxonomy_depth": (
            round(mean_taxonomy_depth, 2) if mean_taxonomy_depth is not None else None
        ),
        "null_counts_by_column": dict(stats.null_counts_by_column),
        "total_null_values": sum(stats.null_counts_by_column.values()),
    }


def add_validation_metadata(
    context: dg.AssetExecutionContext,
    stats: ValidationStats,
) -> None:
    """Attach cumulative validation results to a Dagster asset."""

    context.add_output_metadata(
        {
            key: _to_metadata_value(value)
            for key, value in validation_metadata(stats).items()
        }
    )


# ============================================================================
# Token validation
# ============================================================================


def _validate_tokens(
    batch: Mapping[str, list[Any]],
    stats: ValidationStats,
) -> None:
    """
    Validate established categorical/token contracts.

    gene_type is intentionally not checked here because the complete
    six-value vocabulary has not yet been established.
    """

    for column, valid_values in VALID_SEQUENCE_TOKENS.items():
        if column not in batch:
            continue

        valid = set(valid_values)

        for value in batch[column]:
            if value not in valid:
                stats.token_violations[column] = (
                    stats.token_violations.get(
                        column,
                        0,
                    )
                    + 1
                )


# ============================================================================
# Gene boundary validation
# ============================================================================


def _validate_boundaries(
    batch: Mapping[str, list[Any]],
    stats: ValidationStats,
) -> None:

    begins = batch.get(
        "begin_of_gene",
        [],
    )

    ends = batch.get(
        "end_of_gene",
        [],
    )

    valid_pairs = set(GENE_BOUNDARY_PAIRS)

    for begin, end in zip(
        begins,
        ends,
    ):
        pair = (
            begin,
            end,
        )

        if pair not in valid_pairs:
            stats.mismatched_boundary_pairs += 1

        key = f"{begin}/{end}"

        stats.boundary_pair_distribution[key] = (
            stats.boundary_pair_distribution.get(
                key,
                0,
            )
            + 1
        )


# ============================================================================
# Sequence validation
# ============================================================================


def _validate_sequences(
    batch: Mapping[str, list[Any]],
    stats: ValidationStats,
) -> None:

    for value in batch.get(
        "sequence",
        [],
    ):
        sequence = "" if value is None else str(value)

        length = len(sequence)

        if length == 0:
            stats.empty_sequences += 1
            continue

        valid_alphabet = set(sequence).issubset(IUPAC_NUCLEOTIDE_CHARS)

        if not valid_alphabet:
            stats.invalid_alphabet_rows += 1

        if _is_clean_acgt(sequence):
            stats.clean_acgt_only += 1

        stats.total_sequence_length += length

        if stats.min_sequence_length is None or length < stats.min_sequence_length:
            stats.min_sequence_length = length

        if stats.max_sequence_length is None or length > stats.max_sequence_length:
            stats.max_sequence_length = length


def _is_clean_acgt(
    sequence: str,
) -> bool:

    return all(
        base
        in {
            "A",
            "C",
            "G",
            "T",
        }
        for base in sequence
    )


# ============================================================================
# Coordinate validation
# ============================================================================


def _validate_coordinates(
    batch: Mapping[str, list[Any]],
    stats: ValidationStats,
) -> None:

    for start, end in zip(
        batch.get("start", []),
        batch.get("end", []),
    ):
        try:
            start_value = float(start)
            end_value = float(end)

        except (
            TypeError,
            ValueError,
        ):
            stats.negative_coordinate_rows += 1

            continue

        if start_value < 0 or end_value < 0:
            stats.negative_coordinate_rows += 1

        if start_value >= end_value:
            stats.start_gte_end_rows += 1


# ============================================================================
# Taxonomy validation
# ============================================================================


def _validate_taxonomy(
    batch: Mapping[str, list[Any]],
    stats: ValidationStats,
) -> None:

    for value in batch.get(
        "taxonomy",
        [],
    ):
        taxonomy = "" if value is None else str(value)

        if not taxonomy:
            stats.empty_taxonomy += 1
            continue

        levels = [part.strip() for part in taxonomy.split(";") if part.strip()]

        if not levels:
            stats.empty_taxonomy += 1
            continue

        root = levels[0]

        if root not in KNOWN_TAXONOMY_ROOTS:
            stats.unknown_root_domain_rows += 1

        depth = len(levels)

        stats.total_taxonomy_depth += depth

        if stats.min_taxonomy_depth is None or depth < stats.min_taxonomy_depth:
            stats.min_taxonomy_depth = depth

        if stats.max_taxonomy_depth is None or depth > stats.max_taxonomy_depth:
            stats.max_taxonomy_depth = depth

        if depth < 2:
            stats.thin_lineage_rows += 1


# ============================================================================
# Required-field validation
# ============================================================================


def _validate_required_fields(
    batch: Mapping[str, list[Any]],
    stats: ValidationStats,
) -> None:

    for column in REQUIRED_FIELDS:
        if column not in batch:
            continue

        count = sum(value is None for value in batch[column])

        if count:
            stats.null_counts_by_column[column] = (
                stats.null_counts_by_column.get(
                    column,
                    0,
                )
                + count
            )


# ============================================================================
# Helpers
# ============================================================================


def _batch_length(
    batch: Mapping[str, Any],
) -> int:

    if not batch:
        return 0

    return len(next(iter(batch.values())))


def _to_metadata_value(
    value: Any,
) -> Any:

    if isinstance(
        value,
        (dict, list),
    ):
        return dg.MetadataValue.json(value)

    return value
