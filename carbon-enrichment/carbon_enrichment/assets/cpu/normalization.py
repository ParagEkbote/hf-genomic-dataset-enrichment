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

VECTORIZED (design doc #24): operates on pa.RecordBatch directly via
pyarrow.compute column kernels, not a per-row Python loop over a
dict-of-lists batch. This removes the RecordBatch -> dict -> RecordBatch
boundary that streaming.py previously had to cross once per batch just to
call this function -- see streaming.py's _process_batch for the updated
call shape.

The module does not materialize a Hugging Face Dataset and does not call
Dataset.map(). Streaming/orchestration is handled by streaming.py.
"""

import pyarrow as pa
import pyarrow.compute as pc

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
# Column-level normalization kernels
# ============================================================================


def _normalize_taxonomy_column(taxonomy: pa.Array) -> pa.Array:
    """
    Strip whitespace from each semicolon-delimited component while
    preserving structure -- vectorized equivalent of
    ";".join(part.strip() for part in value.split(";")).

    Nulls propagate automatically through pc.split_pattern /
    pc.utf8_trim_whitespace / pc.binary_join without special-casing.
    """

    parts = pc.split_pattern(taxonomy, pattern=";")

    flat_values = pc.list_flatten(parts)
    trimmed_flat = pc.utf8_trim_whitespace(flat_values)

    offsets = parts.offsets

    trimmed_parts = pa.ListArray.from_arrays(
        offsets,
        trimmed_flat,
        mask=pc.is_null(taxonomy),
    )

    return pc.binary_join(trimmed_parts, ";")


# ============================================================================
# Batch normalization
# ============================================================================


def normalize_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """
    Normalize one RecordBatch. Column order and every non-normalized
    column are preserved exactly; only the token columns, `sequence`, and
    `taxonomy` are rewritten.
    """

    names = batch.schema.names

    token_cols_present = set(_TOKEN_COLUMNS) & set(names)

    columns = []

    for name in names:
        column = batch.column(name)

        if name in token_cols_present:
            column = pc.utf8_trim_whitespace(column)
        elif name == "sequence":
            column = pc.utf8_upper(pc.utf8_trim_whitespace(column))
        elif name == "taxonomy":
            column = _normalize_taxonomy_column(column)

        columns.append(column)

    return pa.RecordBatch.from_arrays(columns, names=names)
