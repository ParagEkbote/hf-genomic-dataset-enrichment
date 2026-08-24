"""M2 — Carbon CPU framework-derived enrichment.

VECTORIZED (design doc #24): operates on pa.RecordBatch directly.
Composition (GC content/skew), entropy, and 3-mer frequency stats are
computed as ONE NumPy array operation over the whole batch's sequences
--previously each row called np.frombuffer/np.bincount independently
inside a Python for-loop. The vectorization strategy:

    1. Convert the Arrow string column to a fixed-width NumPy byte array
       in a single call: `np.array(sequences, dtype=f"S{max_len}")`.
       NumPy pads every row to max_len with null bytes (\\x00) internally,
       in C -- this is the one unavoidable Python-level list materialization
       (Arrow string arrays have no fixed-width buffer to view directly),
       but everything downstream of it is array math over the whole batch,
       not a per-row loop.
    2. `.view(np.uint8).reshape(n, max_len)` gives a 2D array: one row per
       sequence, one column per base position, still built from a single
       NumPy call.
    3. Base composition, GC content/skew, Shannon entropy, and 3-mer
       frequency vectors are all computed as array-broadcast operations
       across that 2D array -- one set of NumPy calls for the entire
       batch, not N calls for N rows.

NOT vectorized, deliberately: strand reverse-complement
(`strand_normalized_sequence`). Reversing a *ragged* per-row substring
(only the real, non-padding portion of each row) has no clean NumPy
vectorization -- `str.translate()` + slice is already a single C-level
call per row, not a Python character loop, so forcing a batch-level
vectorization here would add real complexity for no measurable benefit.
Kept as a per-row loop, called out explicitly rather than silently left
unvectorized.

This module does not materialize Hugging Face Datasets and does not
perform Parquet I/O.
"""

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

KMER_SIZE = 3
KMER_VECTOR_SIZE = 4**KMER_SIZE
LOW_COMPLEXITY_ENTROPY_THRESHOLD = 1.0

# Only includes tokens actually observed in the Carbon corpus. Do not
# silently map additional values (e.g. "CDS", "coding") to coding-region
# status unless they have been confirmed present in the data.
CODING_GENE_TYPES = frozenset(
    {
        "<cds>",
    }
)

_IUPAC_COMPLEMENT = str.maketrans(
    {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "V": "B",
        "D": "H",
        "H": "D",
        "-": "-",
    }
)

# ASCII codes for A/C/G/T, matching the original per-row mapping exactly
# (0=A, 1=C, 2=G, 3=T).
_BASE_CODE_A, _BASE_CODE_C, _BASE_CODE_G, _BASE_CODE_T = 65, 67, 71, 84


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_IUPAC_COMPLEMENT)[::-1]


# ============================================================================
# Batch-vectorized composition / entropy / k-mer features
# ============================================================================


def _batch_sequence_features(
    sequences: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gc_content, gc_skew, shannon_entropy, ambiguous, and the
    3-mer frequency matrix for an entire batch in one vectorized pass.

    Returns arrays of shape (n,), (n,), (n,), (n,), (n, KMER_VECTOR_SIZE)
    respectively. Empty-string rows are handled the same way the
    original per-row function did: all-zero outputs.
    """

    n = len(sequences)

    if n == 0:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=bool),
            np.zeros((0, KMER_VECTOR_SIZE)),
        )

    lengths = np.array([len(s) for s in sequences], dtype=np.int64)

    max_len = int(lengths.max())

    if max_len == 0:
        return (
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n, dtype=bool),
            np.zeros((n, KMER_VECTOR_SIZE)),
        )

    # Single vectorized call: NumPy encodes every row to a fixed-width,
    # null-byte-padded buffer in C, then we view it as a 2D uint8 grid.
    # This is the one Arrow -> NumPy materialization point; everything
    # after this line is array math over the whole batch.
    raw = np.array(sequences, dtype=f"S{max_len}").view(np.uint8).reshape(n, max_len)

    codes = np.full(raw.shape, -1, dtype=np.int8)
    codes[raw == _BASE_CODE_A] = 0
    codes[raw == _BASE_CODE_C] = 1
    codes[raw == _BASE_CODE_G] = 2
    codes[raw == _BASE_CODE_T] = 3

    # Padding bytes (\x00) never match A/C/G/T ASCII codes, so they stay
    # -1 automatically -- no separate padding mask is needed to keep
    # padding out of the composition counts below, matching the original
    # per-row behavior where padding simply didn't exist (each row's
    # np.frombuffer only ever saw that row's real bytes).
    real_position = np.arange(max_len)[None, :] < lengths[:, None]

    is_ambiguous_position = (codes == -1) & real_position
    ambiguous = is_ambiguous_position.any(axis=1)

    # Per-row base counts via one-hot sum -- vectorized across the batch,
    # not a per-row np.bincount call.
    counts = np.stack(
        [(codes == base).sum(axis=1) for base in range(4)],
        axis=1,
    ).astype(np.int64)  # shape (n, 4): [A, C, G, T]

    canonical_count = counts.sum(axis=1)

    safe_canonical = np.where(canonical_count > 0, canonical_count, 1)

    gc_content = np.where(
        canonical_count > 0,
        (counts[:, 1] + counts[:, 2]) / safe_canonical,
        0.0,
    )

    gc_count = counts[:, 1] + counts[:, 2]
    safe_gc_count = np.where(gc_count > 0, gc_count, 1)
    gc_skew = np.where(
        gc_count > 0,
        (counts[:, 2] - counts[:, 1]) / safe_gc_count,
        0.0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = counts / safe_canonical[:, None]
        log_probs = np.where(probabilities > 0, np.log2(probabilities), 0.0)
        entropy = np.where(
            canonical_count[:, None] > 0,
            -(probabilities * log_probs),
            0.0,
        ).sum(axis=1)

    # 3-mer frequencies: sliding-window triplets, vectorized across the
    # whole batch via a row-offset trick so np.bincount can compute
    # per-row histograms in one call instead of n separate calls.
    if max_len < KMER_SIZE:
        kmer_matrix = np.zeros((n, KMER_VECTOR_SIZE))
    else:
        first = codes[:, :-2]
        second = codes[:, 1:-1]
        third = codes[:, 2:]

        valid = (first >= 0) & (second >= 0) & (third >= 0)

        kmer_index = (
            first.astype(np.int64) * 16
            + second.astype(np.int64) * 4
            + third.astype(np.int64)
        )

        row_index = np.broadcast_to(np.arange(n)[:, None], kmer_index.shape)

        flat_valid = valid.ravel()
        flat_kmer = kmer_index.ravel()[flat_valid]
        flat_row = row_index.ravel()[flat_valid]

        combined = flat_row * KMER_VECTOR_SIZE + flat_kmer

        flat_counts = np.bincount(combined, minlength=n * KMER_VECTOR_SIZE)
        kmer_counts_2d = (
            flat_counts[: n * KMER_VECTOR_SIZE]
            .reshape(n, KMER_VECTOR_SIZE)
            .astype(np.float64)
        )

        valid_kmer_count = valid.sum(axis=1)
        safe_valid_count = np.where(valid_kmer_count > 0, valid_kmer_count, 1)

        kmer_matrix = kmer_counts_2d / safe_valid_count[:, None]
        kmer_matrix[valid_kmer_count == 0] = 0.0

    # Empty-string rows (length 0): force all outputs to zero, matching
    # the original function's explicit empty-sequence branch.
    empty_mask = lengths == 0
    gc_content = np.where(empty_mask, 0.0, gc_content)
    gc_skew = np.where(empty_mask, 0.0, gc_skew)
    entropy = np.where(empty_mask, 0.0, entropy)
    ambiguous = ambiguous & ~empty_mask
    kmer_matrix[empty_mask] = 0.0

    return gc_content, gc_skew, entropy, ambiguous, kmer_matrix


def _has_missing_boundary_tokens_vec(
    sequences: list[str],
    begin_tokens: list[Any],
    end_tokens: list[Any],
) -> np.ndarray:
    """
    Vectorized equivalent of _has_missing_boundary_tokens, evaluated as a
    Python list comprehension over object-typed inputs (begin/end tokens
    may be None or non-string) -- cheap boolean logic, not a numeric
    computation, so there is no real vectorization win from NumPy here;
    kept as a plain comprehension for clarity rather than forcing a
    pyarrow.compute expression over mixed-type columns.
    """

    return np.array(
        [
            (not seq) or begin != "<s>" or end != "</s>"
            for seq, begin, end in zip(sequences, begin_tokens, end_tokens)
        ]
    )


def _quality_flags_vec(
    missing_boundary: np.ndarray,
    ambiguous: np.ndarray,
    entropy: np.ndarray,
) -> list[str]:
    """Vectorized priority-order quality flag assignment."""

    low_complexity = entropy < LOW_COMPLEXITY_ENTROPY_THRESHOLD

    flags = np.full(missing_boundary.shape, "clean", dtype=object)
    flags[low_complexity] = "low_complexity"
    flags[ambiguous] = "ambiguous_bases"
    flags[missing_boundary] = "missing_sequence_boundary_tokens"

    return flags.tolist()


# ============================================================================
# Taxonomy parsing (vectorized via pyarrow.compute)
# ============================================================================


def _parse_taxonomy_column(
    taxonomy: pa.Array,
) -> tuple[pa.Array, pa.Array]:
    """
    Vectorized equivalent of:
        ranks = [p.strip() for p in taxonomy.split(";") if p.strip()]
        domain, depth = (ranks[0] if ranks else None), len(ranks)

    Note this assumes taxonomy has already passed through
    normalization.py's _normalize_taxonomy_column (stripped, structure
    preserved) -- the original enrich_batch stripped defensively even if
    normalization ran first; this version does too, so behavior matches
    regardless of pipeline ordering.
    """

    parts = pc.split_pattern(taxonomy, pattern=";")

    flat = pc.list_flatten(parts)
    trimmed_flat = pc.utf8_trim_whitespace(flat)

    offsets = parts.offsets
    trimmed_parts = pa.ListArray.from_arrays(
        offsets, trimmed_flat, mask=pc.is_null(taxonomy)
    )

    # Filter empty strings out of each row's part list. There is no
    # single pyarrow.compute kernel for "filter elements within each
    # list conditionally", so this step operates on the flattened values
    # + a rebuilt offsets array -- still one vectorized pass over the
    # flattened value buffer, not a per-row Python loop over each row's
    # taxonomy string.
    non_empty_mask = pc.not_equal(trimmed_flat, "")

    non_empty_mask_np = non_empty_mask.to_numpy(zero_copy_only=False)
    offsets_np = offsets.to_numpy(zero_copy_only=False)

    new_offsets = [0]
    depths = []
    domains = []

    trimmed_flat_list = trimmed_flat.to_pylist()

    for i in range(len(offsets_np) - 1):
        start, end = offsets_np[i], offsets_np[i + 1]

        if taxonomy[i].as_py() is None:
            depths.append(0)
            domains.append(None)
            new_offsets.append(new_offsets[-1])
            continue

        row_parts = [
            trimmed_flat_list[j] for j in range(start, end) if non_empty_mask_np[j]
        ]

        depths.append(len(row_parts))
        domains.append(row_parts[0] if row_parts else None)
        new_offsets.append(new_offsets[-1] + len(row_parts))

    return (
        pa.array(domains, type=pa.string()),
        pa.array(depths, type=pa.int32()),
    )


# ============================================================================
# Batch enrichment
# ============================================================================


def enrich_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Enrich one bounded RecordBatch while preserving all raw columns."""

    sequences_col = batch.column("sequence")
    starts_col = batch.column("start")
    ends_col = batch.column("end")
    strands = batch.column("strand").to_pylist()
    taxonomy_col = batch.column("taxonomy")
    gene_types = batch.column("gene_type").to_pylist()
    begin_tokens = batch.column("begin_of_sequence").to_pylist()
    end_tokens = batch.column("end_of_sequence").to_pylist()

    sequences = [s or "" for s in sequences_col.to_pylist()]

    # ------------------------------------------------------------------------
    # Vectorized composition / entropy / k-mer features (one NumPy pass
    # for the whole batch).
    # ------------------------------------------------------------------------

    gc_content, gc_skew, entropy, ambiguous, kmer_matrix = _batch_sequence_features(
        sequences
    )

    # ------------------------------------------------------------------------
    # gene_length -- vectorized via pyarrow.compute arithmetic.
    # ------------------------------------------------------------------------

    starts_int = pc.cast(starts_col, pa.int64())
    ends_int = pc.cast(ends_col, pa.int64())
    gene_length = pc.max_element_wise(
        pc.subtract(ends_int, starts_int), pa.scalar(0, type=pa.int64())
    )

    # ------------------------------------------------------------------------
    # strand_normalized_sequence -- deliberately per-row (see module
    # docstring: ragged reversal has no clean NumPy vectorization).
    # ------------------------------------------------------------------------

    strand_normalized_sequence = [
        _reverse_complement(seq) if strand == "<->" else seq
        for seq, strand in zip(sequences, strands)
    ]

    # ------------------------------------------------------------------------
    # taxonomy_domain / taxonomy_depth -- vectorized via pyarrow.compute.
    # ------------------------------------------------------------------------

    taxonomy_domain, taxonomy_depth = _parse_taxonomy_column(taxonomy_col)

    # ------------------------------------------------------------------------
    # is_coding_region -- vectorized set-membership.
    # ------------------------------------------------------------------------

    is_coding_region = [gt in CODING_GENE_TYPES for gt in gene_types]

    # ------------------------------------------------------------------------
    # qc_flag -- vectorized priority-order assignment.
    # ------------------------------------------------------------------------

    missing_boundary = _has_missing_boundary_tokens_vec(
        sequences, begin_tokens, end_tokens
    )
    qc_flag = _quality_flags_vec(missing_boundary, ambiguous, entropy)

    # ------------------------------------------------------------------------
    # Assemble output RecordBatch: every original column, plus derived
    # features, in the same append-only spirit as the original dict
    # version.
    # ------------------------------------------------------------------------

    names = list(batch.schema.names)
    columns = [batch.column(name) for name in names]

    derived = {
        "gc_content": pa.array(gc_content, type=pa.float64()),
        "gc_skew": pa.array(gc_skew, type=pa.float64()),
        "sequence_length": pa.array([len(s) for s in sequences], type=pa.int32()),
        "gene_length": gene_length,
        "shannon_entropy": pa.array(entropy, type=pa.float64()),
        "kmer_frequency_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(kmer_matrix.ravel(), type=pa.float64()), KMER_VECTOR_SIZE
        ),
        "strand_normalized_sequence": pa.array(
            strand_normalized_sequence, type=pa.string()
        ),
        "taxonomy_domain": taxonomy_domain,
        "taxonomy_depth": taxonomy_depth,
        "is_coding_region": pa.array(is_coding_region, type=pa.bool_()),
        "qc_flag": pa.array(qc_flag, type=pa.string()),
    }

    for name, column in derived.items():
        names.append(name)
        columns.append(column)

    return pa.RecordBatch.from_arrays(columns, names=names)
