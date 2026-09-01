from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import math
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv as pa_csv
from numba import njit, prange

from carbon_enrichment.resources.clickhouse import ClickHouseResource


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

A, C, G, T = 65, 67, 71, 84

# Internal processing batch size. ClickHouse streams ArrowStream
# batches at its own internal block size; we re-chunk to this size so
# downstream Numba kernels see consistent batch shapes.
BATCH_SIZE = 50_000

SEQUENCE_COLUMN = "sequence"

TAXONOMY_RANKS = [
    "domain",
    "kingdom",
    "subkingdom",
    "phylum",
    "subphylum",
    "class",
    "subclass",
    "order",
    "suborder",
    "family",
    "genus",
]

TAXONOMY_CLASS_INDEX = (
    TAXONOMY_RANKS.index("class") + 1
)

# Histogram configuration matching the previous reports.
LENGTH_LOG_MIN = 1.5
LENGTH_LOG_MAX = 5.0
LENGTH_LOG_BUCKETS = 40

GC_MIN = 0.0
GC_MAX = 1.0
GC_BUCKETS = 40


# ----------------------------------------------------------------------
# Taxonomy helpers
# ----------------------------------------------------------------------


def taxonomy_rank_expr(
    rank: str,
    column: str = "taxonomy",
) -> str:
    """
    SQL expression extracting one rank level from a ';'-delimited
    lineage string.

    ClickHouse's splitByChar/array indices are 1-based, same as DuckDB.
    """
    if rank not in TAXONOMY_RANKS:
        raise ValueError(
            f"Unknown rank {rank!r}; "
            f"expected one of {TAXONOMY_RANKS}"
        )

    idx = TAXONOMY_RANKS.index(rank) + 1

    return f"splitByChar(';', \"{column}\")[{idx}]"


def taxonomy_rank_cardinality(
    db: ClickHouseResource,
    source_expr: str,
    column: str = "taxonomy",
):
    """
    Compute taxonomy cardinality directly against the source Parquet
    corpus using clickhouse-local.

    No intermediate materialization is required.
    """
    select_clauses = ", ".join(
        f"uniqExact({taxonomy_rank_expr(rank, column)}) "
        f"AS {rank}_distinct_count"
        for rank in TAXONOMY_RANKS
    )

    return db.query_arrow(
        f"""
        SELECT
            {select_clauses}
        FROM {source_expr}
        """,
    )


# ----------------------------------------------------------------------
# Running statistics
# ----------------------------------------------------------------------


@dataclass
class RunningStats:
    """
    Numerically stable aggregate state.

    Maintains:
      - count
      - mean
      - M2

    M2 allows exact merging of batch-level statistics without retaining
    all observations.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def merge(
        self,
        batch_count: int,
        batch_sum: float,
        batch_sum_sq: float,
    ) -> None:
        if batch_count <= 0:
            return

        batch_mean = (
            batch_sum / batch_count
        )

        batch_m2 = (
            batch_sum_sq
            - batch_count * batch_mean * batch_mean
        )

        # Protect against tiny floating-point negative values.
        batch_m2 = max(
            0.0,
            batch_m2,
        )

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = (
            self.count + batch_count
        )

        delta = (
            batch_mean - self.mean
        )

        self.m2 += (
            batch_m2
            + (
                delta
                * delta
                * self.count
                * batch_count
                / total_count
            )
        )

        self.mean += (
            delta
            * batch_count
            / total_count
        )

        self.count = total_count

    @property
    def stddev(self) -> float:
        """
        Sample standard deviation.

        This matches ClickHouse stddevSamp semantics.
        """
        if self.count < 2:
            return 0.0

        return math.sqrt(
            max(
                0.0,
                self.m2 / (self.count - 1),
            )
        )


@dataclass
class GroupedStats:
    """
    Aggregate statistics for a categorical grouping variable.

    Groups are maintained as Python objects, but aggregation over rows
    is performed with NumPy bincount rather than a Python row loop.
    """

    groups: dict[str, RunningStats] = field(
        default_factory=dict
    )

    def update(
        self,
        group_array: pa.Array,
        values: dict[str, np.ndarray],
    ) -> None:
        encoded = pc.dictionary_encode(
            group_array
        )

        dictionary = encoded.dictionary
        codes = np.asarray(
            encoded.indices,
            dtype=np.int64,
        )

        n_groups = len(dictionary)

        if n_groups == 0:
            return

        valid = codes >= 0

        if not np.any(valid):
            return

        valid_codes = codes[valid]

        counts = np.bincount(
            valid_codes,
            minlength=n_groups,
        )

        sums = {
            name: np.bincount(
                valid_codes,
                weights=np.asarray(
                    value,
                    dtype=np.float64,
                )[valid],
                minlength=n_groups,
            )
            for name, value in values.items()
        }

        sums_sq = {
            name: np.bincount(
                valid_codes,
                weights=(
                    np.asarray(
                        value,
                        dtype=np.float64,
                    )[valid]
                    ** 2
                ),
                minlength=n_groups,
            )
            for name, value in values.items()
        }

        labels = dictionary.to_pylist()

        for group_idx, label in enumerate(labels):
            count = int(
                counts[group_idx]
            )

            if count == 0:
                continue

            key = str(label)

            stats = self.groups.setdefault(
                key,
                RunningStats(),
            )

            # This object stores one metric per group, so callers using
            # multiple metrics should maintain separate GroupedStats
            # instances.
            stats.merge(
                count,
                float(
                    sums["_value"][group_idx]
                ),
                float(
                    sums_sq["_value"][group_idx]
                ),
            )


# ----------------------------------------------------------------------
# Fused Numba kernel
#
# This computes only statistics actually required by Phase 2.
#
# IMPORTANT:
# The previous implementation created:
#
#     (250_000, 64) int64
#
# k-mer matrices and then converted every row to a Python list before
# writing Parquet.
#
# Phase 2 distribution analysis never consumed those per-row vectors,
# so they are deliberately removed from this analytical path.
# ----------------------------------------------------------------------


@njit(
    cache=True,
    fastmath=True,
)
def _base_index(
    byte: np.uint8,
) -> int:
    if byte == A:
        return 0

    if byte == C:
        return 1

    if byte == G:
        return 2

    if byte == T:
        return 3

    return -1


@njit(
    cache=True,
    fastmath=True,
)
def _fused_sequence_stats(
    seq: np.ndarray,
) -> tuple:
    """
    Compute all per-sequence statistics required by Phase 2.

    Returns
    -------
    tuple
        gc_content
        gc_skew
        purine_pyrimidine_skew
        max_homopolymer
        cpg_odds_ratio
        fickett_proxy_variance
        stop_frame0
        stop_frame1
        stop_frame2
        tm_estimate
    """
    n = seq.shape[0]

    counts = np.zeros(
        4,
        dtype=np.int64,
    )

    dinuc_counts = np.zeros(
        (4, 4),
        dtype=np.int64,
    )

    codon_pos_counts = np.zeros(
        (3, 4),
        dtype=np.int64,
    )

    stop_frame_counts = np.zeros(
        3,
        dtype=np.int64,
    )

    if n == 0:
        return (
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0,
            0,
            0,
            0.0,
        )

    max_run = 1
    cur_run = 1
    prev_idx = -1

    b0 = -1
    b1 = -1
    b2 = -1

    valid_bases = 0

    for i in range(n):
        idx = _base_index(
            seq[i]
        )

        if idx == -1:
            cur_run = 1
            prev_idx = -1
            b0 = -1
            b1 = -1
            b2 = -1
            continue

        counts[idx] += 1
        valid_bases += 1

        if idx == prev_idx:
            cur_run += 1

            if cur_run > max_run:
                max_run = cur_run

        else:
            cur_run = 1

        prev_idx = idx

        b0, b1, b2 = b1, b2, idx

        if (
            b0 != -1
            and b1 != -1
        ):
            dinuc_counts[
                b0,
                b1,
            ] += 1

        if (
            b0 != -1
            and b1 != -1
            and b2 != -1
        ):
            frame = (
                (i - 2) % 3
            )

            if (
                (
                    b0 == T
                    and b1 == A
                    and b2 == A
                )
                or (
                    b0 == T
                    and b1 == A
                    and b2 == G
                )
                or (
                    b0 == T
                    and b1 == G
                    and b2 == A
                )
            ):
                stop_frame_counts[
                    frame
                ] += 1

        codon_pos_counts[
            i % 3,
            idx,
        ] += 1

    if valid_bases == 0:
        return (
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0,
            0,
            0,
            0.0,
        )

    gc = (
        counts[1]
        + counts[2]
    )

    gc_content = (
        gc / valid_bases
    )

    gc_skew = (
        (counts[2] - counts[1]) / gc
        if gc > 0
        else 0.0
    )

    purine = (
        counts[0]
        + counts[2]
    )

    pyrimidine = (
        counts[1]
        + counts[3]
    )

    purine_pyrimidine_skew = (
        purine / pyrimidine
        if pyrimidine > 0
        else np.inf
    )

    total_dinuc = (
        valid_bases - 1
        if valid_bases > 1
        else 1
    )

    p_c = (
        counts[1] / valid_bases
    )

    p_g = (
        counts[2] / valid_bases
    )

    p_cg = (
        dinuc_counts[1, 2]
        / total_dinuc
    )

    cpg_odds_ratio = (
        p_cg / (p_c * p_g)
        if p_c > 0 and p_g > 0
        else 0.0
    )

    codon_freqs = np.zeros(
        (3, 4),
        dtype=np.float64,
    )

    for pos in range(3):
        pos_total = (
            codon_pos_counts[pos].sum()
        )

        if pos_total > 0:
            for base in range(4):
                codon_freqs[
                    pos,
                    base,
                ] = (
                    codon_pos_counts[
                        pos,
                        base,
                    ]
                    / pos_total
                )

    fickett_proxy_variance = (
        np.var(codon_freqs)
    )

    tm_estimate = (
        64.9
        + 41.0
        * (gc - 16.4)
        / valid_bases
    )

    return (
        gc_content,
        gc_skew,
        purine_pyrimidine_skew,
        max_run,
        cpg_odds_ratio,
        fickett_proxy_variance,
        stop_frame_counts[0],
        stop_frame_counts[1],
        stop_frame_counts[2],
        tm_estimate,
    )


@njit(
    cache=True,
    parallel=True,
)
def compute_batch_stats(
    flat_seq_bytes: np.ndarray,
    offsets: np.ndarray,
) -> tuple:
    """
    Compute derived sequence statistics for one Arrow batch.
    """
    n_seqs = (
        offsets.shape[0] - 1
    )

    gc_content_out = np.zeros(n_seqs, dtype=np.float64)
    gc_skew_out = np.zeros(n_seqs, dtype=np.float64)
    pu_py_skew_out = np.zeros(n_seqs, dtype=np.float64)
    max_homopolymer_out = np.zeros(n_seqs, dtype=np.int64)
    cpg_odds_out = np.zeros(n_seqs, dtype=np.float64)
    fickett_proxy_out = np.zeros(n_seqs, dtype=np.float64)
    stop_frame0_out = np.zeros(n_seqs, dtype=np.int64)
    stop_frame1_out = np.zeros(n_seqs, dtype=np.int64)
    stop_frame2_out = np.zeros(n_seqs, dtype=np.int64)
    tm_estimate_out = np.zeros(n_seqs, dtype=np.float64)

    for i in prange(n_seqs):
        start = offsets[i]
        end = offsets[i + 1]

        seq = flat_seq_bytes[start:end]

        (
            gc_content_out[i],
            gc_skew_out[i],
            pu_py_skew_out[i],
            max_homopolymer_out[i],
            cpg_odds_out[i],
            fickett_proxy_out[i],
            stop_frame0_out[i],
            stop_frame1_out[i],
            stop_frame2_out[i],
            tm_estimate_out[i],
        ) = _fused_sequence_stats(seq)

    return (
        gc_content_out,
        gc_skew_out,
        pu_py_skew_out,
        max_homopolymer_out,
        cpg_odds_out,
        fickett_proxy_out,
        stop_frame0_out,
        stop_frame1_out,
        stop_frame2_out,
        tm_estimate_out,
    )


# ----------------------------------------------------------------------
# Arrow sequence buffer
# ----------------------------------------------------------------------


def _sequences_to_flat_buffer(
    sequences: pa.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expose Arrow's string buffers directly.

    No Python string is created for each sequence.
    """
    if sequences.null_count:
        raise ValueError(
            "Sequence column contains NULL values."
        )

    buffers = sequences.buffers()

    if len(buffers) < 3:
        raise ValueError(
            "Unexpected Arrow string buffer layout."
        )

    offset_buffer = buffers[1]
    data_buffer = buffers[2]

    if offset_buffer is None:
        raise ValueError(
            "Sequence Arrow array has no offset buffer."
        )

    if data_buffer is None:
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        flat_bytes = np.empty(0, dtype=np.uint8)
        return (flat_bytes, offsets)

    if pa.types.is_large_string(sequences.type):
        offset_dtype = np.int64
    else:
        offset_dtype = np.int32

    raw_offsets = np.frombuffer(
        offset_buffer,
        dtype=offset_dtype,
    )

    start = sequences.offset
    stop = start + len(sequences) + 1

    offsets = raw_offsets[start:stop].astype(
        np.int64,
        copy=False,
    )

    flat_bytes = np.frombuffer(data_buffer, dtype=np.uint8)

    return (flat_bytes, offsets)


# ----------------------------------------------------------------------
# Histogram helpers
# ----------------------------------------------------------------------


def _width_bucket(
    values: np.ndarray,
    lower: float,
    upper: float,
    buckets: int,
) -> np.ndarray:
    """
    Reproduce width_bucket-style numbering:

        0              below lower
        1..buckets     normal buckets
        buckets + 1    >= upper
    """
    result = np.zeros(values.shape, dtype=np.int32)

    finite = np.isfinite(values)

    below = finite & (values < lower)
    above = finite & (values >= upper)
    inside = finite & ~below & ~above

    result[below] = 0
    result[above] = buckets + 1

    if np.any(inside):
        scaled = (
            (values[inside] - lower)
            / (upper - lower)
            * buckets
        )

        result[inside] = (
            np.floor(scaled).astype(np.int32) + 1
        )

    result[~finite] = 0

    return result


# ----------------------------------------------------------------------
# Phase 2 aggregate state
# ----------------------------------------------------------------------


@dataclass
class Phase2Aggregates:
    """
    Complete in-memory aggregate state for Phase 2.

    Only aggregate state is retained between batches.
    No derived per-row dataset is materialized.
    """

    total_rows: int = 0

    length_gc_hist: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (
                LENGTH_LOG_BUCKETS + 2,
                GC_BUCKETS + 2,
            ),
            dtype=np.int64,
        )
    )

    coding_gc_skew: GroupedStats = field(default_factory=GroupedStats)
    coding_stop_frame0: GroupedStats = field(default_factory=GroupedStats)
    coding_stop_frame1: GroupedStats = field(default_factory=GroupedStats)
    coding_stop_frame2: GroupedStats = field(default_factory=GroupedStats)
    coding_fickett: GroupedStats = field(default_factory=GroupedStats)
    taxonomy_gc_skew: GroupedStats = field(default_factory=GroupedStats)

    homopolymer_exceeds_length: int = 0
    gc_content_out_of_range: int = 0
    tm_estimate_implausible: int = 0
    cpg_odds_negative: int = 0

    def update(
        self,
        lengths: np.ndarray,
        stats: tuple,
        coding_group: pa.Array,
        taxonomy_class: pa.Array,
    ) -> None:
        (
            gc_content,
            gc_skew,
            pu_py_skew,
            max_homopolymer,
            cpg_odds,
            fickett_proxy,
            stop_f0,
            stop_f1,
            stop_f2,
            tm_estimate,
        ) = stats

        del pu_py_skew

        batch_rows = len(lengths)
        self.total_rows += batch_rows

        positive_lengths = lengths > 0

        length_log = np.zeros(lengths.shape, dtype=np.float64)

        length_log[positive_lengths] = np.log10(
            lengths[positive_lengths].astype(np.float64)
        )

        length_bucket = _width_bucket(
            length_log,
            LENGTH_LOG_MIN,
            LENGTH_LOG_MAX,
            LENGTH_LOG_BUCKETS,
        )

        gc_bucket = _width_bucket(
            gc_content,
            GC_MIN,
            GC_MAX,
            GC_BUCKETS,
        )

        flattened = (
            length_bucket * (GC_BUCKETS + 2) + gc_bucket
        )

        counts = np.bincount(
            flattened,
            minlength=(
                (LENGTH_LOG_BUCKETS + 2) * (GC_BUCKETS + 2)
            ),
        )

        self.length_gc_hist += counts.reshape(self.length_gc_hist.shape)

        self._update_grouped(self.coding_gc_skew, coding_group, gc_skew)

        self._update_grouped(
            self.coding_stop_frame0,
            coding_group,
            stop_f0.astype(np.float64),
        )

        self._update_grouped(
            self.coding_stop_frame1,
            coding_group,
            stop_f1.astype(np.float64),
        )

        self._update_grouped(
            self.coding_stop_frame2,
            coding_group,
            stop_f2.astype(np.float64),
        )

        self._update_grouped(self.coding_fickett, coding_group, fickett_proxy)

        self._update_grouped(self.taxonomy_gc_skew, taxonomy_class, gc_skew)

        self.homopolymer_exceeds_length += int(
            np.count_nonzero(max_homopolymer > lengths)
        )

        self.gc_content_out_of_range += int(
            np.count_nonzero((gc_content < 0.0) | (gc_content > 1.0))
        )

        self.tm_estimate_implausible += int(
            np.count_nonzero((tm_estimate < 0.0) | (tm_estimate > 120.0))
        )

        self.cpg_odds_negative += int(np.count_nonzero(cpg_odds < 0.0))

    @staticmethod
    def _update_grouped(
        target: GroupedStats,
        group_array: pa.Array,
        values: np.ndarray,
    ) -> None:
        target.update(group_array, {"_value": values})


# ----------------------------------------------------------------------
# ClickHouse source scan
# ----------------------------------------------------------------------


def _count_source_rows(
    db: ClickHouseResource,
    source_expr: str,
) -> int:
    """
    Obtain total source rows through ClickHouse.

    count(sequence) is checked separately so a NULL sequence cannot
    silently corrupt the Arrow buffer processing.
    """
    result = db.query_arrow(
        f"""
        SELECT
            count(*) AS total_rows,
            count(sequence) AS sequence_rows
        FROM {source_expr}
        """,
    )

    total_rows = int(result.column("total_rows")[0].as_py())
    sequence_rows = int(result.column("sequence_rows")[0].as_py())

    if total_rows != sequence_rows:
        raise ValueError(
            "Source corpus contains NULL sequence values: "
            f"total_rows={total_rows}, "
            f"sequence_rows={sequence_rows}"
        )

    return total_rows


def _source_batches(
    db: ClickHouseResource,
    source_expr: str,
):
    """
    Build a streaming ClickHouse -> Arrow batch generator.

    Only the columns needed by Phase 2 are projected. Uses ArrowStream
    output so rows never fully materialize on the Python side at once.
    """
    query = f"""
    SELECT
        "{SEQUENCE_COLUMN}" AS sequence,

        coalesce(
            toString(is_coding_region),
            '__NULL__'
        ) AS coding_group,

        coalesce(
            {taxonomy_rank_expr("class")},
            '__NULL__'
        ) AS taxonomy_class

    FROM {source_expr}
    """

    return db.stream_arrow_batches(query)


def _rechunk_batches(batches, batch_size: int):
    """
    Re-chunk an iterator of Arrow RecordBatches into batches of exactly
    `batch_size` rows (except possibly the last), matching the fixed
    batch size the DuckDB `to_arrow_reader(BATCH_SIZE)` reader used to
    provide.
    """
    pending: list[pa.RecordBatch] = []
    pending_rows = 0

    for batch in batches:
        pending.append(batch)
        pending_rows += batch.num_rows

        if pending_rows < batch_size:
            continue

        table = pa.Table.from_batches(pending).combine_chunks()
        out_batches = table.to_batches(max_chunksize=batch_size)

        # Emit every full-size batch except a possible short remainder,
        # which is carried forward into `pending`.
        *full, remainder = out_batches

        for out_batch in full:
            yield out_batch

        if remainder.num_rows == batch_size:
            yield remainder
            pending = []
            pending_rows = 0
        else:
            pending = [remainder]
            pending_rows = remainder.num_rows

    if pending_rows > 0:
        table = pa.Table.from_batches(pending).combine_chunks()
        for out_batch in table.to_batches(max_chunksize=batch_size):
            yield out_batch


# ----------------------------------------------------------------------
# Streaming Phase 2 analysis
# ----------------------------------------------------------------------


def analyze_source(
    source: str | Path,
) -> Phase2Aggregates:
    """
    Analyze the CPU-enriched corpus without materializing another
    Parquet dataset.

    Execution model:

        Parquet
          |
        clickhouse-local (file table function, ArrowStream)
          |
        Arrow RecordBatch
          |
        Numba
          |
        aggregate state
          |
        next batch

    Returns
    -------
    Phase2Aggregates
        Complete Phase 2 aggregate state.
    """
    source_text = str(source)

    db = ClickHouseResource()

    try:
        if source_text.startswith(
            "https://huggingface.co/datasets/"
        ):
            db.register_hf_dataset(
                "phase2_source",
                source_text,
            )
            source_expr = db.source_expr("phase2_source")
            source_label = source_text
        else:
            source_path = Path(source_text).expanduser().resolve()

            if not source_path.exists():
                raise FileNotFoundError(
                    f"Source directory does not exist: {source_path}"
                )

            if not source_path.is_dir():
                raise NotADirectoryError(
                    f"Source path is not a directory: {source_path}"
                )

            source_glob = str(source_path / "*.parquet")
            source_expr = f"file('{source_glob.replace("'", "''")}', Parquet)"
            source_label = str(source_path)

        total_rows = _count_source_rows(db, source_expr)


        n_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE


        aggregates = Phase2Aggregates()

        numba_rows = 0
        numba_start = time.perf_counter()
        last_numba_report = numba_start

        raw_batches = _source_batches(db, source_expr)
        rechunked = _rechunk_batches(raw_batches, BATCH_SIZE)

        for batch in rechunked:
            flat_bytes, offsets = _sequences_to_flat_buffer(
            batch.column("sequence")
            )

            lengths = np.diff(offsets).astype(np.int64, copy=False)

            compute_start = time.perf_counter()

            stats = compute_batch_stats(flat_bytes, offsets)

            compute_elapsed = time.perf_counter() - compute_start
            numba_rows += batch.num_rows

            now = time.perf_counter()

            if now - last_numba_report >= 10.0:
                elapsed = now - numba_start

                print(
                    f"Numba: {numba_rows:,} rows | "
                    f"avg={numba_rows / elapsed:,.0f} rows/s | "
                    f"batch={batch.num_rows / compute_elapsed:,.0f} rows/s",
                    file=sys.stderr,
                    flush=True,
                )

            last_numba_report = now

            aggregates.update(
                lengths=lengths,
                stats=stats,
                coding_group=batch.column("coding_group"),
                taxonomy_class=batch.column("taxonomy_class"),
            )


        return aggregates

    finally:
        db.close()


# ----------------------------------------------------------------------
# Report formatting
# ----------------------------------------------------------------------


def _grouped_report(
    grouped: GroupedStats,
    metric_name: str,
) -> list[dict[str, object]]:
    """
    Convert grouped RunningStats into a compact report.
    """
    rows: list[dict[str, object]] = []

    for group, stats in sorted(
        grouped.groups.items(),
        key=lambda item: (-item[1].count, item[0]),
    ):
        rows.append(
            {
                "group": group,
                metric_name: stats.mean,
                f"{metric_name}_stddev": stats.stddev,
                "n": stats.count,
            }
        )

    return rows


def _length_gc_report(
    aggregates: Phase2Aggregates,
) -> list[dict[str, int]]:
    """
    Return non-empty length x GC histogram cells.
    """
    rows: list[dict[str, int]] = []

    for length_bucket in range(aggregates.length_gc_hist.shape[0]):
        for gc_bucket in range(aggregates.length_gc_hist.shape[1]):
            n = int(aggregates.length_gc_hist[length_bucket, gc_bucket])

            if n == 0:
                continue

            rows.append(
                {
                    "length_log_bucket": length_bucket,
                    "gc_bucket": gc_bucket,
                    "n": n,
                }
            )

    return rows


def _range_sanity_report(
    aggregates: Phase2Aggregates,
) -> dict[str, int]:
    return {
        "homopolymer_exceeds_length": aggregates.homopolymer_exceeds_length,
        "gc_content_out_of_range": aggregates.gc_content_out_of_range,
        "tm_estimate_implausible": aggregates.tm_estimate_implausible,
        "cpg_odds_negative": aggregates.cpg_odds_negative,
        "total_rows": aggregates.total_rows,
    }


def _write_csv(data: list[dict], out_path: Path) -> None:
    """Helper to dump list of dicts to a CSV file."""
    if not data:
        return
    keys = data[0].keys()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


# ----------------------------------------------------------------------
# Phase 2 orchestration
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: analyze deterministic sequence "
            "statistics from the CPU-enriched corpus."
        )
    )

    parser.add_argument(
        "--source",
        required=False,
        help=(
            "Source dataset: either a local directory containing Parquet "
            "files or a Hugging Face Parquet dataset URL/glob."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Deprecated alias for --source when using a local Parquet directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save the generated CSV reports.",
    )

    args = parser.parse_args()

    source = args.source
    if source is None and args.source_dir is not None:
        source = str(args.source_dir)

    if source is None:
        parser.error("one of --source or --source-dir is required")

    if args.source is not None and args.source_dir is not None:
        parser.error("use only one of --source or --source-dir")

    aggregates = analyze_source(source)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 2 analysis complete.")
    print(f"Total rows processed: {aggregates.total_rows}")
    print(f"Writing CSV reports to: {out_dir.resolve()}")

    _write_csv(_length_gc_report(aggregates), out_dir / "length_gc_distribution.csv")
    _write_csv(_grouped_report(aggregates.coding_gc_skew, "mean_gc_skew"), out_dir / "gc_skew_vs_is_coding_region.csv")
    
    stop_codon_data = [
        {
            "group": group,
            "mean_stops_frame0": aggregates.coding_stop_frame0.groups[group].mean,
            "mean_stops_frame1": aggregates.coding_stop_frame1.groups[group].mean,
            "mean_stops_frame2": aggregates.coding_stop_frame2.groups[group].mean,
            "n": aggregates.coding_stop_frame0.groups[group].count,
        }
        for group in sorted(aggregates.coding_stop_frame0.groups)
    ]
    _write_csv(stop_codon_data, out_dir / "stop_codon_validation.csv")

    _write_csv(_grouped_report(aggregates.coding_fickett, "mean_variance"), out_dir / "fickett_proxy_validation.csv")
    _write_csv(_grouped_report(aggregates.taxonomy_gc_skew, "mean_gc_skew"), out_dir / "gc_skew_vs_taxonomy_class.csv")
    _write_csv([_range_sanity_report(aggregates)], out_dir / "range_sanity_report.csv")

    db = ClickHouseResource()
    try:
        source_text = str(source)

        if source_text.startswith(
            "https://huggingface.co/datasets/"
        ):
            db.register_hf_dataset("phase2_source", source_text)
            source_expr = db.source_expr("phase2_source")
        else:
            source_path = Path(source_text).expanduser().resolve()
            source_glob = str(source_path / "*.parquet")
            source_expr = (
                f"file('{source_glob.replace(chr(39), chr(39) + chr(39))}', Parquet)"
            )

        cardinality_table = taxonomy_rank_cardinality(db, source_expr)
        pa_csv.write_csv(cardinality_table, out_dir / "taxonomy_rank_cardinality.csv")
        
    finally:
        db.close()
        
    print("All reports saved successfully.")


if __name__ == "__main__":
    main()