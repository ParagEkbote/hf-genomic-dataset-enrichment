"""
M3.5-pre — Tokenization and <dna> tagging for the Carbon GPU pipeline.

Dedicated, upfront stage between CPU-enriched Parquet and GPU enrichment
(design doc #14.5). Produces its own checkpointed/sharded output (token
IDs + record_id) so a tokenization bug or schema change never forces
re-running CPU enrichment, and a GPU-stage bug never forces re-tokenizing.

Pipeline placement:

    CPU-enriched Parquet (carbon_cpu_enriched_sequences)
            │
            ▼
    tokenize_and_tag  (this module)
      ├── wrap: f"<dna>{seq}</dna>"
      ├── filter to canonical uppercase ACGT (else -> <oov>)
      ├── truncate_to_6mer (trim to a multiple of 6 before tagging)
      ├── tokenize once with the Carbon hybrid 6-mer tokenizer
      └── write token-ID shards + record_id, checkpointed
            │
            ▼
    GPU enrichment (single forward pass, logits + hidden_states — #14)

ARROW-NATIVE, NO LEGACY BOUNDARY
----------------------------------
Unlike streaming.py's CPU stage (which still crosses RecordBatch -> dict ->
RecordBatch once per batch around normalize_batch/enrich_batch — see design
doc #24), this module has no pre-existing row-wise Python kernel to
inherit from. It is written Arrow-native end to end: sequences are read as
an Arrow-backed column, tagging/filtering/truncation are done with
pyarrow.compute over the whole column, and only the tokenizer call itself
(which takes Python str) touches non-Arrow data -- token IDs come back out
as Arrow arrays immediately, never a dict-of-lists batch.

CORRECTNESS, NOT JUST PERFORMANCE
------------------------------------
Forgetting the <dna> tag causes the tokenizer to fall back to BPE
(English-text mode) instead of 6-mer mode -- silently wrong model input,
not a slowdown. `_assert_dna_mode_active` below is a fail-fast, one-time
per-run check against a known probe sequence, run before any real batch is
tokenized (feeds the correctness-check stage, design doc #22 step 1).
"""

import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dagster as dg
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from carbon_enrichment.assets.cpu.streaming import (
    ParquetShardWriter,
    init_worker_threads,
)
from carbon_enrichment.config import CarbonPipelineConfig
from carbon_enrichment.resources.carbon import CarbonModelResource
from carbon_enrichment.schema import (
    GPU_JOIN_KEY,
    TOKENIZED_CORPUS_COLUMNS,
)

# ============================================================================
# Constants
# ============================================================================

DNA_OPEN_TAG = "<dna>"
DNA_CLOSE_TAG = "</dna>"

# Carbon's native model context limit.
MAX_NATIVE_CONTEXT_TOKENS = 32_768

# Carbon DNA mode uses 6-mer tokenization.
DNA_KMER_SIZE = 6

# <dna> and </dna> each occupy one token.
DNA_SPECIAL_TOKEN_COUNT = 2

# Maximum number of DNA bases that can be represented without exceeding
# MAX_NATIVE_CONTEXT_TOKENS, assuming 6-mer tokenization.
MAX_DNA_BASES = (MAX_NATIVE_CONTEXT_TOKENS - DNA_SPECIAL_TOKEN_COUNT) * DNA_KMER_SIZE

# Deterministic DNA-mode correctness probe.
_DNA_MODE_PROBE_SEQUENCE = "ACGTACGTACGT"


# ============================================================================
# Stats
# ============================================================================


@dataclass
class TokenizationStats:
    """Cumulative counters for one tokenize_and_tag execution."""

    rows_read: int = 0
    rows_tokenized: int = 0

    # Number of rows excluded by the upstream canonical-DNA QC contract.
    rows_filtered_qc: int = 0

    # Number of QC-clean rows too short to form one 6-mer.
    rows_filtered_short: int = 0

    # Number of rows that had to be truncated to satisfy the native context
    # limit.
    rows_exceeding_native_context: int = 0

    total_token_count: int = 0
    min_token_length: int | None = None
    max_token_length: int | None = None
    batches_processed: int = 0
    shards_written: int = 0


def merge_tokenization_stats(
    results: list[TokenizationStats],
) -> TokenizationStats:
    """Combine per-worker TokenizationStats into one cumulative result."""

    merged = TokenizationStats()

    for result in results:
        merged.rows_read += result.rows_read
        merged.rows_tokenized += result.rows_tokenized
        merged.rows_filtered_qc += result.rows_filtered_qc
        merged.rows_filtered_short += result.rows_filtered_short
        merged.rows_exceeding_native_context += result.rows_exceeding_native_context
        merged.total_token_count += result.total_token_count

        if result.min_token_length is not None:
            merged.min_token_length = (
                result.min_token_length
                if merged.min_token_length is None
                else min(
                    merged.min_token_length,
                    result.min_token_length,
                )
            )

        if result.max_token_length is not None:
            merged.max_token_length = (
                result.max_token_length
                if merged.max_token_length is None
                else max(
                    merged.max_token_length,
                    result.max_token_length,
                )
            )

    return merged


# ============================================================================
# Tokenized output validation
# ============================================================================


def _validate_tokenized_output(batch: pa.RecordBatch) -> None:
    """Validate the complete tokenized-corpus schema contract."""

    actual_columns = tuple(batch.schema.names)

    if actual_columns != TOKENIZED_CORPUS_COLUMNS:
        raise ValueError(
            "carbon_tokenized_corpus schema mismatch: "
            f"expected {TOKENIZED_CORPUS_COLUMNS}, "
            f"got {actual_columns}"
        )

    missing_identity = [
        column for column in GPU_JOIN_KEY if column not in batch.schema.names
    ]

    if missing_identity:
        raise ValueError(
            "carbon_tokenized_corpus is missing biological join-key "
            f"columns: {missing_identity}"
        )


# ============================================================================
# Carbon DNA tokenizer validation
# ============================================================================


def _assert_dna_mode_active(tokenizer: Any) -> None:
    """
    Fail fast if the tokenizer is not actually in 6-mer <dna> mode.

    A clean 12-base sequence is used as a deterministic probe. With k=6,
    the expected structure is:

        [dna_begin_token_id, kmer_id, kmer_id, dna_end_token_id]

    No OOV token may be produced.
    """

    if tokenizer.k != DNA_KMER_SIZE:
        raise RuntimeError(
            "Unexpected Carbon DNA k-mer size: "
            f"tokenizer.k={tokenizer.k}, "
            f"expected={DNA_KMER_SIZE}"
        )

    tagged = f"{DNA_OPEN_TAG}{_DNA_MODE_PROBE_SEQUENCE}{DNA_CLOSE_TAG}"

    probe_ids = tokenizer(
        tagged,
        add_special_tokens=False,
    )["input_ids"]

    expected_kmer_count = len(_DNA_MODE_PROBE_SEQUENCE) // tokenizer.k
    expected_length = expected_kmer_count + DNA_SPECIAL_TOKEN_COUNT

    correct_structure = (
        len(probe_ids) == expected_length
        and probe_ids[0] == tokenizer.dna_begin_token_id
        and probe_ids[-1] == tokenizer.dna_end_token_id
        and tokenizer.oov_token_id not in probe_ids
    )

    if not correct_structure:
        raise RuntimeError(
            "Tokenizer not in <dna> 6-mer mode: "
            f"probe produced {probe_ids!r}, "
            "expected "
            f"[dna_begin={tokenizer.dna_begin_token_id}, "
            f"{expected_kmer_count} kmers, "
            f"dna_end={tokenizer.dna_end_token_id}]."
        )


# ============================================================================
# Sequence preprocessing
# ============================================================================


def _normalize_and_truncate_sequences(
    sequence_array: pa.Array,
) -> tuple[list[str], pa.BooleanArray, int]:
    """
    Normalize and truncate QC-clean sequences.

    Returns:
        normalized/truncated sequences for retained rows,
        keep mask identifying rows that can produce at least one 6-mer,
        number of rows truncated for native context.
    """

    normalized = pc.utf8_upper(sequence_array)
    raw_sequences = normalized.to_pylist()

    output_sequences: list[str] = []
    keep_values: list[bool] = []
    rows_truncated = 0

    for sequence in raw_sequences:
        if sequence is None:
            keep_values.append(False)
            continue

        if len(sequence) > MAX_DNA_BASES:
            sequence = sequence[:MAX_DNA_BASES]
            rows_truncated += 1

        aligned_length = (len(sequence) // DNA_KMER_SIZE) * DNA_KMER_SIZE

        if aligned_length == 0:
            keep_values.append(False)
            continue

        sequence = sequence[:aligned_length]
        output_sequences.append(sequence)
        keep_values.append(True)

    return (
        output_sequences,
        pa.array(keep_values, type=pa.bool_()),
        rows_truncated,
    )


# ============================================================================
# Optimized worker function
# ============================================================================


def _tokenize_batch(
    raw_batch: pa.RecordBatch,
    tokenizer: Any,
) -> tuple[pa.RecordBatch, TokenizationStats]:
    """
    Tokenize one Arrow batch.

    Worker identity is the biological composite key:

        (record_id, start, end)

    DNA processing:

        raw batch
            -> qc_flag == "clean" filter
            -> uppercase
            -> context-limit truncation
            -> truncate to multiple of 6
            -> <dna>SEQUENCE</dna>
            -> Carbon tokenizer
            -> token_ids / token_mask

    Rows failing the upstream canonical-DNA QC contract are excluded before
    sequence preprocessing and tokenizer invocation. No <oov> replacement is
    performed.
    """

    stats = TokenizationStats(rows_read=raw_batch.num_rows)

    required_columns = (
        *GPU_JOIN_KEY,
        "sequence",
        "qc_flag",
    )

    missing_columns = [
        column for column in required_columns if column not in raw_batch.schema.names
    ]

    if missing_columns:
        raise ValueError(f"Input batch is missing required columns: {missing_columns}")

    clean_mask = pc.equal(
        raw_batch.column("qc_flag"),
        pa.scalar("clean"),
    )
    clean_batch = raw_batch.filter(clean_mask)

    stats.rows_filtered_qc = raw_batch.num_rows - clean_batch.num_rows

    if clean_batch.num_rows == 0:
        empty_arrays = [
            pa.array([], type=clean_batch.schema.field(column).type)
            for column in GPU_JOIN_KEY
        ]
        empty_arrays.extend(
            [
                pa.array([], type=pa.list_(pa.int32())),
                pa.array([], type=pa.list_(pa.int8())),
                pa.array([], type=pa.int32()),
            ]
        )
        output_batch = pa.RecordBatch.from_arrays(
            empty_arrays,
            names=list(TOKENIZED_CORPUS_COLUMNS),
        )
        _validate_tokenized_output(output_batch)
        return output_batch, stats

    (
        normalized_sequences,
        tokenizable_mask,
        rows_truncated,
    ) = _normalize_and_truncate_sequences(
        clean_batch.column("sequence"),
    )

    tokenizable_batch = clean_batch.filter(tokenizable_mask)

    stats.rows_exceeding_native_context = rows_truncated
    stats.rows_filtered_short = clean_batch.num_rows - tokenizable_batch.num_rows

    if tokenizable_batch.num_rows == 0:
        empty_arrays = [
            pa.array([], type=tokenizable_batch.schema.field(column).type)
            for column in GPU_JOIN_KEY
        ]
        empty_arrays.extend(
            [
                pa.array([], type=pa.list_(pa.int32())),
                pa.array([], type=pa.list_(pa.int8())),
                pa.array([], type=pa.int32()),
            ]
        )

        output_batch = pa.RecordBatch.from_arrays(
            empty_arrays,
            names=list(TOKENIZED_CORPUS_COLUMNS),
        )

        _validate_tokenized_output(output_batch)
        return output_batch, stats

    # Identity must be extracted after the short-sequence filter so that
    # biological keys remain exactly aligned with tokenized sequences.
    record_id_array = tokenizable_batch.column("record_id")
    start_array = tokenizable_batch.column("start")
    end_array = tokenizable_batch.column("end")

    tagged_list = [
        f"{DNA_OPEN_TAG}{sequence}{DNA_CLOSE_TAG}" for sequence in normalized_sequences
    ]

    encoded = tokenizer(
        tagged_list,
        add_special_tokens=False,
        return_token_mask=True,
    )

    token_ids_batch = encoded["input_ids"]
    token_mask_batch = encoded["token_mask"]

    if len(token_ids_batch) != tokenizable_batch.num_rows:
        raise ValueError(
            "Tokenizer returned an unexpected number of rows: "
            f"input={tokenizable_batch.num_rows}, "
            f"tokenized={len(token_ids_batch)}"
        )

    if len(token_mask_batch) != tokenizable_batch.num_rows:
        raise ValueError(
            "Tokenizer returned an unexpected number of token masks: "
            f"input={tokenizable_batch.num_rows}, "
            f"masks={len(token_mask_batch)}"
        )

    oversized_indices = [
        index
        for index, ids in enumerate(token_ids_batch)
        if len(ids) > MAX_NATIVE_CONTEXT_TOKENS
    ]

    if oversized_indices:
        raise ValueError(
            "Tokenizer produced sequences exceeding the native context "
            f"limit of {MAX_NATIVE_CONTEXT_TOKENS:,} tokens. "
            f"First offending rows: {oversized_indices[:10]}"
        )

    token_lengths_np = np.fromiter(
        (len(ids) for ids in token_ids_batch),
        dtype=np.int32,
        count=len(token_ids_batch),
    )

    stats.total_token_count = int(token_lengths_np.sum())
    stats.min_token_length = (
        int(token_lengths_np.min()) if len(token_lengths_np) > 0 else None
    )
    stats.max_token_length = (
        int(token_lengths_np.max()) if len(token_lengths_np) > 0 else None
    )
    stats.rows_tokenized = len(token_lengths_np)

    token_ids_array = pa.array(
        token_ids_batch,
        type=pa.list_(pa.int32()),
    )
    token_mask_array = pa.array(
        token_mask_batch,
        type=pa.list_(pa.int8()),
    )
    token_length_array = pa.array(
        token_lengths_np,
        type=pa.int32(),
    )

    output_batch = pa.RecordBatch.from_arrays(
        [
            record_id_array,
            start_array,
            end_array,
            token_ids_array,
            token_mask_array,
            token_length_array,
        ],
        names=list(TOKENIZED_CORPUS_COLUMNS),
    )

    _validate_tokenized_output(output_batch)
    return output_batch, stats


# ============================================================================
# Streaming batch reader
# ============================================================================


def iter_cpu_enriched_batches(
    input_dir: str | Path,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """
    Stream RecordBatches directly from the deduplicated/CPU-enriched
    Parquet shards.

    Only columns required by tokenize_and_tag are projected.

    Input contract:

        record_id
        start
        end
        sequence
        qc_flag
    """

    input_dir = Path(input_dir)

    shard_paths = sorted(input_dir.glob("shard-*.parquet"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No CPU-enriched/pilot Parquet shards found in {input_dir}"
        )

    required_columns = [
        *GPU_JOIN_KEY,
        "sequence",
        "qc_flag",
    ]

    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)

        available_columns = parquet_file.schema_arrow.names

        missing_columns = [
            column for column in required_columns if column not in available_columns
        ]

        if missing_columns:
            raise ValueError(
                f"{shard_path} is missing required tokenize/tag "
                f"columns: {missing_columns}"
            )

        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=required_columns,
        ):
            if record_batch.num_rows:
                yield record_batch


# ============================================================================
# Ordered processing helpers
# ============================================================================


def _write_ready_batches(
    completed_batches: dict[
        int,
        tuple[pa.RecordBatch, TokenizationStats],
    ],
    writer: ParquetShardWriter,
    validation_results: list[TokenizationStats],
    next_batch_to_write: int,
) -> int:
    """
    Write every contiguous completed batch starting at next_batch_to_write.

    Worker completion order is arbitrary. Output order is deterministic.
    """

    while next_batch_to_write in completed_batches:
        token_batch, batch_stats = completed_batches.pop(next_batch_to_write)

        validation_results.append(batch_stats)
        writer.write_batch(token_batch)

        next_batch_to_write += 1

    return next_batch_to_write


# ============================================================================
# Continuous processing loop
# ============================================================================


def process_corpus(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int,
    rows_per_shard: int,
    compression: str,
    carbon_resource: CarbonModelResource,
    context: dg.AssetExecutionContext | None = None,
    max_workers: int | None = None,
) -> TokenizationStats:
    """
    Tokenize the deduplicated/CPU-enriched corpus with bounded
    multiprocessing.

    Worker execution is parallel, but output batches are written in original
    submission order.

    Biological identity is preserved directly through:

        record_id
        start
        end

    No row-position-based alignment or surrogate row_key is used.

    Rows with qc_flag != "clean" are filtered before tokenization.
    """

    tokenizer = carbon_resource.tokenizer

    # Fail before launching workers if the Carbon tokenizer is not configured
    # for the expected DNA 6-mer representation.
    _assert_dna_mode_active(tokenizer)

    n_workers = (
        max_workers if max_workers is not None else max(1, (os.cpu_count() or 2) - 1)
    )

    # Keep workers fed without allowing unbounded IPC/memory growth.
    inflight_limit = max(
        1,
        n_workers * 3,
    )

    mp_context = multiprocessing.get_context("fork")

    validation_results: list[TokenizationStats] = []

    batches_processed = 0

    # Maps Future -> original submission order.
    pending_futures: dict[Future, int] = {}

    # Completed batches that cannot yet be written because an earlier batch
    # has not completed.
    completed_batches: dict[
        int,
        tuple[pa.RecordBatch, TokenizationStats],
    ] = {}

    next_batch_to_write = 0

    batch_iter = iter_cpu_enriched_batches(
        input_dir,
        batch_size,
    )

    with (
        ParquetShardWriter(
            output_dir=output_dir,
            rows_per_shard=rows_per_shard,
            compression=compression,
            compression_level=1,
        ) as writer,
        ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
            initializer=init_worker_threads,
        ) as pool,
    ):
        # --------------------------------------------------------------
        # Submit batches while keeping the worker pool saturated.
        # --------------------------------------------------------------

        for raw_batch in batch_iter:
            if raw_batch.num_rows == 0:
                continue

            batch_id = batches_processed

            future = pool.submit(
                _tokenize_batch,
                raw_batch,
                tokenizer,
            )

            pending_futures[future] = batch_id

            batches_processed += 1

            # ----------------------------------------------------------
            # Once the bounded in-flight queue is full, consume whichever
            # worker completes first.
            # ----------------------------------------------------------

            while len(pending_futures) >= inflight_limit:
                done = next(as_completed(pending_futures))

                batch_id = pending_futures.pop(done)

                token_batch, batch_stats = done.result()

                completed_batches[batch_id] = (
                    token_batch,
                    batch_stats,
                )

                next_batch_to_write = _write_ready_batches(
                    completed_batches,
                    writer,
                    validation_results,
                    next_batch_to_write,
                )

        # --------------------------------------------------------------
        # Drain remaining workers.
        # --------------------------------------------------------------

        for done in as_completed(pending_futures):
            batch_id = pending_futures[done]

            token_batch, batch_stats = done.result()

            completed_batches[batch_id] = (
                token_batch,
                batch_stats,
            )

            next_batch_to_write = _write_ready_batches(
                completed_batches,
                writer,
                validation_results,
                next_batch_to_write,
            )

        if completed_batches:
            raise RuntimeError(
                "Tokenization completed with unwritten batches: "
                f"{sorted(completed_batches)[:20]}"
            )

        if next_batch_to_write != batches_processed:
            raise RuntimeError(
                "Tokenization batch accounting mismatch: "
                f"next_batch_to_write={next_batch_to_write}, "
                f"batches_processed={batches_processed}"
            )

        shards_written = writer.shards_written

    stats = merge_tokenization_stats(validation_results)

    stats.batches_processed = batches_processed
    stats.shards_written = shards_written

    return stats


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    name="carbon_tokenized_corpus",
    group_name="cpu",
    compute_kind="cpu",
    deps=["carbon_pilot_corpus"],
    description=(
        "Deterministic Carbon DNA 6-mer tokenization/tagging pass with "
        "upstream canonical-DNA QC filtering, context enforcement, bounded "
        "multiprocessing, stable composite biological identity, and "
        "ordered output."
    ),
)
def carbon_tokenized_corpus(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
    carbon: CarbonModelResource,
) -> dg.MaterializeResult:
    context.log.info("Starting Carbon DNA tokenize_and_tag stage.")

    stats = process_corpus(
        input_dir=config.pilot_output_dir,
        output_dir=config.tokenized_output_dir,
        batch_size=getattr(
            config,
            "tokenization_batch_size",
            2048,
        ),
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        carbon_resource=carbon,
        context=context,
        max_workers=getattr(
            config,
            "cpu_workers",
            None,
        ),
    )

    if stats.rows_exceeding_native_context:
        context.log.info(
            f"{stats.rows_exceeding_native_context:,} rows were "
            f"truncated to the {MAX_NATIVE_CONTEXT_TOKENS:,}-token "
            "native context limit."
        )

    context.log.info(
        "Tokenization completed: "
        f"rows_read={stats.rows_read:,}, "
        f"rows_tokenized={stats.rows_tokenized:,}, "
        f"rows_filtered_qc={stats.rows_filtered_qc:,}, "
        f"rows_filtered_short={stats.rows_filtered_short:,}, "
        f"shards={stats.shards_written:,}"
    )

    return dg.MaterializeResult(
        metadata={
            "rows_read": stats.rows_read,
            "rows_tokenized": stats.rows_tokenized,
            "rows_filtered_qc": stats.rows_filtered_qc,
            "rows_filtered_short": stats.rows_filtered_short,
            "rows_truncated_to_native_context": (stats.rows_exceeding_native_context),
            "max_native_context_tokens": MAX_NATIVE_CONTEXT_TOKENS,
            "max_dna_bases": MAX_DNA_BASES,
            "dna_kmer_size": DNA_KMER_SIZE,
            "min_token_length": stats.min_token_length,
            "max_token_length": stats.max_token_length,
            "mean_token_length": (
                round(
                    stats.total_token_count / stats.rows_tokenized,
                    1,
                )
                if stats.rows_tokenized
                else None
            ),
            "batches_processed": stats.batches_processed,
            "parquet_shards": stats.shards_written,
            "tokenizer_revision": carbon.tokenizer_revision,
            "gpu_join_key": list(GPU_JOIN_KEY),
            "qc_filter": "qc_flag == clean",
        },
    )
