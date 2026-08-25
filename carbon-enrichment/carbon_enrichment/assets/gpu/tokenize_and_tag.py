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
import pyarrow.parquet as pq

from carbon_enrichment.assets.cpu.streaming import ParquetShardWriter
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
MAX_NATIVE_CONTEXT_TOKENS = 32_768
_DNA_MODE_PROBE_SEQUENCE = "ACGTACGTACGT"


# ============================================================================
# Worker initialization
# ============================================================================


def _init_worker_threads() -> None:
    """Clamp internal C/C++ threadpools inside each forked worker process."""
    pa.set_cpu_count(1)
    pa.set_io_cpu_count(1)


# ============================================================================
# Stats
# ============================================================================


@dataclass
class TokenizationStats:
    """Cumulative counters for one tokenize_and_tag execution."""

    rows_read: int = 0
    rows_tokenized: int = 0
    oov_bases_filtered: int = 0
    rows_exceeding_native_context: int = 0
    total_token_count: int = 0
    min_token_length: int | None = None
    max_token_length: int | None = None
    batches_processed: int = 0
    shards_written: int = 0


def merge_tokenization_stats(results: list[TokenizationStats]) -> TokenizationStats:
    """Combine per-worker TokenizationStats into one cumulative result."""
    merged = TokenizationStats()
    for r in results:
        merged.rows_read += r.rows_read
        merged.rows_tokenized += r.rows_tokenized
        merged.oov_bases_filtered += r.oov_bases_filtered
        merged.rows_exceeding_native_context += r.rows_exceeding_native_context
        merged.total_token_count += r.total_token_count

        if r.min_token_length is not None:
            merged.min_token_length = (
                r.min_token_length
                if merged.min_token_length is None
                else min(merged.min_token_length, r.min_token_length)
            )

        if r.max_token_length is not None:
            merged.max_token_length = (
                r.max_token_length
                if merged.max_token_length is None
                else max(merged.max_token_length, r.max_token_length)
            )
    return merged


def _validate_tokenized_output(batch: pa.RecordBatch) -> None:
    actual_columns = tuple(batch.schema.names)
    if actual_columns != TOKENIZED_CORPUS_COLUMNS:
        raise ValueError(
            f"carbon_tokenized_corpus schema mismatch: expected {TOKENIZED_CORPUS_COLUMNS}, got {actual_columns}"
        )
    if GPU_JOIN_KEY not in batch.schema.names:
        raise ValueError(f"carbon_tokenized_corpus is missing join key {GPU_JOIN_KEY!r}")


def _assert_dna_mode_active(tokenizer: Any) -> None:
    """
    Fail fast if the tokenizer is not actually in 6-mer <dna> mode.

    Unlike a standard AutoTokenizer, <dna>/</dna>/<oov> are NOT registered
    HF special tokens (they don't appear in tokenizer_config.json's
    added_tokens_decoder) -- HybridDNATokenizer assigns them custom vocab
    IDs programmatically in _init_dna_vocab, exposed directly as
    tokenizer.dna_begin_token_id / .dna_end_token_id. This checks the
    exact expected token structure rather than a fuzzy token-count
    heuristic: a clean 12-base sequence, a multiple of k=6, must tokenize
    to exactly [dna_begin_token_id, kmer_id, kmer_id, dna_end_token_id].
    """

    tagged = f"{DNA_OPEN_TAG}{_DNA_MODE_PROBE_SEQUENCE}{DNA_CLOSE_TAG}"
    probe_ids = tokenizer(tagged, add_special_tokens=False)["input_ids"]
    expected_kmer_count = len(_DNA_MODE_PROBE_SEQUENCE) // tokenizer.k
    expected_length = expected_kmer_count + 2

    correct_structure = (
        len(probe_ids) == expected_length
        and probe_ids[0] == tokenizer.dna_begin_token_id
        and probe_ids[-1] == tokenizer.dna_end_token_id
        and tokenizer.oov_token_id not in probe_ids
    )

    if not correct_structure:
        raise RuntimeError(
            f"Tokenizer not in <dna> 6-mer mode: probe produced {probe_ids!r}, "
            f"expected [dna_begin={tokenizer.dna_begin_token_id}, {expected_kmer_count} kmers, "
            f"dna_end={tokenizer.dna_end_token_id}]."
        )


# ============================================================================
# Optimized Worker Function
# ============================================================================


def _tokenize_batch(
    raw_batch: pa.RecordBatch,
    tokenizer: Any,
) -> tuple[pa.RecordBatch, TokenizationStats]:
    """Worker task: Vectorized tokenization and direct Arrow construction."""
    stats = TokenizationStats(rows_read=raw_batch.num_rows)

    if GPU_JOIN_KEY not in raw_batch.schema.names:
        raise ValueError(f"Input batch is missing required join key {GPU_JOIN_KEY!r}")

    # Direct Python list formatting avoids Arrow join kernel + to_pylist() round-trip
    raw_seqs = raw_batch.column("sequence").to_pylist()
    tagged_list = [f"{DNA_OPEN_TAG}{s}{DNA_CLOSE_TAG}" for s in raw_seqs]

    encoded = tokenizer(
        tagged_list,
        add_special_tokens=False,
        return_token_mask=True,
    )

    token_ids_batch = encoded["input_ids"]
    token_mask_batch = encoded["token_mask"]

    token_lengths_np = np.fromiter((len(ids) for ids in token_ids_batch), dtype=np.int32, count=len(token_ids_batch))

    # Fast OOV count computation
    oov_id = tokenizer.oov_token_id
    oov_count = sum(ids.count(oov_id) for ids in token_ids_batch)

    stats.oov_bases_filtered = oov_count
    stats.total_token_count = int(token_lengths_np.sum())
    stats.min_token_length = int(token_lengths_np.min()) if len(token_lengths_np) > 0 else None
    stats.max_token_length = int(token_lengths_np.max()) if len(token_lengths_np) > 0 else None
    stats.rows_exceeding_native_context = int((token_lengths_np > MAX_NATIVE_CONTEXT_TOKENS).sum())
    stats.rows_tokenized = len(token_lengths_np)

    # Convert directly to PyArrow arrays
    token_ids_array = pa.array(token_ids_batch, type=pa.list_(pa.int32()))
    token_mask_array = pa.array(token_mask_batch, type=pa.list_(pa.int8()))
    token_length_array = pa.array(token_lengths_np, type=pa.int32())

    output_batch = pa.RecordBatch.from_arrays(
        [
            raw_batch.column(GPU_JOIN_KEY),
            token_ids_array,
            token_mask_array,
            token_length_array,
        ],
        names=list(TOKENIZED_CORPUS_COLUMNS),
    )

    _validate_tokenized_output(output_batch)
    return output_batch, stats


# ============================================================================
# Streaming Batch Reader
# ============================================================================


def iter_cpu_enriched_batches(
    input_dir: str | Path,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """
    Stream RecordBatches directly from CPU-enriched Parquet shards.

    Reads only the columns tokenize_and_tag actually needs (record_id,
    sequence) via ParquetFile.iter_batches column projection, rather than
    materializing full shards or round-tripping through pandas.
    """

    input_dir = Path(input_dir)
    shard_paths = sorted(input_dir.glob("shard-*.parquet"))

    if not shard_paths:
        raise FileNotFoundError(f"No CPU-enriched Parquet shards found in {input_dir}")

    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["record_id", "sequence"],
        ):
            yield record_batch


# ============================================================================
# Continuous Processing Loop
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
    tokenizer = carbon_resource.tokenizer
    _assert_dna_mode_active(tokenizer)

    n_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    inflight_limit = n_workers * 3  # Keeps workers fed without excessive IPC queue memory

    mp_context = multiprocessing.get_context("fork")
    validation_results: list[TokenizationStats] = []
    batches_processed = 0

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
            initializer=_init_worker_threads,
        ) as pool,
    ):
        # Dictionary tracking in-flight futures: {Future: submit_order}
        pending_futures: dict[Future, None] = {}
        batch_iter = iter_cpu_enriched_batches(input_dir, batch_size)

        for raw_batch in batch_iter:
            if raw_batch.num_rows == 0:
                continue

            future = pool.submit(_tokenize_batch, raw_batch, tokenizer)
            pending_futures[future] = None
            batches_processed += 1

            # When capacity is reached, consume as soon as ANY worker completes
            while len(pending_futures) >= inflight_limit:
                done = next(as_completed(pending_futures))
                pending_futures.pop(done)
                token_batch, batch_stats = done.result()
                validation_results.append(batch_stats)
                writer.write_batch(token_batch)

                if context is not None and batches_processed % 100 == 0:
                    context.log.info(f"Tokenization progress: {batches_processed:,} batches submitted")

        # Drain all remaining in-flight tasks
        for done in as_completed(pending_futures):
            token_batch, batch_stats = done.result()
            validation_results.append(batch_stats)
            writer.write_batch(token_batch)

    stats = merge_tokenization_stats(validation_results)
    stats.batches_processed = batches_processed
    stats.shards_written = writer.shards_written
    return stats


# ============================================================================
# Dagster Asset
# ============================================================================


@dg.asset(
    name="carbon_tokenized_corpus",
    group_name="cpu",
    compute_kind="cpu",
    deps=["carbon_pilot_corpus"],
    description="Optimized tokenization/tagging pass with continuous worker streaming.",
)
def carbon_tokenized_corpus(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
    carbon: CarbonModelResource,
) -> dg.MaterializeResult:
    context.log.info("Starting optimized tokenize_and_tag stage.")

    stats = process_corpus(
        input_dir=config.pilot_output_dir,
        output_dir=config.tokenized_output_dir,
        batch_size=getattr(config, "tokenization_batch_size", 2048),  # Use larger batch sizes (1k-2k) for CPU workers
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        carbon_resource=carbon,
        context=context,
        max_workers=getattr(config, "cpu_workers", None),
    )

    if stats.rows_exceeding_native_context:
        context.log.warning(
            f"{stats.rows_exceeding_native_context} rows exceeded {MAX_NATIVE_CONTEXT_TOKENS} native-context tokens."
        )

    context.log.info(
        f"Tokenization completed: rows={stats.rows_tokenized:,}, shards={stats.shards_written:,}, oov_filtered={stats.oov_bases_filtered:,}"
    )

    return dg.MaterializeResult(
        metadata={
            "rows_read": stats.rows_read,
            "rows_tokenized": stats.rows_tokenized,
            "oov_bases_filtered": stats.oov_bases_filtered,
            "rows_exceeding_native_context": stats.rows_exceeding_native_context,
            "min_token_length": stats.min_token_length,
            "max_token_length": stats.max_token_length,
            "mean_token_length": (
                round(stats.total_token_count / stats.rows_tokenized, 1) if stats.rows_tokenized else None
            ),
            "batches_processed": stats.batches_processed,
            "parquet_shards": stats.shards_written,
            "tokenizer_revision": carbon.tokenizer_revision,
        }
    )