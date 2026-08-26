"""
M2 — Streaming CPU execution for the Carbon pipeline.

Processing flow:

    Hugging Face IterableDataset
                │
                ▼
          bounded batch (pa.RecordBatch)
                │
                ▼
          schema validation (main process, first batch only)
                │
                ▼
   ┌─── worker process (pool) ───────────────────────────┐
   │  RecordBatch → dict   (unavoidable: normalize/       │
   │       ↓         validate/enrich are row-wise Python) │
   │  normalization                                       │
   │       ↓                                               │
   │  validation (local stats)                             │
   │       ↓                                               │
   │  NumPy enrichment                                     │
   │       ↓                                               │
   │  dict → RecordBatch                                   │
   └───────────────────────────────────────────────────────┘
                │
                ▼
          Parquet shards (main process, Arrow write, no
          intermediate Table/dict materialization)

The complete Carbon corpus is never materialized as a Hugging Face Dataset,
pandas DataFrame, or unbounded Python collection.

ARROW-NATIVE TRANSPORT
-----------------------
`pa.RecordBatch` is the canonical batch type for this module — it is what
flows into worker processes, what the writer receives, and what gets
sliced across shard boundaries. This avoids the
Parquet -> dict-of-lists -> ... -> Arrow round-trip that would otherwise
happen once per batch at write time.

The one place this module still touches plain Python containers is inside
`_process_batch`, immediately around the calls to `normalize_batch`,
`ValidationStats.update`, and `enrich_batch`. Those three functions are
row-wise Python kernels (string strip/upper, IUPAC translate tables,
taxonomy string splitting) — they are not vectorized Arrow kernels, and
rewriting them to operate on `pa.Array`/`pyarrow.compute` is out of scope
here. The `RecordBatch -> dict -> RecordBatch` conversion is therefore
narrowed to exactly that one boundary, inside the worker process, once per
batch — not repeated at every stage of the pipeline.

PARALLEL EXECUTION
-------------------
Normalization, validation, and enrichment are CPU-bound, per-batch-pure
operations with no shared state between batches. They are fused into a
single `_process_batch` unit of work and submitted to a
`ProcessPoolExecutor`, one call per batch. This keeps the previously
sequential normalize -> validate -> enrich chain off the main process,
which now only handles streaming iteration, submission, and Parquet I/O.

In-flight futures are bounded (`inflight_limit`) so memory use stays
proportional to `batch_size * n_workers`, preserving the bounded-memory
streaming contract described above -- an unbounded queue of pending
futures would otherwise defeat the point of streaming.

`pa.RecordBatch` objects are picklable (Arrow IPC under the hood), so they
cross the `ProcessPoolExecutor` boundary the same way the previous
dict-of-lists batches did -- no change to the submission/result contract.

Per-worker ValidationStats instances are merged in the main process via
`merge_validation_stats()`, since dataclass mutation does not cross
process boundaries.
"""

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dagster as dg
import pyarrow as pa
import pyarrow.parquet as pq
from dagster_hf_datasets import HuggingFaceResource
from datasets import load_dataset

from carbon_enrichment.assets.cpu.enrichment import enrich_batch
from carbon_enrichment.assets.cpu.ingest import create_carbon_stream
from carbon_enrichment.assets.cpu.normalization import normalize_batch
from carbon_enrichment.assets.cpu.validation import (
    ValidationStats,
    add_validation_metadata,
    merge_validation_stats,
    validate_schema,
)
from carbon_enrichment.config import CarbonPipelineConfig

# ============================================================================
# Batch type
# ============================================================================

Batch = pa.RecordBatch

BatchTransform = Callable[[Batch], Batch]


# ============================================================================
# Worker initialization
# ============================================================================


def init_worker_threads() -> None:
    """Clamp internal C/C++ threadpools inside each forked worker process."""
    pa.set_cpu_count(1)
    pa.set_io_cpu_count(1)


# ============================================================================
# Input readers
# ============================================================================


def _read_local_parquet(
    input_dir: str | Path,
    *,
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    """
    Stream rows from locally materialized CPU-enriched Parquet shards.

    Reads each shard under `input_dir` in filename order via Arrow's
    batched reader, yielding one row dict at a time. Mirrors the bounded-
    memory contract used elsewhere in the pipeline -- at most one
    `batch_size`-sized Arrow batch is held in memory at a time, never a
    full shard or the full corpus.
    """

    input_path = Path(input_dir)
    shard_paths = sorted(input_path.glob("*.parquet"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No Parquet shards found in {input_path!s} "
            "(expected carbon_cpu_enriched_sequences to have run first)"
        )

    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            yield record_batch


def _read_hub_dataset(
    dataset_id: str,
    *,
    batch_size: int,
    revision: str | None = None,
) -> Iterator[Mapping[str, Any]]:
    """
    Stream rows from a Hugging Face Hub dataset without materializing it.

    Uses `load_dataset(..., streaming=True)` so the CPU-enriched corpus is
    never pulled fully into memory or disk -- consistent with this
    pipeline's bounded-memory streaming contract. `batch_size` is accepted
    for interface symmetry with `_read_local_parquet` but is not used to
    chunk here; row-by-row iteration is left to the caller's own batching
    (e.g. `iter_batches`), same as the local reader's per-row yield.

    Pin `revision` to a commit SHA for reproducible sampling runs -- the Hub
    repository at `dataset_id` is mutable, and an unpinned revision means
    "whatever main currently contains" rather than a fixed input.

    The upstream Hub stream is explicitly closed when iteration terminates,
    including when a downstream consumer stops early because it has reached
    its configured row limit.
    """

    hub_stream = load_dataset(
        dataset_id,
        split="train",
        streaming=True,
        revision=revision,
    )
    try:
        yield from hub_stream
    finally:
        close = getattr(hub_stream, "close", None)
        if callable(close):
            close()


# ============================================================================
# Streaming statistics
# ============================================================================


@dataclass
class StreamingStats:
    """Counters collected during one streaming execution."""

    rows_read: int = 0
    rows_normalized: int = 0
    rows_enriched: int = 0
    batches_processed: int = 0
    shards_written: int = 0


# ============================================================================
# Batch conversion
# ============================================================================


def iter_batches(
    stream: Iterable[Any],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Convert a row stream or pass-through RecordBatch stream into bounded batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    # If the stream is already an iterator of RecordBatches, yield directly
    stream_iter = iter(stream)
    try:
        first_item = next(stream_iter)
    except StopIteration:
        return

    if isinstance(first_item, pa.RecordBatch):
        yield first_item
        yield from stream_iter
        return

    # Otherwise buffer row dictionaries into RecordBatches
    rows: list[Mapping[str, Any]] = [first_item]
    try:
        for row in stream_iter:
            rows.append(row)
            if len(rows) >= batch_size:
                yield _rows_to_record_batch(rows)
                rows = []

        if rows:
            yield _rows_to_record_batch(rows)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def _rows_to_record_batch(rows: list[Mapping[str, Any]]) -> pa.RecordBatch:
    """Convert row-oriented records directly into a pa.RecordBatch."""
    table = pa.Table.from_pylist(rows)
    return table.combine_chunks().to_batches()[0]


# ============================================================================
# Fused worker unit: normalize -> validate -> enrich
# ============================================================================


def _process_batch(
    raw_batch: pa.RecordBatch,
) -> tuple[pa.RecordBatch, ValidationStats, int]:
    """
    Run normalization, validation, and enrichment for one batch.

    Executed inside a worker process. Returns the enriched batch as a
    pa.RecordBatch, a worker-local ValidationStats, and the raw row count
    (for stats bookkeeping in the parent process).

    Per design doc #24, normalize_batch/enrich_batch are now vectorized
    pyarrow.compute kernels operating on RecordBatch directly -- the
    RecordBatch -> dict -> RecordBatch boundary this function used to
    cross for the ENTIRE chain now only exists around the
    ValidationStats.update call, since validation.py is explicitly out of
    #24's scope and still expects a dict-of-lists batch.
    """

    raw_rows = raw_batch.num_rows

    normalized_batch = normalize_batch(raw_batch)
    normalized_rows = normalized_batch.num_rows

    if normalized_rows != raw_rows:
        raise RuntimeError(
            f"Normalization changed the number of rows: input={raw_rows}, output={normalized_rows}"
        )

    local_stats = ValidationStats()
    # Fast dict update for validation stats
    local_stats.update(normalized_batch.to_pydict())

    enriched_batch = enrich_batch(normalized_batch)
    enriched_rows = enriched_batch.num_rows

    if enriched_rows != normalized_rows:
        raise RuntimeError(
            f"Enrichment changed the number of rows: input={normalized_rows}, output={enriched_rows}"
        )

    return enriched_batch, local_stats, raw_rows


# ============================================================================
# Parquet shard writer
# ============================================================================


class ParquetShardWriter:
    """Write bounded pa.RecordBatch objects into sequential Parquet shards with fast compression."""

    def __init__(
        self,
        output_dir: str | Path,
        rows_per_shard: int,
        compression: str = "zstd",
        compression_level: int = 1,  # Fast level 1 for high throughput intermediate staging
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be greater than zero")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.compression_level = compression_level

        self._writer: pq.ParquetWriter | None = None
        self._rows_in_current_shard = 0
        self._shard_index = 0
        self._rows_written = 0
        self._shards_written = 0
        self._is_closed = False

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def shards_written(self) -> int:
        return self._shards_written

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Write a bounded RecordBatch, splitting it across shards if required."""
        if batch.num_rows == 0 or self._is_closed:
            return

        offset = 0
        while offset < batch.num_rows:
            capacity = self.rows_per_shard - self._rows_in_current_shard
            take = min(capacity, batch.num_rows - offset)
            if take > 0:
                self._write_chunk(batch.slice(offset, take))
                offset += take

    def _write_chunk(self, record_batch: pa.RecordBatch) -> None:
        if record_batch.num_rows == 0:
            return

        if self._writer is None:
            kwargs: dict[str, Any] = {}
            if self.compression.lower() == "zstd":
                kwargs["compression_level"] = self.compression_level

            shard_path = self._shard_path(self._shard_index)
            self._writer = pq.ParquetWriter(
                shard_path,
                record_batch.schema,
                compression=self.compression,
                **kwargs,
            )

        self._writer.write_batch(record_batch)
        self._rows_in_current_shard += record_batch.num_rows
        self._rows_written += record_batch.num_rows

        if self._rows_in_current_shard >= self.rows_per_shard:
            self._close_current_shard()

    def _shard_path(self, index: int) -> Path:
        return self.output_dir / f"shard-{index:05d}.parquet"

    def _close_current_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._shards_written += 1
            self._shard_index += 1
            self._rows_in_current_shard = 0

    def close(self) -> None:
        """Close the active shard and mark writer as finalized."""
        if not self._is_closed:
            self._close_current_shard()
            self._is_closed = True

    def __enter__(self) -> "ParquetShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


# ============================================================================
# Streaming processor
# ============================================================================


def process_stream(
    stream: Iterable[Any],
    output_dir: str | Path,
    *,
    batch_size: int,
    rows_per_shard: int,
    compression: str,
    context: dg.AssetExecutionContext | None = None,
    max_workers: int | None = None,
) -> tuple[StreamingStats, ValidationStats]:
    stats = StreamingStats()
    validation_results: list[ValidationStats] = []
    first_batch = True

    n_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    inflight_limit = n_workers * 3

    mp_context = multiprocessing.get_context("fork")

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
        pending_futures: dict[Future, None] = {}
        batch_iter = iter_batches(stream, batch_size=batch_size)

        for raw_batch in batch_iter:
            raw_rows = raw_batch.num_rows
            if raw_rows == 0:
                continue

            stats.rows_read += raw_rows

            if first_batch:
                validate_schema({name: None for name in raw_batch.schema.names})
                first_batch = False

            future = pool.submit(_process_batch, raw_batch)
            pending_futures[future] = None

            # Asynchronously consume worker outputs as soon as ANY worker finishes
            while len(pending_futures) >= inflight_limit:
                done = next(as_completed(pending_futures))
                pending_futures.pop(done)
                enriched, local_stats, _raw_rows = done.result()

                enriched_rows = enriched.num_rows
                stats.rows_normalized += enriched_rows
                stats.rows_enriched += enriched_rows
                validation_results.append(local_stats)
                writer.write_batch(enriched)
                stats.batches_processed += 1

                if context is not None and stats.batches_processed % 100 == 0:
                    context.log.info(
                        f"Streaming progress: {stats.rows_read:,} rows read, "
                        f"{stats.rows_enriched:,} enriched, {stats.batches_processed:,} batches"
                    )

        # Drain remaining futures
        for done in as_completed(pending_futures):
            enriched, local_stats, _raw_rows = done.result()
            enriched_rows = enriched.num_rows
            stats.rows_normalized += enriched_rows
            stats.rows_enriched += enriched_rows
            validation_results.append(local_stats)
            writer.write_batch(enriched)
            stats.batches_processed += 1

    stats.shards_written = writer.shards_written
    validation_stats = merge_validation_stats(validation_results)

    return stats, validation_stats


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    name="carbon_cpu_enriched_sequences",
    group_name="cpu",
    compute_kind="cpu",
    description="Streaming Carbon CPU pipeline with non-blocking process pool execution.",
)
def carbon_cpu_enriched_sequences(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
    hf_resource: HuggingFaceResource,
) -> dg.MaterializeResult:
    if not config.streaming:
        raise ValueError("carbon_cpu_enriched_sequences requires streaming=True.")

    context.log.info("Starting streaming Carbon CPU pipeline.")
    cpu_workers = getattr(config, "cpu_workers", None)

    stream = create_carbon_stream(
        config=config,
        hf_resource=hf_resource,
    )

    stats, validation_stats = process_stream(
        stream=stream,
        output_dir=config.output_dir,
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        context=context,
        max_workers=cpu_workers,
    )

    if stats.rows_read != stats.rows_normalized:
        raise RuntimeError(
            f"Normalization changed total row count: input={stats.rows_read}, normalized={stats.rows_normalized}"
        )

    if stats.rows_normalized != stats.rows_enriched:
        raise RuntimeError(
            f"Enrichment changed total row count: normalized={stats.rows_normalized}, enriched={stats.rows_enriched}"
        )

    add_validation_metadata(context, validation_stats)

    context.log.info(
        f"Streaming Carbon CPU pipeline completed: rows={stats.rows_enriched:,}, batches={stats.batches_processed:,}, shards={stats.shards_written:,}"
    )

    return dg.MaterializeResult(
        metadata={
            "rows_read": stats.rows_read,
            "rows_normalized": stats.rows_normalized,
            "rows_enriched": stats.rows_enriched,
            "batches_processed": stats.batches_processed,
            "parquet_shards": stats.shards_written,
            "batch_size": config.batch_size,
            "rows_per_shard": config.rows_per_shard,
            "compression": config.compression,
            "output_dir": config.output_dir,
            "cpu_workers": cpu_workers,
            "streaming": True,
        }
    )
