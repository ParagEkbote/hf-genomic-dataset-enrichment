"""
M2 — Streaming CPU execution for the Carbon pipeline.

Processing flow:

    Hugging Face IterableDataset
                │
                ▼
          bounded batch
                │
                ▼
          schema validation (main process, first batch only)
                │
                ▼
   ┌─── worker process (pool) ───┐
   │  normalization               │
   │       ↓                      │
   │  validation (local stats)    │
   │       ↓                      │
   │  NumPy enrichment             │
   └───────────────────────────────┘
                │
                ▼
          Parquet shards (main process)

The complete Carbon corpus is never materialized as a Hugging Face Dataset,
pandas DataFrame, or unbounded Python collection.

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

Per-worker ValidationStats instances are merged in the main process via
`merge_validation_stats()`, since dataclass mutation does not cross
process boundaries.
"""

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dagster as dg
import pyarrow as pa
import pyarrow.parquet as pq
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
from dagster_hf_datasets import HuggingFaceResource

Batch = dict[str, list[Any]]

BatchTransform = Callable[
    [Batch],
    Batch,
]


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
    stream: Iterable[Mapping[str, Any]],
    batch_size: int,
) -> Iterator[Batch]:
    """
    Convert a row-oriented stream into bounded column-oriented batches.

    At most `batch_size` rows are retained by this batching layer.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    rows: list[Mapping[str, Any]] = []

    for row in stream:
        rows.append(row)

        if len(rows) >= batch_size:
            yield _rows_to_batch(rows)

            rows = []

    if rows:
        yield _rows_to_batch(rows)


def _rows_to_batch(
    rows: list[Mapping[str, Any]],
) -> Batch:
    """Convert row-oriented records into a column-oriented batch."""

    if not rows:
        return {}

    columns = rows[0].keys()

    return {column: [row.get(column) for row in rows] for column in columns}


def _batch_length(
    batch: Mapping[str, Any],
) -> int:
    """Return the number of rows in a column-oriented batch."""

    if not batch:
        return 0

    return len(next(iter(batch.values())))


def _validate_batch_column_lengths(
    batch: Mapping[str, Any],
) -> None:
    """
    Ensure every column in a batch contains the same number of rows.
    """

    if not batch:
        return

    lengths = {column: len(values) for column, values in batch.items()}

    if len(set(lengths.values())) > 1:
        raise RuntimeError(f"Batch columns have inconsistent lengths: {lengths}")


# ============================================================================
# Fused worker unit: normalize -> validate -> enrich
# ============================================================================
#
# This function is the payload submitted to the process pool. It must be a
# top-level, picklable function (no closures) since ProcessPoolExecutor
# pickles the callable and its arguments to send to worker processes.
#
# Each call gets its own local ValidationStats -- never shared across
# processes -- and returns it alongside the enriched batch so the parent
# process can merge it with merge_validation_stats().


def _process_batch(
    raw_batch: Batch,
) -> tuple[Batch, ValidationStats, int]:
    """
    Run normalization, validation, and enrichment for one batch.

    Executed inside a worker process. Returns the enriched batch, a
    worker-local ValidationStats, and the raw row count (for stats
    bookkeeping in the parent process).
    """

    raw_rows = _batch_length(raw_batch)

    normalized = normalize_batch(raw_batch)

    normalized_rows = _batch_length(normalized)

    if normalized_rows != raw_rows:
        raise RuntimeError(
            "Normalization changed the number of rows: "
            f"input={raw_rows}, output={normalized_rows}"
        )

    _validate_batch_column_lengths(normalized)

    local_stats = ValidationStats()
    local_stats.update(normalized)

    enriched = enrich_batch(normalized)

    enriched_rows = _batch_length(enriched)

    if enriched_rows != normalized_rows:
        raise RuntimeError(
            "Enrichment changed the number of rows: "
            f"input={normalized_rows}, output={enriched_rows}"
        )

    _validate_batch_column_lengths(enriched)

    return enriched, local_stats, raw_rows


# ============================================================================
# Parquet shard writer
# ============================================================================


class ParquetShardWriter:
    """Write bounded batches into sequential Parquet shards."""

    def __init__(
        self,
        output_dir: str | Path,
        rows_per_shard: int,
        compression: str,
    ) -> None:

        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be greater than zero")

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.rows_per_shard = rows_per_shard

        self.compression = compression

        self._writer: pq.ParquetWriter | None = None

        self._rows_in_current_shard = 0

        self._shard_index = 0

        self._rows_written = 0

        self._shards_written = 0

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def shards_written(self) -> int:
        return self._shards_written

    def write_batch(
        self,
        batch: Batch,
    ) -> None:
        """Write a bounded batch, splitting it across shards if required."""

        if not batch:
            return

        _validate_batch_column_lengths(batch)

        table = pa.Table.from_pydict(batch)

        offset = 0

        while offset < table.num_rows:
            capacity = self.rows_per_shard - self._rows_in_current_shard

            take = min(
                capacity,
                table.num_rows - offset,
            )

            self._write_chunk(
                table.slice(
                    offset,
                    take,
                )
            )

            offset += take

    def _write_chunk(
        self,
        table: pa.Table,
    ) -> None:

        if table.num_rows == 0:
            return

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._shard_path(self._shard_index),
                table.schema,
                compression=self.compression,
            )

        self._writer.write_table(table)

        self._rows_in_current_shard += table.num_rows

        self._rows_written += table.num_rows

        if self._rows_in_current_shard >= self.rows_per_shard:
            self._close_current_shard()

    def _shard_path(
        self,
        index: int,
    ) -> Path:

        return self.output_dir / f"shard-{index:05d}.parquet"

    def _close_current_shard(self) -> None:

        if self._writer is None:
            return

        self._writer.close()

        self._writer = None

        self._shards_written += 1

        self._shard_index += 1

        self._rows_in_current_shard = 0

    def close(self) -> None:
        """Close the final partial shard."""

        self._close_current_shard()

    def __enter__(
        self,
    ) -> "ParquetShardWriter":

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.close()


# ============================================================================
# Streaming processor
# ============================================================================


def process_stream(
    stream: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    batch_size: int,
    rows_per_shard: int,
    compression: str,
    context: dg.AssetExecutionContext | None = None,
    max_workers: int | None = None,
) -> tuple[
    StreamingStats,
    ValidationStats,
]:
    """
    Process one Carbon stream using bounded memory.

    Per-batch work order (executed inside worker processes):

        raw
         ↓
        normalization
         ↓
        validation
         ↓
        enrichment

    The main process handles only: schema check (first batch), streaming
    iteration, submitting batches to the pool, draining completed futures,
    and Parquet writing.
    """

    stats = StreamingStats()

    validation_results: list[ValidationStats] = []

    first_batch = True

    n_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    # Bound how many batches can be in flight at once so memory stays
    # proportional to worker count, preserving bounded-memory streaming.
    inflight_limit = n_workers * 4

    # Use "fork" explicitly rather than relying on the platform default.
    # Dagster's multiprocess executor already runs this asset inside its
    # own subprocess (STEP_WORKER). Nesting a "spawn"-based pool inside
    # that subprocess forces each worker to reimport the Python process
    # from scratch, which can fail to resolve locally/editable-installed
    # packages such as carbon_enrichment. "fork" instead inherits the
    # parent's already-loaded modules and sys.path, avoiding the reimport
    # entirely. This requires Linux/macOS; fork is not available on
    # Windows, where spawn is the only option.
    mp_context = multiprocessing.get_context("fork")

    with (
        ParquetShardWriter(
            output_dir=output_dir,
            rows_per_shard=rows_per_shard,
            compression=compression,
        ) as writer,
        ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
        ) as pool,
    ):
        pending: list[Future] = []

        def _drain(
            futures: list[Future],
        ) -> None:

            for future in futures:
                enriched, local_stats, _raw_rows = future.result()

                enriched_rows = _batch_length(enriched)

                _validate_batch_column_lengths(enriched)

                stats.rows_normalized += enriched_rows
                stats.rows_enriched += enriched_rows

                validation_results.append(local_stats)

                writer.write_batch(enriched)

                stats.batches_processed += 1

        for raw_batch in iter_batches(
            stream,
            batch_size=batch_size,
        ):
            raw_rows = _batch_length(raw_batch)

            if raw_rows == 0:
                continue

            _validate_batch_column_lengths(raw_batch)

            stats.rows_read += raw_rows

            # --------------------------------------------------------------
            # Schema contract
            #
            # Only the first batch needs the blocking schema check. This
            # stays sequential and in the main process, since it is a
            # cheap, one-time, fail-fast gate.
            # --------------------------------------------------------------

            if first_batch:
                validate_schema(raw_batch)

                first_batch = False

            # --------------------------------------------------------------
            # Submit normalize -> validate -> enrich as one unit of work.
            # --------------------------------------------------------------

            pending.append(pool.submit(_process_batch, raw_batch))

            if len(pending) >= inflight_limit:
                _drain(pending)

                pending = []

                if context is not None and stats.batches_processed % 100 == 0:
                    context.log.info(
                        "Streaming progress: "
                        f"{stats.rows_read:,} rows read, "
                        f"{stats.rows_enriched:,} enriched, "
                        f"{stats.batches_processed:,} batches"
                    )

        # Flush any remaining in-flight batches.
        _drain(pending)

        stats.shards_written = writer.shards_written

    validation_stats = merge_validation_stats(validation_results)

    return (
        stats,
        validation_stats,
    )


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    name="carbon_cpu_enriched_sequences",
    group_name="cpu",
    compute_kind="cpu",
    description=(
        "Streaming Carbon CPU pipeline. Reads the Hugging Face corpus "
        "incrementally, normalizes and validates bounded batches, applies "
        "NumPy CPU enrichment, and writes Parquet shards. Normalization, "
        "validation, and enrichment run in parallel worker processes."
    ),
)
def carbon_cpu_enriched_sequences(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
    hf_resource: HuggingFaceResource,
) -> dg.MaterializeResult:
    """Execute the complete streaming Carbon CPU pipeline."""

    # ------------------------------------------------------------------------
    # Streaming is mandatory.
    # ------------------------------------------------------------------------

    if not config.streaming:
        raise ValueError("carbon_cpu_enriched_sequences requires streaming=True.")

    context.log.info("Starting streaming Carbon CPU pipeline.")

    context.log.info(f"validation_level={config.validation_level!r}")

    context.log.info(f"batch_size={config.batch_size:,}")

    context.log.info(f"rows_per_shard={config.rows_per_shard:,}")

    context.log.info(f"compression={config.compression!r}")

    context.log.info(f"output_dir={config.output_dir!r}")

    cpu_workers = getattr(config, "cpu_workers", None)

    context.log.info(f"cpu_workers={cpu_workers!r} (None => os.cpu_count() - 1)")

    # ------------------------------------------------------------------------
    # Create lazy Hugging Face stream.
    # ------------------------------------------------------------------------

    stream = create_carbon_stream(
        config=config,
        hf_resource=hf_resource,
    )

    # ------------------------------------------------------------------------
    # Execute bounded streaming pipeline.
    # ------------------------------------------------------------------------

    stats, validation_stats = process_stream(
        stream=stream,
        output_dir=config.output_dir,
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        context=context,
        max_workers=cpu_workers,
    )

    # ------------------------------------------------------------------------
    # Global row-count invariants.
    # ------------------------------------------------------------------------

    if stats.rows_read != stats.rows_normalized:
        raise RuntimeError(
            "Normalization changed total row count: "
            f"input={stats.rows_read}, "
            f"normalized={stats.rows_normalized}"
        )

    if stats.rows_normalized != stats.rows_enriched:
        raise RuntimeError(
            "Enrichment changed total row count: "
            f"normalized={stats.rows_normalized}, "
            f"enriched={stats.rows_enriched}"
        )

    # ------------------------------------------------------------------------
    # Validation metadata.
    # ------------------------------------------------------------------------

    add_validation_metadata(
        context,
        validation_stats,
    )

    # ------------------------------------------------------------------------
    # Final metadata.
    # ------------------------------------------------------------------------

    context.log.info(
        "Streaming Carbon CPU pipeline completed: "
        f"rows={stats.rows_enriched:,}, "
        f"batches={stats.batches_processed:,}, "
        f"shards={stats.shards_written:,}"
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
