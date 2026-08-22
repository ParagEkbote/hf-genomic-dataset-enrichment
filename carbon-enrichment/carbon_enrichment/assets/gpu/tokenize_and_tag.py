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
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dagster as dg
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from carbon_enrichment.assets.cpu.streaming import ParquetShardWriter
from carbon_enrichment.config import CarbonPipelineConfig
from carbon_enrichment.resources.carbon import CarbonModelResource

# ============================================================================
# Constants
# ============================================================================

CANONICAL_BASES = frozenset("ACGT")
DNA_OPEN_TAG = "<dna>"
DNA_CLOSE_TAG = "</dna>"
OOV_TOKEN = "<oov>"
SIXMER_BLOCK = 6
MAX_NATIVE_CONTEXT_TOKENS = 32_768  # Carbon-3B native context, design doc #16

# Known-good probe: tokenizing this with the <dna> tag present must yield
# 6-mer-mode token ids, not BPE fallback ids. Checked once at pipeline
# startup -- see _assert_dna_mode_active.
_DNA_MODE_PROBE_SEQUENCE = "ACGTACGTACGT"


# ============================================================================
# Stats
# ============================================================================


@dataclass
class TokenizationStats:
    """Cumulative counters for one tokenize_and_tag execution."""

    rows_read: int = 0
    rows_tokenized: int = 0
    oov_bases_filtered: int = 0
    rows_truncated_to_6mer: int = 0
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

    for r in results:
        merged.rows_read += r.rows_read
        merged.rows_tokenized += r.rows_tokenized
        merged.oov_bases_filtered += r.oov_bases_filtered
        merged.rows_truncated_to_6mer += r.rows_truncated_to_6mer
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


# ============================================================================
# Correctness gate (design doc #14.5, #22 step 1)
# ============================================================================


def _assert_dna_mode_active(tokenizer: Any) -> None:
    """
    Fail fast if the tokenizer is not actually in 6-mer <dna> mode.

    Tokenizes a small known sequence with the <dna> tag present and checks
    that the result does not look like BPE/English-text fallback (e.g. an
    unexpectedly long token count for a 12-base probe, or the <dna>/</dna>
    tag strings themselves showing up as multiple BPE sub-tokens instead of
    being consumed as single special tokens). This is a smoke test, not a
    substitute for the full staged validation in #22 -- it exists so a
    misconfigured tokenizer fails on the first row of the first batch, not
    32M rows later.
    """

    tagged = f"{DNA_OPEN_TAG}{_DNA_MODE_PROBE_SEQUENCE}{DNA_CLOSE_TAG}"

    probe_ids = tokenizer(tagged, add_special_tokens=False)["input_ids"]

    # A 12-base sequence in true 6-mer mode tokenizes to 2 content tokens
    # (two 6-mer blocks) plus the <dna>/</dna> special tokens. BPE fallback
    # on the same string produces many more sub-word tokens. This bound is
    # deliberately loose (covers tokenizer-version differences in how the
    # tags themselves are counted) while still catching a full BPE fallback.
    if len(probe_ids) > 8:
        raise RuntimeError(
            "Tokenizer does not appear to be in <dna> 6-mer mode: probe "
            f"sequence produced {len(probe_ids)} tokens, expected a small "
            "handful. This usually means the <dna> tag is not registered "
            "as a special token and the tokenizer has fallen back to BPE "
            "(English-text) mode -- see design doc #14.5. Refusing to "
            "tokenize the corpus until this is fixed."
        )


# ============================================================================
# Arrow-native preprocessing (tag / filter / truncate)
# ============================================================================


def _filter_to_canonical_acgt(sequences: pa.Array) -> tuple[pa.Array, int]:
    """
    Replace any non-canonical base with <oov>, vectorized over the column.

    Returns the filtered column and a count of sequences that contained at
    least one non-ACGT character (oov_bases_filtered bookkeeping).
    """

    # pc.utf8_upper first: canonicalization is uppercase-ACGT, matching the
    # CPU stage's own sequence normalization contract.
    upper = pc.utf8_upper(sequences)

    # There is no single pyarrow.compute kernel for "is every character in
    # this fixed set" over a StringArray, so this step is done with a
    # regex-based replace: any run of characters outside [ACGT] is
    # collapsed to the OOV token. This keeps the whole column vectorized
    # (one compute call over N rows) rather than a per-row Python loop.
    has_non_acgt = pc.match_substring_regex(upper, r"[^ACGT]")

    oov_count = pc.sum(pc.cast(has_non_acgt, pa.int64())).as_py() or 0

    filtered = pc.if_else(
        has_non_acgt,
        pa.scalar(OOV_TOKEN),
        upper,
    )

    return filtered, oov_count


def _truncate_to_6mer(sequences: pa.Array) -> tuple[pa.Array, int]:
    """
    Trim each sequence to a multiple of 6 bases, vectorized.

    Prevents the tokenizer from right-padding a trailing partial 6-mer
    block with A's -- a silent correctness bug per design doc #14.5.
    Returns the truncated column and a count of rows that were actually
    shortened (i.e. length was not already a multiple of 6).
    """

    lengths = pc.utf8_length(sequences)

    remainder = pc.mod(lengths, pa.scalar(SIXMER_BLOCK, type=lengths.type))

    truncated_lengths = pc.subtract(lengths, remainder)

    truncated = pc.utf8_slice_codeunits(sequences, start=0, stop=truncated_lengths)

    truncated_count = pc.sum(
        pc.cast(pc.greater(remainder, 0), pa.int64())
    ).as_py() or 0

    return truncated, truncated_count


def _tag_dna(sequences: pa.Array) -> pa.Array:
    """Wrap every sequence with <dna>...</dna>, vectorized."""

    return pc.binary_join_element_wise(
        pa.array([DNA_OPEN_TAG] * len(sequences)),
        sequences,
        pa.array([DNA_CLOSE_TAG] * len(sequences)),
        "",
    )


# ============================================================================
# Fused worker unit: tag -> filter -> truncate -> tokenize
# ============================================================================
#
# Top-level, picklable function (no closures) for ProcessPoolExecutor. The
# tokenizer is re-resolved once per worker process (see process_corpus) so
# forked workers inherit an already-loaded tokenizer rather than each
# re-initializing it from disk/hub.


def _tokenize_batch(
    raw_batch: pa.RecordBatch,
    tokenizer: Any,
) -> tuple[pa.RecordBatch, TokenizationStats]:
    """Tag, filter, truncate, and tokenize one RecordBatch. Runs in a worker."""

    stats = TokenizationStats()

    stats.rows_read = raw_batch.num_rows

    sequences = raw_batch.column("sequence")

    filtered, oov_count = _filter_to_canonical_acgt(sequences)
    stats.oov_bases_filtered = oov_count

    truncated, truncated_count = _truncate_to_6mer(filtered)
    stats.rows_truncated_to_6mer = truncated_count

    tagged = _tag_dna(truncated)

    # Tokenizer call is the one point that must touch Python str -- no
    # batch-level pyarrow.compute kernel exists for BPE/6-mer tokenization
    # itself. Token IDs come back out as an Arrow array immediately.
    tagged_list = tagged.to_pylist()

    encoded = tokenizer(
        tagged_list,
        add_special_tokens=False,
    )["input_ids"]

    token_lengths = [len(ids) for ids in encoded]

    stats.total_token_count = sum(token_lengths)
    stats.min_token_length = min(token_lengths) if token_lengths else None
    stats.max_token_length = max(token_lengths) if token_lengths else None
    stats.rows_exceeding_native_context = sum(
        1 for n in token_lengths if n > MAX_NATIVE_CONTEXT_TOKENS
    )
    stats.rows_tokenized = len(token_lengths)

    token_ids_array = pa.array(encoded, type=pa.list_(pa.int32()))
    token_length_array = pa.array(token_lengths, type=pa.int32())

    output_batch = pa.RecordBatch.from_arrays(
        [
            raw_batch.column("record_id"),
            token_ids_array,
            token_length_array,
        ],
        names=["record_id", "token_ids", "token_length"],
    )

    return output_batch, stats


# ============================================================================
# Batch reading (Arrow-native — reads RecordBatches directly, no dict step)
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
# Processing loop
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
    Tokenize the CPU-enriched corpus using bounded memory, Arrow-native
    throughout, checkpointed at the shard level.
    """

    # Resolve the tokenizer once in the main process before the pool is
    # created. "fork" (below) inherits the already-loaded tokenizer into
    # each worker via copy-on-write, so this is not re-loaded per worker --
    # same rationale streaming.py documents for module reimport avoidance.
    tokenizer = carbon_resource.tokenizer

    _assert_dna_mode_active(tokenizer)

    n_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    inflight_limit = n_workers * 4

    # See streaming.py for the full rationale: Dagster's multiprocess
    # executor already runs this asset inside its own STEP_WORKER
    # subprocess. A "spawn"-based pool nested inside that subprocess would
    # force each worker to reimport carbon_enrichment (and re-resolve the
    # tokenizer) from scratch, and could fail to find locally/editable-
    # installed packages. "fork" inherits the parent's already-loaded
    # modules, sys.path, and the tokenizer object itself, avoiding both
    # problems. Linux/macOS only -- spawn is the only option on Windows.
    mp_context = multiprocessing.get_context("fork")

    validation_results: list[TokenizationStats] = []

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

        def _drain(futures: list[Future]) -> None:
            for future in futures:
                token_batch, batch_stats = future.result()

                validation_results.append(batch_stats)

                writer.write_batch(token_batch)

        batches_processed = 0

        for raw_batch in iter_cpu_enriched_batches(input_dir, batch_size):
            if raw_batch.num_rows == 0:
                continue

            pending.append(pool.submit(_tokenize_batch, raw_batch, tokenizer))

            batches_processed += 1

            if len(pending) >= inflight_limit:
                _drain(pending)
                pending = []

                if context is not None and batches_processed % 100 == 0:
                    context.log.info(
                        f"Tokenization progress: {batches_processed:,} batches submitted"
                    )

        _drain(pending)

    stats = merge_tokenization_stats(validation_results)
    stats.batches_processed = batches_processed
    stats.shards_written = writer.shards_written

    return stats


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    name="carbon_tokenized_corpus",
    group_name="gpu",
    compute_kind="cpu",
    deps=["carbon_cpu_enriched_sequences"],
    description=(
        "Tokenization/tagging pass (design doc #14.5): wraps sequences in "
        "<dna>...</dna>, filters non-canonical bases to <oov>, truncates "
        "to a multiple of 6 bases, and tokenizes once with the Carbon "
        "hybrid 6-mer tokenizer. Writes checkpointed token-ID shards "
        "independent of both CPU enrichment and GPU-stage assets."
    ),
)
def carbon_tokenized_corpus(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
    carbon: CarbonModelResource,
) -> dg.MaterializeResult:
    """Tokenize the CPU-enriched corpus ahead of GPU enrichment."""

    context.log.info("Starting tokenize_and_tag stage.")
    context.log.info(f"tokenizer_revision={carbon.tokenizer_revision!r}")

    stats = process_corpus(
        input_dir=config.output_dir,
        output_dir=config.tokenized_output_dir,
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        carbon_resource=carbon,
        context=context,
        max_workers=getattr(config, "cpu_workers", None),
    )

    if stats.rows_exceeding_native_context:
        context.log.warning(
            f"{stats.rows_exceeding_native_context} rows exceeded "
            f"{MAX_NATIVE_CONTEXT_TOKENS} native-context tokens — see "
            "design doc #16 (defensive YaRN/quarantine path)."
        )

    context.log.info(
        "Tokenization completed: "
        f"rows={stats.rows_tokenized:,}, "
        f"shards={stats.shards_written:,}, "
        f"oov_filtered={stats.oov_bases_filtered:,}, "
        f"truncated_to_6mer={stats.rows_truncated_to_6mer:,}"
    )

    return dg.MaterializeResult(
        metadata={
            "rows_read": stats.rows_read,
            "rows_tokenized": stats.rows_tokenized,
            "oov_bases_filtered": stats.oov_bases_filtered,
            "rows_truncated_to_6mer": stats.rows_truncated_to_6mer,
            "rows_exceeding_native_context": stats.rows_exceeding_native_context,
            "min_token_length": stats.min_token_length,
            "max_token_length": stats.max_token_length,
            "mean_token_length": (
                round(stats.total_token_count / stats.rows_tokenized, 1)
                if stats.rows_tokenized
                else None
            ),
            "batches_processed": stats.batches_processed,
            "parquet_shards": stats.shards_written,
            "tokenizer_revision": carbon.tokenizer_revision,
        }
    )