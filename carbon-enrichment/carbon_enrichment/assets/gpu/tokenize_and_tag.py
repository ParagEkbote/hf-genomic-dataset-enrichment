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
from carbon_enrichment.schema import (
    GPU_JOIN_KEY,
    TOKENIZED_CORPUS_COLUMNS,
)

# ============================================================================
# Constants
# ============================================================================

CANONICAL_BASES = frozenset("ACGT")
DNA_OPEN_TAG = "<dna>"
DNA_CLOSE_TAG = "</dna>"
OOV_TOKEN = "<oov>"
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


def _validate_tokenized_output(
    batch: pa.RecordBatch,
) -> None:
    """Validate the schema contract of the tokenized GPU-stage asset."""

    actual_columns = tuple(batch.schema.names)

    if actual_columns != TOKENIZED_CORPUS_COLUMNS:
        raise ValueError(
            "carbon_tokenized_corpus schema mismatch: "
            f"expected columns {TOKENIZED_CORPUS_COLUMNS}, "
            f"got {actual_columns}"
        )

    if GPU_JOIN_KEY not in batch.schema.names:
        raise ValueError(
            "carbon_tokenized_corpus is missing the required "
            f"GPU join key {GPU_JOIN_KEY!r}"
        )


# ============================================================================
# Correctness gate (design doc #14.5, #22 step 1)
# ============================================================================


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

    expected_length = expected_kmer_count + 2  # + dna_begin, dna_end

    correct_structure = (
        len(probe_ids) == expected_length
        and probe_ids[0] == tokenizer.dna_begin_token_id
        and probe_ids[-1] == tokenizer.dna_end_token_id
        and tokenizer.oov_token_id not in probe_ids
    )

    if not correct_structure:
        raise RuntimeError(
            "Tokenizer does not appear to be in <dna> 6-mer mode: probe "
            f"sequence produced {probe_ids!r}, expected "
            f"[dna_begin_token_id={tokenizer.dna_begin_token_id}, "
            f"{expected_kmer_count} kmer id(s), "
            f"dna_end_token_id={tokenizer.dna_end_token_id}] with no "
            "<oov> tokens. This usually means the <dna> tag was not "
            "wrapped correctly or the tokenizer fell back to plain BPE "
            "-- see design doc #14.5. Refusing to tokenize the corpus "
            "until this is fixed."
        )


# ============================================================================
# Arrow-native preprocessing (tag only — see note below)
# ============================================================================
#
# No upstream OOV filtering or 6-mer truncation here. Both were removed
# after reading tokenizer.py: HybridDNATokenizer._process_dna_sequence
# already does per-kmer OOV detection (only the offending 6-mer becomes
# <oov>, not the whole sequence -- our earlier whole-sequence regex
# collapse was a real bug) and right-pads a trailing partial block with
# 'A' while tracking valid_length via token_mask (not a silent bug, a
# deliberate FNS-supervision mechanism -- pre-truncating discarded real
# trailing bases the tokenizer is designed to handle correctly). See
# design doc #14.5's revised understanding and _tokenize_batch below.


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

    if GPU_JOIN_KEY not in raw_batch.schema.names:
        raise ValueError(f"Input batch is missing required join key {GPU_JOIN_KEY!r}")

    sequences = raw_batch.column("sequence")

    tagged = _tag_dna(sequences)

    # Tokenizer call is the one point that must touch Python str -- no
    # batch-level pyarrow.compute kernel exists for BPE/6-mer tokenization
    # itself. Token IDs and token_mask come back out as Arrow arrays
    # immediately.
    #
    # OOV and partial-trailing-kmer handling are NOT done upstream here --
    # HybridDNATokenizer._process_dna_sequence already does both correctly
    # on its own: non-ACGT bases invalidate only the specific 6-mer they
    # fall in (not the whole sequence), and a trailing partial block is
    # right-padded with 'A' with valid_length tracked via token_mask, not
    # silently discarded. An upstream sequence-level OOV filter or 6-mer
    # truncation would either destroy valid k-mers elsewhere in the
    # sequence or throw away real trailing bases the tokenizer is designed
    # to handle. See design doc #14.5 and the tokenizer.py docstring for
    # the token_mask convention consumed downstream by embeddings.py.
    tagged_list = tagged.to_pylist()

    encoded = tokenizer(
        tagged_list,
        add_special_tokens=False,
        return_token_mask=True,
    )

    token_ids_batch = encoded["input_ids"]
    token_mask_batch = encoded["token_mask"]

    token_lengths = [len(ids) for ids in token_ids_batch]

    oov_count = sum(
        sum(1 for tid in ids if tid == tokenizer.oov_token_id)
        for ids in token_ids_batch
    )

    stats.oov_bases_filtered = oov_count
    stats.total_token_count = sum(token_lengths)
    stats.min_token_length = min(token_lengths) if token_lengths else None
    stats.max_token_length = max(token_lengths) if token_lengths else None
    stats.rows_exceeding_native_context = sum(
        1 for n in token_lengths if n > MAX_NATIVE_CONTEXT_TOKENS
    )
    stats.rows_tokenized = len(token_lengths)

    token_ids_array = pa.array(token_ids_batch, type=pa.list_(pa.int32()))
    token_mask_array = pa.array(token_mask_batch, type=pa.list_(pa.int8()))
    token_length_array = pa.array(token_lengths, type=pa.int32())

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
    deps=["carbon_pilot_corpus"],
    description=(
        "Tokenization/tagging pass (design doc #14.5): wraps sequences in "
        "<dna>...</dna> and tokenizes once with the Carbon hybrid 6-mer "
        "tokenizer. OOV detection and partial trailing-k-mer handling are "
        "performed by HybridDNATokenizer so valid k-mers are preserved. "
        "Writes checkpointed token-ID shards independent of both CPU "
        "enrichment and GPU-stage assets."
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
        input_dir=config.pilot_output_dir,
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
        f"oov_filtered={stats.oov_bases_filtered:,}"
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
                round(stats.total_token_count / stats.rows_tokenized, 1)
                if stats.rows_tokenized
                else None
            ),
            "batches_processed": stats.batches_processed,
            "parquet_shards": stats.shards_written,
            "tokenizer_revision": carbon.tokenizer_revision,
        }
    )
