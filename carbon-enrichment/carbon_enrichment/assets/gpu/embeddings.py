"""
M4/M5 — Single-pass GPU enrichment: embeddings + likelihood stats.

Design doc #14: Carbon-3B is a stock LlamaForCausalLM. One forward pass
with output_hidden_states=True yields both `logits` (-> M4 likelihood
stats) and `hidden_states` (-> M5 embedding pooling). This module owns
that single forward pass and emits BOTH outputs as one Dagster
multi_asset -- `likelihood.py` does NOT run its own forward pass; it
consumes `carbon_likelihood_stats` from here. Running the model twice
(once per file) would silently violate #14 and double GPU cost for no
benefit, since both outputs come from the same call.

Pipeline:

    carbon_tokenized_corpus (token_ids, record_id, token_length)
                │
                ▼
          bucket by token_length (#1)
                │
                ▼
    ┌── model(ids, output_hidden_states=True) ──┐
    │        │                    │              │
    │     logits              hidden_states      │
    │        │                    │              │
    │   likelihood stats     pooled embeddings   │
    │   (extract, discard     (extract, keep)    │
    │    logits immediately)                     │
    └────────────────────────────────────────────┘
                │                    │
                ▼                    ▼
    carbon_likelihood_stats   carbon_embeddings
    (sharded Parquet)         (sharded Parquet)

Principles applied here (see design doc for full rationale):
- #1  bucket by token length, not raw bp length
- #2  GPU batching (bucket-internal) separate from Dagster partitioning
- #3  never retain logits beyond the batch -- extract stats, discard
- #4  model.eval() + torch.inference_mode() (eval() lives in
     resources/carbon.py; inference_mode is applied here per forward call)
- #6  sharded Parquet output, not one monolithic file
- #7  checkpoint/manifest at the shard level
- #8  record_id is the immutable join key -- asserted in/out
- #9  (downstream, M5.5) cluster on raw embeddings, never UMAP coords --
     not this module's concern, just don't violate it by e.g. reducing
     dimensionality here
- #12 provenance metadata per shard
- #13 keep corpus-wide ops (NN, clustering) out of this row-wise loop
- #16 long-sequence exception path is a defensive assertion, not a branch
- #19 torch.compile requires bucket-stable static shapes
- #23 OOM retry cascade feeds back into the static config as a signal
"""
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dagster as dg
import pyarrow as pa
import pyarrow.parquet as pq

from carbon_enrichment.assets.cpu.streaming import ParquetShardWriter
from carbon_enrichment.config import CarbonPipelineConfig
from carbon_enrichment.resources.carbon import (
    MAX_NATIVE_CONTEXT_TOKENS,
    BucketBatchConfig,
    CarbonModelResource,
)
from carbon_enrichment.schema import (
    EMBEDDING_COLUMNS,
    GPU_JOIN_KEY,
    LIKELIHOOD_COLUMNS,
    TOKEN_MASK_PADDING,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Placeholder single-bucket config until the real M3.5 calibration artifact
# (design doc #15) exists. This is NOT the M3.5 lookup table itself -- it
# is a deliberately conservative fallback so this module is runnable for
# staged validation step 1 (~100-1,000 rows, one bucket, #22) before M3.5
# has been benchmarked. Replace with the real per-bucket table before any
# run beyond staged-validation step 1.
_FALLBACK_BUCKET_CONFIG = [
    BucketBatchConfig(
        bucket_max_tokens=2048,
        batch_size=32,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
]

# Hidden-state layer used for embedding pooling. -1 = final layer. Design
# doc #9/#10 leaves layer choice open for M5.5 evaluation to inform; this
# is the current default, not a conclusion.
EMBEDDING_LAYER_INDEX = -1


# ============================================================================
# Stats
# ============================================================================


@dataclass
class GpuEnrichmentStats:
    """Cumulative counters and runtime telemetry for one GPU pass."""

    # Row accounting.
    rows_read: int = 0
    rows_enriched: int = 0
    rows_quarantined: int = 0
    rows_exceeding_native_context: int = 0

    # GPU execution counters.
    batches_processed: int = 0
    forward_passes: int = 0
    tokens_processed: int = 0

    # OOM telemetry, keyed by configured bucket size.
    oom_retries_by_bucket: dict[int, int] = field(default_factory=dict)

    # Runtime measurements.
    compile_setup_seconds: float | None = None
    warmup_seconds: float | None = None
    inference_seconds: float | None = None

    # Peak CUDA memory observed during GPU inference.
    peak_memory_allocated_bytes: int = 0
    peak_memory_reserved_bytes: int = 0

    # Output telemetry.
    embedding_shards_written: int = 0
    likelihood_shards_written: int = 0


def merge_gpu_stats(results: list[GpuEnrichmentStats]) -> GpuEnrichmentStats:
    """Merge additive GPU statistics from multiple execution results."""

    merged = GpuEnrichmentStats()

    for r in results:
        merged.rows_read += r.rows_read
        merged.rows_enriched += r.rows_enriched
        merged.rows_quarantined += r.rows_quarantined
        merged.rows_exceeding_native_context += r.rows_exceeding_native_context

        merged.batches_processed += r.batches_processed
        merged.forward_passes += r.forward_passes
        merged.tokens_processed += r.tokens_processed

        for bucket, count in r.oom_retries_by_bucket.items():
            merged.oom_retries_by_bucket[bucket] = (
                merged.oom_retries_by_bucket.get(bucket, 0) + count
            )

    return merged


# ============================================================================
# Output validation
# ============================================================================


def _validate_output_columns(
    batch: pa.RecordBatch,
    expected_columns: tuple[str, ...],
    *,
    output_name: str,
) -> None:
    """Validate the column contract of a GPU-derived output batch."""

    actual_columns = tuple(batch.schema.names)

    if actual_columns != expected_columns:
        raise ValueError(
            f"{output_name} schema mismatch: "
            f"expected columns {expected_columns}, "
            f"got {actual_columns}"
        )


def _validate_join_key(
    batch: pa.RecordBatch,
    *,
    output_name: str,
) -> None:
    """Ensure every GPU-derived output preserves the immutable join key."""

    if GPU_JOIN_KEY not in batch.schema.names:
        raise ValueError(
            f"{output_name} is missing required GPU join key {GPU_JOIN_KEY!r}"
        )


# ============================================================================
# Bucketing (design doc #1)
# ============================================================================


def _bucket_for(
    token_length: int,
    buckets: list[BucketBatchConfig],
) -> BucketBatchConfig:
    """Select the smallest bucket that fits token_length, largest as ceiling."""

    for bucket in sorted(buckets, key=lambda b: b.bucket_max_tokens):
        if token_length <= bucket.bucket_max_tokens:
            return bucket

    return buckets[-1]


def _group_by_bucket(
    record_batch: pa.RecordBatch,
    buckets: list[BucketBatchConfig],
) -> dict[int, pa.RecordBatch]:
    """Split one input RecordBatch into per-bucket RecordBatches.

    Grouping happens per-input-batch (bounded memory), not corpus-wide --
    #2's separation of GPU batching from Dagster partitioning means the
    bucket lookup itself stays a row-wise, streaming operation.
    """

    token_lengths = record_batch.column("token_length").to_pylist()

    bucket_indices = [_bucket_for(n, buckets).bucket_max_tokens for n in token_lengths]

    grouped: dict[int, pa.RecordBatch] = {}

    for bucket_max in sorted(set(bucket_indices)):
        mask = pa.array([b == bucket_max for b in bucket_indices])
        grouped[bucket_max] = record_batch.filter(mask)

    return grouped


# ============================================================================
# Padding — manual, since we now carry a parallel token_mask array
# ============================================================================

# tokenizer.pad() (the generic PreTrainedTokenizer method) only knows about
# input_ids/attention_mask -- it has no idea about the FNS token_mask
# convention tokenize_and_tag.py now writes per design doc #5 ("tokenize
# once, feed the same token stream to every downstream consumer"). Padding
# is therefore done by hand here, matching exactly what HybridDNATokenizer's
# own __call__ does internally (right-pad input_ids with pad_token_id,
# right-pad token_mask with TOKEN_MASK_PADDING) so the two arrays stay
# index-aligned.


def _pad_batch(
    token_ids_list: list[list[int]],
    token_mask_list: list[list[int]],
    pad_token_id: int,
    max_length: int,
) -> tuple[Any, Any]:
    """Right-pad input_ids and token_mask using the shared schema contract."""

    import torch

    padded_ids = []
    padded_masks = []

    for ids, mask in zip(token_ids_list, token_mask_list):
        pad_len = max_length - len(ids)

        if pad_len > 0:
            padded_ids.append(ids + [pad_token_id] * pad_len)
            padded_masks.append(mask + [TOKEN_MASK_PADDING] * pad_len)
        else:
            padded_ids.append(ids[:max_length])
            padded_masks.append(mask[:max_length])

    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(padded_masks, dtype=torch.int8),
    )


# ============================================================================
# Single forward pass — the core of #14
# ============================================================================


def _run_forward_pass(
    model: Any,
    tokenizer: Any,
    token_ids_column: pa.Array,
    token_mask_column: pa.Array,
    bucket: BucketBatchConfig,
) -> tuple[Any, Any, Any, Any]:
    """
    One model(ids, output_hidden_states=True) call for one GPU-batch.

    Returns (logits, hidden_states, input_ids, token_mask) as torch
    tensors -- token_mask is returned (not just attention_mask) because
    likelihood extraction and embedding pooling need different masking
    rules from the same FNS convention (see _extract_likelihood_stats /
    _extract_pooled_embeddings below), not just "real vs. padding".

    Caller is responsible for extracting required stats and discarding
    logits/hidden_states immediately afterward -- #3.
    """

    import torch

    ids_list = token_ids_column.to_pylist()
    mask_list = token_mask_column.to_pylist()

    input_ids, token_mask = _pad_batch(
        ids_list,
        mask_list,
        tokenizer.pad_token_id,
        bucket.bucket_max_tokens,
    )

    input_ids = input_ids.to(model.device)
    token_mask = token_mask.to(model.device)

    # attention_mask for the model call itself is just "not padding" --
    # padding is the only thing the model's attention needs to ignore.
    # The finer-grained content-type distinctions (special/oov/kmer/text)
    # live in token_mask and are applied downstream, not fed to the model.
    attention_mask = (token_mask != TOKEN_MASK_PADDING).to(torch.long)

    with torch.inference_mode():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    return outputs.logits, outputs.hidden_states, input_ids, token_mask


def _extract_likelihood_stats(
    logits: Any,
    input_ids: Any,
    token_mask: Any,
) -> list[dict[str, float]]:
    """
    Per-sequence likelihood summary stats from logits, extracted
    immediately -- logits are discarded by the caller right after this
    returns, per #3.

    Masking follows the FNS token_mask convention from tokenizer.py's
    docstring exactly, not plain attention_mask: positions are included
    only where token_mask > 0 (valid k-mer content -- 1..k for a partial
    trailing block, k for a full 6-mer). This excludes padding (-2),
    non-DNA BPE text (-1, shouldn't occur for pure-DNA input but excluded
    defensively), AND the <dna>/</dna>/<oov> special tokens (0) --
    matching exactly what the model's own FNS loss supervises during
    training.
    """

    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_content_mask = token_mask[:, 1:] > 0

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs.masked_fill(
        ~shift_content_mask,
        0.0,
    )

    seq_lengths = shift_content_mask.sum(dim=1).clamp(min=1)
    seq_log_prob_sum = token_log_probs.sum(dim=1)
    mean_log_prob = seq_log_prob_sum / seq_lengths
    perplexity = torch.exp(-mean_log_prob)

    results = []

    for i in range(logits.shape[0]):
        valid_log_probs = token_log_probs[i][shift_content_mask[i]]

        if valid_log_probs.numel() > 0:
            min_token_logprob = valid_log_probs.min().item()
            argmin_position = int(valid_log_probs.argmin().item())
            per_token_logprob_std = valid_log_probs.float().std(
                unbiased=False
            ).item()
        else:
            min_token_logprob = 0.0
            argmin_position = -1
            per_token_logprob_std = 0.0

        results.append(
            {
                "mean_log_prob": mean_log_prob[i].item(),
                "sum_log_prob": seq_log_prob_sum[i].item(),
                "perplexity": perplexity[i].item(),
                "supervised_position_count": int(seq_lengths[i].item()),
                "min_token_logprob": min_token_logprob,
                "argmin_position": argmin_position,
                "per_token_logprob_std": per_token_logprob_std,
            }
        )

    return results


def _extract_pooled_embeddings(
    hidden_states: tuple,
    token_mask: Any,
) -> list[list[float]]:
    """
    Mean-pool the selected hidden-state layer over real (non-pad) tokens.

    Deliberately a DIFFERENT mask rule than _extract_likelihood_stats:
    pooling includes every non-padding position (token_mask != -2) --
    the <dna>/</dna> boundary special tokens (mask == 0) ARE included
    here, since they carry structural/positional signal useful to a
    representation even though they aren't nucleotide content and
    correctly should NOT count toward a likelihood average.
    """

    layer = hidden_states[EMBEDDING_LAYER_INDEX]

    real_token_mask = (token_mask != TOKEN_MASK_PADDING).unsqueeze(-1).to(layer.dtype)

    summed = (layer * real_token_mask).sum(dim=1)
    counts = real_token_mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts

    return pooled.float().cpu().tolist()


# ============================================================================
# OOM retry cascade (#23)
# ============================================================================


def _run_bucket_batch_with_oom_retry(
    model: Any,
    tokenizer: Any,
    record_batch: pa.RecordBatch,
    bucket: BucketBatchConfig,
    stats: GpuEnrichmentStats,
) -> tuple[pa.RecordBatch, pa.RecordBatch, pa.RecordBatch]:
    """
    Run the forward pass with OOM backoff: full batch -> 1/2 -> 1/4 ->
    singleton -> quarantine.

    Every OOM is logged against `oom_retries_by_bucket` -- a signal to
    revise the M3.5 lookup table for next run, not just a per-run retry
    tax (#23).

    Returns (embeddings_batch, likelihood_batch, quarantined_batch).
    Quarantined rows are tracked, never silently dropped.
    """

    import torch

    fractions = [1, 2, 4, record_batch.num_rows]

    n = record_batch.num_rows

    offset = 0
    embedding_rows: list[dict[str, Any]] = []
    likelihood_rows: list[dict[str, Any]] = []
    quarantined_record_ids: list[Any] = []

    while offset < n:
        remaining = n - offset
        succeeded = False

        for divisor in fractions:
            chunk_size = max(1, remaining // divisor) if divisor != n else 1

            chunk = record_batch.slice(
                offset,
                min(chunk_size, remaining),
            )

            token_lengths = chunk.column("token_length").to_pylist()

            if any(tl > MAX_NATIVE_CONTEXT_TOKENS for tl in token_lengths):
                # Defensive assertion, not an engineered branch (#16) --
                # every observed sequence fits native context; anything
                # over is routed to quarantine rather than chunked further.
                stats.rows_exceeding_native_context += sum(
                    1 for tl in token_lengths if tl > MAX_NATIVE_CONTEXT_TOKENS
                )

                quarantined_record_ids.extend(chunk.column("record_id").to_pylist())

                offset += chunk.num_rows
                succeeded = True
                break

            try:
                (
                    logits,
                    hidden_states,
                    padded_ids,
                    padded_token_mask,
                ) = _run_forward_pass(
                    model,
                    tokenizer,
                    chunk.column("token_ids"),
                    chunk.column("token_mask"),
                    bucket,
                )

                # Count only successful model invocations. An OOM attempt
                # below is not a completed forward pass.
                stats.forward_passes += 1

                # Count actual non-padding tokens rather than
                # batch_size * bucket_max_tokens. Padding can be substantial
                # within a bucket.
                stats.tokens_processed += int(
                    (padded_token_mask != TOKEN_MASK_PADDING).sum().item()
                )

                likelihood_stats = _extract_likelihood_stats(
                    logits,
                    padded_ids,
                    padded_token_mask,
                )

                embeddings = _extract_pooled_embeddings(
                    hidden_states,
                    padded_token_mask,
                )

                # #3: discard logits/hidden_states immediately -- nothing
                # beyond this point holds a reference to either.
                del logits, hidden_states

                record_ids = chunk.column("record_id").to_pylist()

                for rid, emb, like in zip(
                    record_ids,
                    embeddings,
                    likelihood_stats,
                ):
                    embedding_norm = math.sqrt(sum(x * x for x in emb))

                    embedding_rows.append(
                        {
                            GPU_JOIN_KEY: rid,
                            "embedding": emb,
                            "embedding_norm": embedding_norm,
                        }
                    )

                    likelihood_rows.append(
                        {
                            GPU_JOIN_KEY: rid,
                            **like,
                        }
                    )

                offset += chunk.num_rows
                succeeded = True
                break

            except torch.cuda.OutOfMemoryError:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()

                torch.cuda.empty_cache()

                stats.oom_retries_by_bucket[bucket.bucket_max_tokens] = (
                    stats.oom_retries_by_bucket.get(
                        bucket.bucket_max_tokens,
                        0,
                    )
                    + 1
                )

                logger.warning(
                    "GPU OOM: "
                    f"bucket={bucket.bucket_max_tokens}, "
                    f"requested_batch={chunk.num_rows}, "
                    f"allocated={allocated / 1e9:.2f}GB, "
                    f"reserved={reserved / 1e9:.2f}GB"
                )

                continue

        if not succeeded:
            # Exhausted every fraction down to singleton and still OOM'd --
            # quarantine this one row as a tracked exception, never a
            # silent drop (#23, carried from v1 §26).
            quarantined_record_ids.append(
                record_batch.column("record_id")[offset].as_py()
            )

            offset += 1

    stats.rows_quarantined += len(quarantined_record_ids)

    embedding_batch = (
        pa.RecordBatch.from_pylist(embedding_rows)
        if embedding_rows
        else pa.RecordBatch.from_pylist(
            [],
            schema=pa.schema([]),
        )
    )

    likelihood_batch = (
        pa.RecordBatch.from_pylist(likelihood_rows)
        if likelihood_rows
        else pa.RecordBatch.from_pylist(
            [],
            schema=pa.schema([]),
        )
    )

    if embedding_batch.num_rows:
        _validate_output_columns(
            embedding_batch,
            EMBEDDING_COLUMNS,
            output_name="carbon_embeddings",
        )

        _validate_join_key(
            embedding_batch,
            output_name="carbon_embeddings",
        )

    if likelihood_batch.num_rows:
        _validate_output_columns(
            likelihood_batch,
            LIKELIHOOD_COLUMNS,
            output_name="carbon_likelihood_stats",
        )

        _validate_join_key(
            likelihood_batch,
            output_name="carbon_likelihood_stats",
        )

    quarantine_batch = (
        pa.RecordBatch.from_pylist(
            [{"record_id": rid} for rid in quarantined_record_ids]
        )
        if quarantined_record_ids
        else pa.RecordBatch.from_pylist(
            [],
            schema=pa.schema([]),
        )
    )

    return embedding_batch, likelihood_batch, quarantine_batch


# ============================================================================
# Batch reading — Arrow-native, tokenized shards
# ============================================================================


def iter_tokenized_batches(
    input_dir: str | Path,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Stream RecordBatches directly from tokenize_and_tag's output shards."""

    input_dir = Path(input_dir)

    shard_paths = sorted(input_dir.glob("shard-*.parquet"))

    if not shard_paths:
        raise FileNotFoundError(f"No tokenized shards found in {input_dir}")

    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)

        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            yield record_batch


# ============================================================================
# Processing loop
# ============================================================================


def process_gpu_enrichment(
    input_dir: str | Path,
    embeddings_output_dir: str | Path,
    likelihood_output_dir: str | Path,
    quarantine_output_dir: str | Path,
    *,
    batch_size: int,
    rows_per_shard: int,
    compression: str,
    carbon_resource: CarbonModelResource,
    buckets: list[BucketBatchConfig] | None = None,
    context: dg.AssetExecutionContext | None = None,
) -> GpuEnrichmentStats:
    """
    Single-pass GPU enrichment: embeddings + likelihood stats, bucketed,
    checkpointed, OOM-resilient.

    Deliberately single-process (no ProcessPoolExecutor here, unlike the
    CPU stage) -- the GPU itself is the shared resource being scheduled;
    fanning out across worker processes would just contend for the same
    device. #2's separation of GPU-internal batching from Dagster
    partitioning is what provides parallelism here (across partitions/runs),
    not intra-process forking.

    Runtime instrumentation records:
    - raw input batches processed
    - successful model forward passes
    - actual non-padding tokens processed
    - inference elapsed time and tokens/sec
    - OOM retries by bucket
    - peak CUDA allocated/reserved memory
    - row reconciliation
    """

    import torch

    buckets = buckets or _FALLBACK_BUCKET_CONFIG

    model = carbon_resource.model
    tokenizer = carbon_resource.tokenizer

    stats = GpuEnrichmentStats()

    # Reset peak-memory counters after model loading so the reported peak
    # reflects the GPU enrichment workload rather than model initialization.
    torch.cuda.reset_peak_memory_stats()

    run_start = time.perf_counter()
    inference_start = run_start
    last_heartbeat = run_start

    with (
        ParquetShardWriter(
            embeddings_output_dir,
            rows_per_shard,
            compression,
        ) as emb_writer,
        ParquetShardWriter(
            likelihood_output_dir,
            rows_per_shard,
            compression,
        ) as like_writer,
        ParquetShardWriter(
            quarantine_output_dir,
            rows_per_shard,
            compression,
        ) as quarantine_writer,
    ):
        for raw_batch in iter_tokenized_batches(
            input_dir,
            batch_size,
        ):
            if raw_batch.num_rows == 0:
                continue

            stats.rows_read += raw_batch.num_rows

            grouped = _group_by_bucket(
                raw_batch,
                buckets,
            )

            for bucket_max, bucket_batch in grouped.items():
                bucket = next(b for b in buckets if b.bucket_max_tokens == bucket_max)

                # Compilation is intentionally NOT wired here yet.
                # When M3.5 compilation is enabled, the model selected here
                # will become bucket-specific and the corresponding compile
                # / warm-up timings will populate stats.
                model_for_bucket = model

                (
                    emb_batch,
                    like_batch,
                    quarantine_batch,
                ) = _run_bucket_batch_with_oom_retry(
                    model_for_bucket,
                    tokenizer,
                    bucket_batch,
                    bucket,
                    stats,
                )

                if emb_batch.num_rows:
                    emb_writer.write_batch(emb_batch)

                if like_batch.num_rows:
                    like_writer.write_batch(like_batch)

                if quarantine_batch.num_rows:
                    quarantine_writer.write_batch(quarantine_batch)

                stats.rows_enriched += emb_batch.num_rows

            stats.batches_processed += 1

            now = time.perf_counter()

            if context is not None and now - last_heartbeat >= 30:
                elapsed = now - run_start
                inference_elapsed = now - inference_start

                tokens_per_second = (
                    stats.tokens_processed / inference_elapsed
                    if inference_elapsed > 0
                    else 0.0
                )

                context.log.info(
                    "GPU enrichment progress: "
                    f"rows_read={stats.rows_read:,}, "
                    f"rows_enriched={stats.rows_enriched:,}, "
                    f"rows_quarantined={stats.rows_quarantined:,}, "
                    f"batches={stats.batches_processed:,}, "
                    f"forward_passes={stats.forward_passes:,}, "
                    f"tokens={stats.tokens_processed / 1e6:.2f}M, "
                    f"tokens_per_sec={tokens_per_second:,.0f}, "
                    f"elapsed={elapsed:.1f}s"
                )

                last_heartbeat = now

        stats.embedding_shards_written = emb_writer.shards_written
        stats.likelihood_shards_written = like_writer.shards_written

    stats.inference_seconds = time.perf_counter() - inference_start

    stats.peak_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    stats.peak_memory_reserved_bytes = torch.cuda.max_memory_reserved()

    # Correctness invariant: every input row must have exactly one terminal
    # outcome -- enriched or quarantined.
    reconciled_rows = stats.rows_enriched + stats.rows_quarantined

    if reconciled_rows != stats.rows_read:
        raise RuntimeError(
            "GPU enrichment row reconciliation failed: "
            f"rows_read={stats.rows_read}, "
            f"rows_enriched={stats.rows_enriched}, "
            f"rows_quarantined={stats.rows_quarantined}"
        )

    tokens_per_second = (
        stats.tokens_processed / stats.inference_seconds
        if stats.inference_seconds > 0
        else 0.0
    )

    if context is not None:
        context.log.info(
            "GPU enrichment complete: "
            f"elapsed={stats.inference_seconds:.2f}s, "
            f"tokens={stats.tokens_processed / 1e6:.2f}M, "
            f"tokens_per_sec={tokens_per_second:,.0f}, "
            f"forward_passes={stats.forward_passes:,}, "
            f"peak_allocated="
            f"{stats.peak_memory_allocated_bytes / 1e9:.2f}GB, "
            f"peak_reserved="
            f"{stats.peak_memory_reserved_bytes / 1e9:.2f}GB, "
            f"rows_read={stats.rows_read:,}, "
            f"rows_enriched={stats.rows_enriched:,}, "
            f"rows_quarantined={stats.rows_quarantined:,}"
        )

    return stats


# ============================================================================
# Dagster multi_asset — single forward pass, two outputs (#14)
# ============================================================================


@dg.multi_asset(
    outs={
        "carbon_embeddings": dg.AssetOut(
            description=(
                "Mean-pooled raw hidden-state embeddings, one row per "
                "record_id (#9: raw only, never UMAP coords)."
            )
        ),
        "carbon_likelihood_stats": dg.AssetOut(
            description=(
                "Per-sequence log-prob/perplexity stats derived from "
                "logits, discarded immediately after extraction (#3)."
            )
        ),
    },
    deps=["carbon_tokenized_corpus"],
    group_name="gpu",
    compute_kind="gpu",
)
def carbon_gpu_enrichment(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
    carbon: CarbonModelResource,
):
    """
    Single model(ids, output_hidden_states=True) forward pass per bucketed
    batch, emitting both embeddings and likelihood stats (#14). Bucketed
    by token length (#1), OOM-resilient (#23), record_id preserved as the
    immutable join key (#8) and asserted below.
    """

    context.log.info(f"model_checkpoint={carbon.model_checkpoint!r}")
    context.log.info(f"tokenizer_revision={carbon.tokenizer_revision!r}")

    stats = process_gpu_enrichment(
        input_dir=config.tokenized_output_dir,
        embeddings_output_dir=(f"{config.tokenized_output_dir}_embeddings"),
        likelihood_output_dir=(f"{config.tokenized_output_dir}_likelihood"),
        quarantine_output_dir=(f"{config.tokenized_output_dir}_quarantine"),
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        carbon_resource=carbon,
        context=context,
    )

    if stats.oom_retries_by_bucket:
        context.log.warning(
            "OOM retries observed — revise the M3.5 bucket/batch "
            "lookup table for next run (#23): "
            f"{stats.oom_retries_by_bucket}"
        )

    if stats.rows_quarantined:
        context.log.warning(f"{stats.rows_quarantined:,} rows quarantined.")

    tokens_per_second = (
        stats.tokens_processed / stats.inference_seconds
        if stats.inference_seconds
        else None
    )

    metadata = {
        "rows_read": stats.rows_read,
        "rows_enriched": stats.rows_enriched,
        "rows_quarantined": stats.rows_quarantined,
        "rows_exceeding_native_context": (stats.rows_exceeding_native_context),
        "oom_retries_by_bucket": dg.MetadataValue.json(stats.oom_retries_by_bucket),
        "batches_processed": stats.batches_processed,
        "forward_passes": stats.forward_passes,
        "tokens_processed": stats.tokens_processed,
        "tokens_per_second": tokens_per_second,
        "inference_seconds": stats.inference_seconds,
        "compile_setup_seconds": stats.compile_setup_seconds,
        "warmup_seconds": stats.warmup_seconds,
        "peak_memory_allocated_bytes": (stats.peak_memory_allocated_bytes),
        "peak_memory_reserved_bytes": (stats.peak_memory_reserved_bytes),
        "model_checkpoint": carbon.model_checkpoint,
        "tokenizer_revision": carbon.tokenizer_revision,
        "kernel_revision": carbon.kernel_revision,
    }

    yield dg.MaterializeResult(
        None,
        output_name="carbon_embeddings",
        metadata={
            **metadata,
            "parquet_shards": stats.embedding_shards_written,
        },
    )

    yield dg.MaterializeResult(
        None,
        output_name="carbon_likelihood_stats",
        metadata={
            **metadata,
            "parquet_shards": stats.likelihood_shards_written,
        },
    )