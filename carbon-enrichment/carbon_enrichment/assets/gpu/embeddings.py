"""
M4/M5 — Single-pass GPU enrichment: embeddings + likelihood stats.

Design doc #14: Carbon-3B is a stock LlamaForCausalLM. One forward pass
yields both `logits` (-> M4 likelihood stats via memory-safe logsumexp)
and `last_hidden_state` (-> M5 embedding pooling). This module owns that
single forward pass and emits BOTH outputs as one Dagster multi_asset --
`likelihood.py` does NOT run its own forward pass; it consumes
`carbon_likelihood_stats` from here. Running the model twice (once per file)
would silently violate #14 and double GPU cost for no benefit, since both
outputs come from the same call.

Pipeline:

    carbon_tokenized_corpus (token_ids, record_id, token_length)
                │
                ▼
          bucket by token_length (#1)
                │
                ▼
    ┌── model(ids, output_hidden_states=False) ──┐
    │        │                                    │
    │     logits                          last_hidden_state
    │        │                                    │
    │   likelihood stats                  pooled embeddings
    │   (extract, discard                 (extract, keep)
    │    logits immediately)                      │
    └─────────────────────────────────────────────┘
                │                                 │
                ▼                                 ▼
    carbon_likelihood_stats               carbon_embeddings
    (sharded Parquet)                     (sharded Parquet)

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
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Safeguard against VRAM virtual address fragmentation during high-occupancy (>70GB) runs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import dagster as dg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

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
# Constants & Bucket Configuration (A100 80GB High Occupancy)
# ============================================================================

CARBON_3B_A100_HIGH_OCCUPANCY_CONFIGS: list[BucketBatchConfig] = [
    BucketBatchConfig(
        bucket_max_tokens=512,
        batch_size=320,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=1024,
        batch_size=160,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=2048,
        batch_size=80,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=4096,
        batch_size=40,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=8192,
        batch_size=20,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=16384,
        batch_size=8,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=32768,
        batch_size=4,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=False,
    ),
]


# ============================================================================
# Stats
# ============================================================================


@dataclass
class GpuEnrichmentStats:
    """Cumulative counters and dual-profile runtime telemetry for one GPU pass."""

    # Row accounting
    rows_read: int = 0
    rows_enriched: int = 0
    rows_quarantined: int = 0
    rows_exceeding_native_context: int = 0

    # GPU execution counters
    batches_processed: int = 0
    forward_passes: int = 0
    tokens_processed: int = 0

    # OOM telemetry, keyed by configured bucket size
    oom_retries_by_bucket: dict[int, int] = field(default_factory=dict)

    # Runtime measurements
    compile_setup_seconds: float = 0.0
    warmup_seconds: float = 0.0
    gpu_forward_seconds: float = 0.0
    inference_seconds: float = 0.0

    # Peak CUDA memory observed during GPU inference
    peak_memory_allocated_bytes: int = 0
    peak_memory_reserved_bytes: int = 0

    # Output telemetry
    embedding_shards_written: int = 0
    likelihood_shards_written: int = 0

    @property
    def pure_gpu_tokens_per_sec(self) -> float:
        return (
            self.tokens_processed / self.gpu_forward_seconds
            if self.gpu_forward_seconds > 0
            else 0.0
        )

    @property
    def e2e_pipeline_tokens_per_sec(self) -> float:
        return (
            self.tokens_processed / self.inference_seconds
            if self.inference_seconds > 0
            else 0.0
        )

    @property
    def e2e_pipeline_samples_per_sec(self) -> float:
        return (
            self.rows_enriched / self.inference_seconds
            if self.inference_seconds > 0
            else 0.0
        )


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
        merged.gpu_forward_seconds += r.gpu_forward_seconds

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
            f"{output_name} schema mismatch: expected columns {expected_columns}, got {actual_columns}"
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
    token_lengths = record_batch.column("token_length").to_numpy(zero_copy_only=False)
    bucket_indices = np.array(
        [_bucket_for(int(n), buckets).bucket_max_tokens for n in token_lengths]
    )

    grouped: dict[int, pa.RecordBatch] = {}
    for bucket_max in sorted(set(bucket_indices)):
        mask = pa.array(bucket_indices == bucket_max)
        grouped[int(bucket_max)] = record_batch.filter(mask)

    return grouped


# ============================================================================
# Vectorized Batch Padding
# ============================================================================


def _pad_batch(
    token_ids_col: pa.ListArray | pa.ChunkedArray,
    token_mask_col: pa.ListArray | pa.ChunkedArray,
    pad_token_id: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-Python-loop padding directly from Arrow buffers to pinned PyTorch tensors."""
    if isinstance(token_ids_col, pa.ChunkedArray):
        token_ids_col = token_ids_col.combine_chunks()
    if isinstance(token_mask_col, pa.ChunkedArray):
        token_mask_col = token_mask_col.combine_chunks()

    num_rows = len(token_ids_col)

    flat_ids = token_ids_col.values.to_numpy(zero_copy_only=False)
    offsets = token_ids_col.offsets.to_numpy(zero_copy_only=False)
    flat_masks = token_mask_col.values.to_numpy(zero_copy_only=False)

    lengths = np.diff(offsets)
    clipped_lengths = np.minimum(lengths, max_length)

    padded_ids = torch.full(
        (num_rows, max_length),
        pad_token_id,
        dtype=torch.int64,
        pin_memory=True,
    )
    padded_masks = torch.full(
        (num_rows, max_length),
        TOKEN_MASK_PADDING,
        dtype=torch.int8,
        pin_memory=True,
    )

    row_indices = np.repeat(np.arange(num_rows), clipped_lengths)
    col_indices = np.concatenate([np.arange(l) for l in clipped_lengths])
    src_indices = np.concatenate(
        [
            np.arange(offsets[i], offsets[i] + clipped_lengths[i])
            for i in range(num_rows)
        ]
    )

    padded_ids[row_indices, col_indices] = torch.from_numpy(
        flat_ids[src_indices]
    ).long()
    padded_masks[row_indices, col_indices] = torch.from_numpy(
        flat_masks[src_indices]
    ).to(torch.int8)

    return padded_ids, padded_masks


# ============================================================================
# Single forward pass — the core of #14
# ============================================================================


def _run_forward_pass(
    model: Any,
    tokenizer: Any,
    chunk: pa.RecordBatch,
    bucket: BucketBatchConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One model(ids, output_hidden_states=False) call for one GPU-batch.

    Returns (logits, last_hidden_state, input_ids, token_mask) as torch tensors.
    """
    input_ids, token_mask = _pad_batch(
        chunk.column("token_ids"),
        chunk.column("token_mask"),
        tokenizer.pad_token_id,
        bucket.bucket_max_tokens,
    )

    input_ids = input_ids.to(model.device, non_blocking=True)
    token_mask = token_mask.to(model.device, non_blocking=True)

    # attention_mask for the model call itself is just "not padding"
    attention_mask = (token_mask != TOKEN_MASK_PADDING).to(torch.long)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,  # Saves ~30x activation memory vs storing all layers
        )

    return outputs.logits, outputs.last_hidden_state, input_ids, token_mask


def _extract_likelihood_stats(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    use_fp32_reduction: bool = False,
) -> dict[str, pa.Array]:
    """Per-sequence likelihood summary stats from logits, extracted immediately.

    Uses torch.logsumexp reduction to bypass 100GB+ log_softmax allocations.
    Masking follows the FNS token_mask convention: positions are included
    only where token_mask > 0 (valid k-mer content -- 1..k).
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_content_mask = token_mask[:, 1:] > 0

    target_logits = torch.gather(
        shift_logits,
        dim=2,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    if use_fp32_reduction:
        lse = torch.logsumexp(shift_logits.float(), dim=-1)
        token_log_probs = target_logits.float() - lse
    else:
        lse = torch.logsumexp(shift_logits, dim=-1)
        token_log_probs = (target_logits - lse).float()

    masked_log_probs = token_log_probs.masked_fill(~shift_content_mask, 0.0)
    seq_lengths = shift_content_mask.sum(dim=1).clamp(min=1)

    sum_log_prob = masked_log_probs.sum(dim=1)
    mean_log_prob = sum_log_prob / seq_lengths
    perplexity = torch.exp(-mean_log_prob)

    sum_sq_log_prob = (masked_log_probs**2).sum(dim=1)
    variance = ((sum_sq_log_prob / seq_lengths) - (mean_log_prob**2)).clamp(min=0.0)
    per_token_logprob_std = torch.sqrt(variance)

    inf_masked = token_log_probs.masked_fill(~shift_content_mask, float("inf"))
    min_vals, argmin_pos = inf_masked.min(dim=1)

    has_valid = shift_content_mask.any(dim=1)
    min_vals = torch.where(has_valid, min_vals, torch.zeros_like(min_vals))
    argmin_pos = torch.where(has_valid, argmin_pos, torch.full_like(argmin_pos, -1))

    return {
        "mean_log_prob": pa.array(mean_log_prob.cpu().numpy(), type=pa.float32()),
        "sum_log_prob": pa.array(sum_log_prob.cpu().numpy(), type=pa.float32()),
        "perplexity": pa.array(perplexity.cpu().numpy(), type=pa.float32()),
        "supervised_position_count": pa.array(
            seq_lengths.cpu().numpy(), type=pa.int64()
        ),
        "min_token_logprob": pa.array(min_vals.cpu().numpy(), type=pa.float32()),
        "argmin_position": pa.array(argmin_pos.cpu().numpy(), type=pa.int64()),
        "per_token_logprob_std": pa.array(
            per_token_logprob_std.cpu().numpy(), type=pa.float32()
        ),
    }


def _extract_pooled_embeddings(
    last_hidden_state: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[pa.Array, pa.Array]:
    """Mean-pool the last hidden-state layer over real (non-pad) tokens.

    Deliberately includes every non-padding position (token_mask != -2) --
    the <dna>/</dna> boundary special tokens (mask == 0) ARE included
    here, since they carry structural signal.
    """
    mask = (token_mask != TOKEN_MASK_PADDING).unsqueeze(-1).to(last_hidden_state.dtype)
    counts = mask.sum(dim=1).clamp(min=1.0)

    # Batched matrix-vector contraction: avoids (B, L, H) intermediate broadcast allocation
    pooled = torch.bmm(last_hidden_state.transpose(1, 2), mask).squeeze(-1) / counts
    norms = torch.linalg.vector_norm(pooled, dim=-1)

    pooled_cpu = pooled.to(torch.float32).cpu().numpy()
    norms_cpu = norms.to(torch.float32).cpu().numpy()

    embedding_array = pa.FixedSizeListArray.from_arrays(
        pa.array(pooled_cpu.ravel(), type=pa.float32()),
        list_size=last_hidden_state.shape[-1],
    )
    norm_array = pa.array(norms_cpu, type=pa.float32())

    return embedding_array, norm_array


# ============================================================================
# OOM retry cascade (#23)
# ============================================================================


def _run_bucket_batch_with_oom_retry(
    model: Any,
    tokenizer: Any,
    record_batch: pa.RecordBatch,
    bucket: BucketBatchConfig,
    stats: GpuEnrichmentStats,
    use_fp32_reduction: bool = False,
) -> tuple[pa.RecordBatch | None, pa.RecordBatch | None, pa.RecordBatch | None]:
    """Run the forward pass with OOM backoff: full batch -> 1/2 -> 1/4 -> singleton -> quarantine."""
    n = record_batch.num_rows
    offset = 0

    emb_batches: list[pa.RecordBatch] = []
    like_batches: list[pa.RecordBatch] = []
    quarantined_record_ids: list[Any] = []

    fractions = [1, 2, 4, n]

    while offset < n:
        remaining = n - offset
        succeeded = False

        for divisor in fractions:
            chunk_size = max(1, remaining // divisor) if divisor != n else 1
            chunk = record_batch.slice(offset, min(chunk_size, remaining))

            token_lengths = chunk.column("token_length").to_numpy(zero_copy_only=False)
            if (token_lengths > MAX_NATIVE_CONTEXT_TOKENS).any():
                stats.rows_exceeding_native_context += int(
                    (token_lengths > MAX_NATIVE_CONTEXT_TOKENS).sum()
                )
                quarantined_record_ids.extend(chunk.column("record_id").to_pylist())
                offset += chunk.num_rows
                succeeded = True
                break

            try:
                gpu_start = time.perf_counter()
                logits, last_hidden, input_ids, token_mask = _run_forward_pass(
                    model, tokenizer, chunk, bucket
                )
                torch.cuda.synchronize()
                stats.gpu_forward_seconds += time.perf_counter() - gpu_start

                stats.forward_passes += 1
                stats.tokens_processed += int(
                    (token_mask != TOKEN_MASK_PADDING).sum().item()
                )

                record_ids = chunk.column("record_id")

                # 1. Embeddings Arrow Table
                emb_array, norm_array = _extract_pooled_embeddings(
                    last_hidden, token_mask
                )
                emb_batch = pa.RecordBatch.from_arrays(
                    [record_ids, emb_array, norm_array],
                    names=list(EMBEDDING_COLUMNS),
                )
                emb_batches.append(emb_batch)

                # 2. Likelihood Arrow Table
                like_arrays = _extract_likelihood_stats(
                    logits, input_ids, token_mask, use_fp32_reduction=use_fp32_reduction
                )
                like_batch = pa.RecordBatch.from_arrays(
                    [
                        record_ids,
                        like_arrays["mean_log_prob"],
                        like_arrays["sum_log_prob"],
                        like_arrays["perplexity"],
                        like_arrays["supervised_position_count"],
                        like_arrays["min_token_logprob"],
                        like_arrays["argmin_position"],
                        like_arrays["per_token_logprob_std"],
                    ],
                    names=list(LIKELIHOOD_COLUMNS),
                )
                like_batches.append(like_batch)

                del logits, last_hidden, input_ids, token_mask
                offset += chunk.num_rows
                succeeded = True
                break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                stats.oom_retries_by_bucket[bucket.bucket_max_tokens] = (
                    stats.oom_retries_by_bucket.get(bucket.bucket_max_tokens, 0) + 1
                )
                logger.warning(
                    f"GPU OOM: bucket={bucket.bucket_max_tokens}, "
                    f"requested_batch={chunk.num_rows}, retrying with backoff."
                )
                continue

        if not succeeded:
            quarantined_record_ids.append(
                record_batch.column("record_id")[offset].as_py()
            )
            offset += 1

    stats.rows_quarantined += len(quarantined_record_ids)

    out_emb = (
        pa.concat_tables([pa.Table.from_batches(emb_batches)]).to_batches()[0]
        if emb_batches
        else None
    )
    out_like = (
        pa.concat_tables([pa.Table.from_batches(like_batches)]).to_batches()[0]
        if like_batches
        else None
    )
    out_quarantine = (
        pa.RecordBatch.from_arrays(
            [pa.array(quarantined_record_ids)], names=[GPU_JOIN_KEY]
        )
        if quarantined_record_ids
        else None
    )

    if out_emb is not None:
        _validate_output_columns(
            out_emb, EMBEDDING_COLUMNS, output_name="carbon_embeddings"
        )
        _validate_join_key(out_emb, output_name="carbon_embeddings")

    if out_like is not None:
        _validate_output_columns(
            out_like, LIKELIHOOD_COLUMNS, output_name="carbon_likelihood_stats"
        )
        _validate_join_key(out_like, output_name="carbon_likelihood_stats")

    return out_emb, out_like, out_quarantine


# ============================================================================
# Batch reading — Arrow-native, tokenized shards with live progress
# ============================================================================


def iter_tokenized_batches(
    input_dir: str | Path,
    batch_size: int,
    context: dg.AssetExecutionContext | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream RecordBatches directly from tokenize_and_tag's output shards."""
    input_dir = Path(input_dir)
    shard_paths = sorted(input_dir.glob("shard-*.parquet"))
    total_shards = len(shard_paths)

    if not shard_paths:
        raise FileNotFoundError(f"No tokenized shards found in {input_dir}")

    total_rows_read = 0
    start_time = time.perf_counter()

    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        parquet_file = pq.ParquetFile(shard_path)

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            total_rows_read += batch.num_rows
            yield batch

        if context is not None:
            elapsed = time.perf_counter() - start_time
            shards_done_pct = (shard_idx / total_shards) * 100
            shards_per_sec = shard_idx / elapsed if elapsed > 0 else 0
            eta_seconds = (
                (total_shards - shard_idx) / shards_per_sec if shards_per_sec > 0 else 0
            )

            context.log.info(
                f"[Shard Progress {shard_idx}/{total_shards} ({shards_done_pct:.1f}%)] "
                f"Rows: {total_rows_read:,} | Elapsed: {elapsed / 60:.1f}m | ETA: {eta_seconds / 60:.1f}m"
            )


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
    """Single-pass GPU enrichment: embeddings + likelihood stats, bucketed, checkpointed, OOM-resilient."""
    import torch

    buckets = buckets or CARBON_3B_A100_HIGH_OCCUPANCY_CONFIGS

    raw_model = carbon_resource.model
    tokenizer = carbon_resource.tokenizer

    # Pre-warm static compilation buckets
    compile_start = time.perf_counter()
    compiled_model = carbon_resource.compile_for_buckets(buckets)
    compile_setup_seconds = time.perf_counter() - compile_start

    stats = GpuEnrichmentStats(compile_setup_seconds=compile_setup_seconds)
    torch.cuda.reset_peak_memory_stats()

    inference_start = time.perf_counter()

    with (
        ParquetShardWriter(
            embeddings_output_dir,
            rows_per_shard,
            compression,
            compression_level=1,
        ) as emb_writer,
        ParquetShardWriter(
            likelihood_output_dir,
            rows_per_shard,
            compression,
            compression_level=1,
        ) as like_writer,
        ParquetShardWriter(
            quarantine_output_dir,
            rows_per_shard,
            compression,
            compression_level=1,
        ) as quarantine_writer,
    ):
        for raw_batch in iter_tokenized_batches(input_dir, batch_size, context=context):
            if raw_batch.num_rows == 0:
                continue

            stats.rows_read += raw_batch.num_rows
            grouped = _group_by_bucket(raw_batch, buckets)

            for bucket_max, bucket_batch in grouped.items():
                bucket = next(b for b in buckets if b.bucket_max_tokens == bucket_max)
                active_model = compiled_model if bucket.compile_enabled else raw_model

                emb_batch, like_batch, quarantine_batch = (
                    _run_bucket_batch_with_oom_retry(
                        active_model, tokenizer, bucket_batch, bucket, stats
                    )
                )

                if emb_batch is not None and emb_batch.num_rows > 0:
                    emb_writer.write_batch(emb_batch)
                    stats.rows_enriched += emb_batch.num_rows

                if like_batch is not None and like_batch.num_rows > 0:
                    like_writer.write_batch(like_batch)

                if quarantine_batch is not None and quarantine_batch.num_rows > 0:
                    quarantine_writer.write_batch(quarantine_batch)

            stats.batches_processed += 1

        stats.embedding_shards_written = emb_writer.shards_written
        stats.likelihood_shards_written = like_writer.shards_written

    stats.inference_seconds = time.perf_counter() - inference_start
    stats.peak_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    stats.peak_memory_reserved_bytes = torch.cuda.max_memory_reserved()

    # Integrity reconciliation check
    reconciled_rows = stats.rows_enriched + stats.rows_quarantined
    if reconciled_rows != stats.rows_read:
        raise RuntimeError(
            "GPU enrichment row reconciliation failed: "
            f"rows_read={stats.rows_read}, "
            f"rows_enriched={stats.rows_enriched}, "
            f"rows_quarantined={stats.rows_quarantined}"
        )

    if context is not None:
        context.log.info(
            "GPU enrichment complete: "
            f"elapsed={stats.inference_seconds:.2f}s, "
            f"pure_gpu_tok/s={stats.pure_gpu_tokens_per_sec:,.0f}, "
            f"e2e_tok/s={stats.e2e_pipeline_tokens_per_sec:,.0f}, "
            f"e2e_samples/s={stats.e2e_pipeline_samples_per_sec:,.1f}, "
            f"peak_allocated={stats.peak_memory_allocated_bytes / 1e9:.2f}GB, "
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
    """Single model(ids, output_hidden_states=False) forward pass per bucketed

    batch, emitting both embeddings and likelihood stats (#14). Bucketed
    by token length (#1), OOM-resilient (#23), record_id preserved as the
    immutable join key (#8) and asserted below.
    """
    context.log.info(f"model_checkpoint={carbon.model_checkpoint!r}")
    context.log.info(f"tokenizer_revision={carbon.tokenizer_revision!r}")

    stats = process_gpu_enrichment(
        input_dir=config.tokenized_output_dir,
        embeddings_output_dir=f"{config.tokenized_output_dir}_embeddings",
        likelihood_output_dir=f"{config.tokenized_output_dir}_likelihood",
        quarantine_output_dir=f"{config.tokenized_output_dir}_quarantine",
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        carbon_resource=carbon,
        context=context,
    )

    if stats.oom_retries_by_bucket:
        context.log.warning(
            "OOM retries observed — revise the M3.5 bucket/batch lookup table: "
            f"{stats.oom_retries_by_bucket}"
        )

    if stats.rows_quarantined:
        context.log.warning(f"{stats.rows_quarantined:,} rows quarantined.")

    metadata = {
        "rows_read": stats.rows_read,
        "rows_enriched": stats.rows_enriched,
        "rows_quarantined": stats.rows_quarantined,
        "rows_exceeding_native_context": stats.rows_exceeding_native_context,
        "oom_retries_by_bucket": dg.MetadataValue.json(stats.oom_retries_by_bucket),
        "batches_processed": stats.batches_processed,
        "forward_passes": stats.forward_passes,
        "tokens_processed": stats.tokens_processed,
        "pure_gpu_tokens_per_sec": stats.pure_gpu_tokens_per_sec,
        "e2e_pipeline_tokens_per_sec": stats.e2e_pipeline_tokens_per_sec,
        "e2e_pipeline_samples_per_sec": stats.e2e_pipeline_samples_per_sec,
        "inference_seconds": stats.inference_seconds,
        "gpu_forward_seconds": stats.gpu_forward_seconds,
        "compile_setup_seconds": stats.compile_setup_seconds,
        "peak_memory_allocated_gb": stats.peak_memory_allocated_bytes / 1e9,
        "peak_memory_reserved_gb": stats.peak_memory_reserved_bytes / 1e9,
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
