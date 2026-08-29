import gc
import json
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
import pyarrow.compute as pc
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
# Run-budget configuration
# ============================================================================

DEFAULT_GPU_TOKEN_BUDGET = 500_000_000
GPU_CHECKPOINT_FILENAME = "gpu_enrichment_checkpoint.json"

# Max rows to run through the lm_head at once. With a 156k vocab in bf16,
# a (chunk, T, V) logits tensor costs roughly chunk * T * 156_000 * 2 bytes.
# At T=512 that's ~2.55 GB for chunk=16. Tune empirically via
# stats.peak_memory_allocated_bytes; raise for throughput if headroom exists,
# lower if still tight.
DEFAULT_LOGIT_CHUNK_SIZE = 16


# ============================================================================
# Constants & Bucket Configuration (A100 80GB High Occupancy)
# ============================================================================

CARBON_3B_A100_HIGH_OCCUPANCY_CONFIGS: list[BucketBatchConfig] = [
    BucketBatchConfig(
        bucket_max_tokens=512,
        batch_size=160,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=1024,
        batch_size=80,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=2048,
        batch_size=40,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=4096,
        batch_size=20,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=8192,
        batch_size=8,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=True,
    ),
    BucketBatchConfig(
        bucket_max_tokens=16384,
        batch_size=2,
        dtype="bfloat16",
        attn_backend="kernels-community/flash-attn2",
        compile_enabled=False,
    ),
    BucketBatchConfig(
        bucket_max_tokens=32768,
        batch_size=1,
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
    missing = [column for column in GPU_JOIN_KEY if column not in batch.schema.names]
    if missing:
        raise ValueError(
            f"{output_name} is missing required GPU join columns: {missing!r}"
        )


# ============================================================================
# Bucketing
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
    """Split one input RecordBatch into per-bucket RecordBatches."""
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
        flat_ids[src_indices].astype(np.int64, copy=False)
    )
    padded_masks[row_indices, col_indices] = torch.from_numpy(
        flat_masks[src_indices].astype(np.int8, copy=False)
    )

    return padded_ids, padded_masks


# ============================================================================
# Likelihood stats (operates on whatever logits slice it's given —
# full batch or a chunk; shapes just need to line up)
# ============================================================================


def _extract_likelihood_stats(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    use_fp32_reduction: bool = False,
) -> dict[str, pa.Array]:
    """Per-sequence likelihood summary stats from logits."""
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


def _concat_likelihood_arrays(
    chunks: list[dict[str, pa.Array]],
) -> dict[str, pa.Array]:
    """Reassemble per-chunk likelihood stat dicts into full-batch arrays."""
    keys = chunks[0].keys()
    return {key: pa.concat_arrays([c[key] for c in chunks]) for key in keys}


def _extract_pooled_embeddings(
    last_hidden_state: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[pa.Array, pa.Array]:
    """Mean-pool the last hidden-state layer over real (non-pad) tokens."""
    mask = (token_mask != TOKEN_MASK_PADDING).unsqueeze(-1).to(last_hidden_state.dtype)
    counts = mask.sum(dim=1).clamp(min=1.0)

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


def _resolve_base_model(model: Any) -> Any:
    """Best-effort resolution of the transformer body without the lm_head.

    Adjust this if CarbonModelResource exposes the backbone under a
    different attribute (e.g. model.transformer) — the goal is simply to
    run the forward pass without the final vocab projection so we control
    when/how the lm_head is applied.
    """
    return getattr(model, "model", None) or getattr(model, "base_model", model)


# ============================================================================
# Single forward pass — chunked lm_head to avoid materializing full-batch
# (B, T, V) logits. With vocab ~156k, a full (320, 512, 156_000) bf16 logits
# tensor is ~51 GB by itself; chunking keeps peak logits memory bounded to
# (logit_chunk_size, T, V) regardless of the outer batch size.
# ============================================================================


def _resolve_logit_chunk_size(
    bucket_max_tokens: int,
    base_chunk: int = DEFAULT_LOGIT_CHUNK_SIZE,
    base_tokens: int = 512,
) -> int:
    """Scale the lm_head row-chunk size inversely with sequence length so
    chunk_size * T (and hence the (chunk, T, V) logits tensor) stays roughly
    constant across buckets, instead of the fixed row-count blowing up
    memory linearly with T at long sequence lengths."""
    return max(1, (base_chunk * base_tokens) // bucket_max_tokens)


def _run_forward_pass(
    model: Any,
    tokenizer: Any,
    chunk: pa.RecordBatch,
    bucket: BucketBatchConfig,
    logit_chunk_size: int = DEFAULT_LOGIT_CHUNK_SIZE,
    use_fp32_reduction: bool = False,
) -> tuple[dict[str, pa.Array], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Base-model forward once; lm_head + likelihood reduction run per
    row-chunk so full-batch logits are never resident at once.

    logit_chunk_size here is the *base* chunk size (i.e. the value tuned
    for bucket.bucket_max_tokens == base_tokens, default 512). The actual
    per-call chunk size is resolved against bucket.bucket_max_tokens so
    that chunk_size * T stays roughly constant across buckets — a fixed
    row count would otherwise make the (chunk, T, V) logits tensor grow
    linearly with T and OOM at long sequence lengths.

    Returns (likelihood_arrays, last_hidden_state, input_ids, token_mask).
    last_hidden_state is still returned in full — it's needed for pooled
    embeddings and is far smaller than logits (H << V).
    """
    input_ids, token_mask = _pad_batch(
        chunk.column("token_ids"),
        chunk.column("token_mask"),
        tokenizer.pad_token_id,
        bucket.bucket_max_tokens,
    )

    input_ids = input_ids.to(model.device, non_blocking=True)
    token_mask = token_mask.to(model.device, non_blocking=True)

    attention_mask = (token_mask != TOKEN_MASK_PADDING).to(torch.long)

    base_model = _resolve_base_model(model)

    resolved_logit_chunk_size = _resolve_logit_chunk_size(
        bucket.bucket_max_tokens,
        base_chunk=logit_chunk_size,
    )

    with torch.inference_mode():
        base_outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = base_outputs.last_hidden_state  # (B, T, H)

        num_rows = last_hidden_state.shape[0]
        likelihood_chunks: list[dict[str, pa.Array]] = []

        for start in range(0, num_rows, resolved_logit_chunk_size):
            end = min(start + resolved_logit_chunk_size, num_rows)

            chunk_logits = model.lm_head(last_hidden_state[start:end])  # (c, T, V)
            chunk_ids = input_ids[start:end]
            chunk_mask = token_mask[start:end]

            likelihood_chunks.append(
                _extract_likelihood_stats(
                    chunk_logits,
                    chunk_ids,
                    chunk_mask,
                    use_fp32_reduction=use_fp32_reduction,
                )
            )

            del chunk_logits  # freed before the next chunk allocates

    likelihood_arrays = _concat_likelihood_arrays(likelihood_chunks)

    return likelihood_arrays, last_hidden_state, input_ids, token_mask


# ============================================================================
# OOM retry cascade
# ============================================================================


def _run_bucket_batch_with_oom_retry(
    model: Any,
    tokenizer: Any,
    record_batch: pa.RecordBatch,
    bucket: BucketBatchConfig,
    stats: GpuEnrichmentStats,
    use_fp32_reduction: bool = False,
    logit_chunk_size: int = DEFAULT_LOGIT_CHUNK_SIZE,
) -> tuple[pa.RecordBatch | None, pa.RecordBatch | None, pa.RecordBatch | None]:
    """Run the forward pass with OOM backoff: full batch -> 1/2 -> 1/4 -> singleton -> quarantine."""
    n = record_batch.num_rows
    offset = 0

    emb_batches: list[pa.RecordBatch] = []
    like_batches: list[pa.RecordBatch] = []
    quarantined_record_ids: list[Any] = []
    quarantined_starts: list[Any] = []
    quarantined_ends: list[Any] = []

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
                quarantined_starts.extend(chunk.column("start").to_pylist())
                quarantined_ends.extend(chunk.column("end").to_pylist())
                offset += chunk.num_rows
                succeeded = True
                break

            try:
                gpu_start = time.perf_counter()
                like_arrays, last_hidden, input_ids, token_mask = _run_forward_pass(
                    model,
                    tokenizer,
                    chunk,
                    bucket,
                    logit_chunk_size=logit_chunk_size,
                    use_fp32_reduction=use_fp32_reduction,
                )
                torch.cuda.synchronize()
                stats.gpu_forward_seconds += time.perf_counter() - gpu_start

                stats.forward_passes += 1
                stats.tokens_processed += int(
                    (token_mask != TOKEN_MASK_PADDING).sum().item()
                )

                record_ids = chunk.column("record_id")
                starts = chunk.column("start")
                ends = chunk.column("end")

                # 1. Embeddings Record Batch
                emb_array, norm_array = _extract_pooled_embeddings(
                    last_hidden, token_mask
                )
                emb_batch = pa.RecordBatch.from_arrays(
                    [record_ids, starts, ends, emb_array, norm_array],
                    names=list(EMBEDDING_COLUMNS),
                )
                emb_batches.append(emb_batch)

                # 2. Likelihood Record Batch (already computed per-chunk in
                #    _run_forward_pass and reassembled into full-batch arrays)
                like_batch = pa.RecordBatch.from_arrays(
                    [
                        record_ids,
                        starts,
                        ends,
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

                del last_hidden, input_ids, token_mask
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
            quarantined_starts.append(record_batch.column("start")[offset].as_py())
            quarantined_ends.append(record_batch.column("end")[offset].as_py())
            offset += 1

    stats.rows_quarantined += len(quarantined_record_ids)

    out_emb = (
        pa.Table.from_batches(emb_batches).combine_chunks().to_batches()[0]
        if emb_batches
        else None
    )
    out_like = (
        pa.Table.from_batches(like_batches).combine_chunks().to_batches()[0]
        if like_batches
        else None
    )
    out_quarantine = (
        pa.RecordBatch.from_arrays(
            [
                pa.array(quarantined_record_ids),
                pa.array(quarantined_starts),
                pa.array(quarantined_ends),
            ],
            names=list(GPU_JOIN_KEY),
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
# Resumable output/checkpoint helpers
# ============================================================================


def _checkpoint_path(input_dir: str | Path) -> Path:
    return Path(input_dir) / GPU_CHECKPOINT_FILENAME


def _load_checkpoint(input_dir: str | Path) -> dict[str, Any]:
    path = _checkpoint_path(input_dir)
    if not path.exists():
        return {"version": 1, "completed_shards": {}}

    with path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)

    if checkpoint.get("version") != 1:
        raise ValueError(
            f"Unsupported GPU enrichment checkpoint version: "
            f"{checkpoint.get('version')!r}"
        )

    completed = checkpoint.get("completed_shards", {})
    if not isinstance(completed, dict):
        raise ValueError("GPU enrichment checkpoint has invalid completed_shards")

    return checkpoint


def _save_checkpoint(
    input_dir: str | Path,
    checkpoint: dict[str, Any],
) -> None:
    """Atomically persist completed input-shard state."""
    path = _checkpoint_path(input_dir)
    tmp = path.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2, sort_keys=True)
        handle.write("\n")

    tmp.replace(path)


def _tokenized_shard_paths(input_dir: str | Path) -> list[Path]:
    paths = sorted(Path(input_dir).glob("shard-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No tokenized shards found in {input_dir}")
    return paths


def _shard_token_count(path: Path) -> int:
    """Read only token_length to determine the exact shard token budget."""
    total = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=262_144,
        columns=["token_length"],
    ):
        total += int(pc.sum(batch.column("token_length")).as_py())
    return total


def _next_output_shard_index(output_dir: str | Path) -> int:
    """Return the next free sequential output-shard index."""
    output_dir = Path(output_dir)
    existing = sorted(output_dir.glob("shard-*.parquet"))
    if not existing:
        return 0

    return max(int(path.stem.split("-")[-1]) for path in existing) + 1


class ResumableParquetShardWriter(ParquetShardWriter):
    """ParquetShardWriter variant that never overwrites prior run output."""

    def __init__(
        self,
        output_dir: str | Path,
        rows_per_shard: int,
        compression: str,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            rows_per_shard=rows_per_shard,
            compression=compression,
            compression_level=1,
        )
        self._shard_index = _next_output_shard_index(output_dir)


def _new_output_writers(
    embeddings_output_dir: str | Path,
    likelihood_output_dir: str | Path,
    quarantine_output_dir: str | Path,
    rows_per_shard: int,
    compression: str,
):
    return (
        ResumableParquetShardWriter(
            embeddings_output_dir,
            rows_per_shard,
            compression,
        ),
        ResumableParquetShardWriter(
            likelihood_output_dir,
            rows_per_shard,
            compression,
        ),
        ResumableParquetShardWriter(
            quarantine_output_dir,
            rows_per_shard,
            compression,
        ),
    )


def _remove_output_shards_created_after(
    output_dir: str | Path,
    first_new_index: int,
) -> None:
    """Remove output files created by an input shard that did not commit."""
    output_dir = Path(output_dir)
    for path in output_dir.glob("shard-*.parquet"):
        try:
            index = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        if index >= first_new_index:
            path.unlink(missing_ok=True)


# ============================================================================
# Batch reading
# ============================================================================


def iter_tokenized_shard_batches(
    shard_path: Path,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Stream one tokenized input shard with only GPU-required columns."""
    parquet_file = pq.ParquetFile(shard_path)
    required_columns = [
        "record_id",
        "start",
        "end",
        "token_ids",
        "token_mask",
        "token_length",
    ]

    available = parquet_file.schema_arrow.names
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(f"{shard_path} is missing required GPU columns: {missing}")

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=required_columns,
    ):
        if batch.num_rows:
            yield batch


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
    max_tokens_per_run: int = DEFAULT_GPU_TOKEN_BUDGET,
    logit_chunk_size: int = DEFAULT_LOGIT_CHUNK_SIZE,
) -> GpuEnrichmentStats:
    """Run one resumable token-budgeted GPU enrichment pass."""
    if max_tokens_per_run <= 0:
        raise ValueError("max_tokens_per_run must be greater than zero")

    buckets = buckets or CARBON_3B_A100_HIGH_OCCUPANCY_CONFIGS

    input_dir = Path(input_dir)
    embedding_output_dir = Path(embeddings_output_dir)
    likelihood_output_dir = Path(likelihood_output_dir)
    quarantine_output_dir = Path(quarantine_output_dir)

    shard_paths = _tokenized_shard_paths(input_dir)
    checkpoint = _load_checkpoint(input_dir)
    completed_shards: dict[str, Any] = checkpoint["completed_shards"]

    raw_model = carbon_resource.model
    tokenizer = carbon_resource.tokenizer

    compile_start = time.perf_counter()
    compiled_model = carbon_resource.compile_for_buckets(buckets)
    compile_setup_seconds = time.perf_counter() - compile_start

    stats = GpuEnrichmentStats(
        compile_setup_seconds=compile_setup_seconds,
    )
    torch.cuda.reset_peak_memory_stats()

    inference_start = time.perf_counter()
    run_tokens = 0
    run_rows = 0
    shards_committed_this_run = 0

    for shard_path in shard_paths:
        shard_name = shard_path.name

        if shard_name in completed_shards:
            continue

        shard_tokens = _shard_token_count(shard_path)

        if run_tokens > 0 and run_tokens + shard_tokens > max_tokens_per_run:
            if context is not None:
                context.log.info(
                    f"GPU token budget reached: run_tokens={run_tokens:,}, "
                    f"next_shard={shard_name}, next_shard_tokens={shard_tokens:,}, "
                    f"budget={max_tokens_per_run:,}. Stopping cleanly."
                )
            break

        if context is not None:
            context.log.info(
                f"Processing {shard_name}: {shard_tokens:,} tokens; "
                f"run budget {run_tokens:,}/{max_tokens_per_run:,}."
            )

        emb_start_index = _next_output_shard_index(embedding_output_dir)
        like_start_index = _next_output_shard_index(likelihood_output_dir)
        quarantine_start_index = _next_output_shard_index(quarantine_output_dir)

        emb_writer, like_writer, quarantine_writer = _new_output_writers(
            embedding_output_dir,
            likelihood_output_dir,
            quarantine_output_dir,
            rows_per_shard,
            compression,
        )

        shard_stats_before = (
            stats.rows_read,
            stats.rows_enriched,
            stats.rows_quarantined,
            stats.tokens_processed,
            stats.batches_processed,
        )

        try:
            with emb_writer, like_writer, quarantine_writer:
                for raw_batch in iter_tokenized_shard_batches(
                    shard_path,
                    batch_size,
                ):
                    stats.rows_read += raw_batch.num_rows
                    grouped = _group_by_bucket(raw_batch, buckets)

                    for bucket_max, bucket_batch in grouped.items():
                        bucket = next(
                            b for b in buckets if b.bucket_max_tokens == bucket_max
                        )
                        active_model = (
                            compiled_model if bucket.compile_enabled else raw_model
                        )

                        emb_batch, like_batch, quarantine_batch = (
                            _run_bucket_batch_with_oom_retry(
                                active_model,
                                tokenizer,
                                bucket_batch,
                                bucket,
                                stats,
                                logit_chunk_size=logit_chunk_size,
                            )
                        )

                        if emb_batch is not None and emb_batch.num_rows > 0:
                            emb_writer.write_batch(emb_batch)
                            stats.rows_enriched += emb_batch.num_rows

                        if like_batch is not None and like_batch.num_rows > 0:
                            like_writer.write_batch(like_batch)

                        if (
                            quarantine_batch is not None
                            and quarantine_batch.num_rows > 0
                        ):
                            quarantine_writer.write_batch(quarantine_batch)

                    stats.batches_processed += 1

            completed_shards[shard_name] = {
                "tokens": shard_tokens,
                "rows": stats.rows_read - shard_stats_before[0],
            }
            checkpoint["completed_shards"] = completed_shards
            checkpoint["last_committed_shard"] = shard_name
            checkpoint["last_committed_at"] = time.time()
            _save_checkpoint(input_dir, checkpoint)

            run_tokens += shard_tokens
            run_rows += completed_shards[shard_name]["rows"]
            shards_committed_this_run += 1

            stats.embedding_shards_written += emb_writer.shards_written
            stats.likelihood_shards_written += like_writer.shards_written

            if context is not None:
                context.log.info(
                    f"Committed {shard_name}: "
                    f"{shard_tokens:,} tokens; "
                    f"run total={run_tokens:,}/{max_tokens_per_run:,}."
                )

        except Exception:
            _remove_output_shards_created_after(
                embedding_output_dir,
                emb_start_index,
            )
            _remove_output_shards_created_after(
                likelihood_output_dir,
                like_start_index,
            )
            _remove_output_shards_created_after(
                quarantine_output_dir,
                quarantine_start_index,
            )
            torch.cuda.empty_cache()
            gc.collect()
            raise

    stats.inference_seconds = time.perf_counter() - inference_start
    stats.peak_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    stats.peak_memory_reserved_bytes = torch.cuda.max_memory_reserved()

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
            "GPU enrichment run complete: "
            f"shards_committed={shards_committed_this_run}, "
            f"rows={run_rows:,}, "
            f"run_tokens={run_tokens:,}, "
            f"budget={max_tokens_per_run:,}, "
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
# Dagster multi_asset
# ============================================================================


@dg.multi_asset(
    outs={
        "carbon_embeddings": dg.AssetOut(
            description=(
                "Mean-pooled raw hidden-state embeddings, one row per record_id."
            )
        ),
        "carbon_likelihood_stats": dg.AssetOut(
            description=("Per-sequence log-prob/perplexity stats derived from logits.")
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
    """Base-model forward pass with chunked lm_head per bucketed batch,
    so full-batch (B, T, V) logits are never materialized at once."""
    context.log.info(f"model_checkpoint={carbon.model_checkpoint!r}")
    context.log.info(f"tokenizer_revision={carbon.tokenizer_revision!r}")

    stats = process_gpu_enrichment(
        input_dir=config.tokenized_output_dir,
        embeddings_output_dir=config.embeddings_output_dir,
        likelihood_output_dir=config.likelihood_output_dir,
        quarantine_output_dir=f"{config.tokenized_output_dir}_quarantine",
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        carbon_resource=carbon,
        context=context,
        max_tokens_per_run=getattr(
            config,
            "gpu_token_budget",
            DEFAULT_GPU_TOKEN_BUDGET,
        ),
        logit_chunk_size=getattr(
            config,
            "logit_chunk_size",
            DEFAULT_LOGIT_CHUNK_SIZE,
        ),
    )

    if stats.oom_retries_by_bucket:
        context.log.warning(f"OOM retries observed: {stats.oom_retries_by_bucket}")

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
        "gpu_token_budget": getattr(
            config,
            "gpu_token_budget",
            DEFAULT_GPU_TOKEN_BUDGET,
        ),
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
        asset_key="carbon_embeddings",
        metadata={
            **metadata,
            "parquet_shards": stats.embedding_shards_written,
        },
    )

    yield dg.MaterializeResult(
        asset_key="carbon_likelihood_stats",
        metadata={
            **metadata,
            "parquet_shards": stats.likelihood_shards_written,
        },
    )
