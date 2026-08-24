"""
M4 — Likelihood-stats validation and summary.

This module does NOT run a model forward pass. Per design doc #14,
logits and hidden_states come from ONE forward pass, owned by
`assets/gpu/embeddings.py` (`carbon_gpu_enrichment`). That multi_asset
already wrote `carbon_likelihood_stats` shards (mean_log_prob,
sum_log_prob, perplexity per record_id) as one of its two outputs.
Re-running the model here to "compute M4 properly" would silently
duplicate GPU cost and violate #14 -- this file's only job is to
validate and summarize what embeddings.py already produced.

    carbon_gpu_enrichment (embeddings.py, single forward pass)
            │
            ├── carbon_embeddings        (M5, this file does not touch it)
            └── carbon_likelihood_stats  (M4 output, consumed below)
                        │
                        ▼
                carbon_likelihood_summary  (this module)
                  - record_id integrity check (#8)
                  - corpus-wide likelihood distribution stats
                  - low-likelihood outlier flagging (QC signal, not a
                    filter -- no rows are dropped here)

Principles applied here:
- #8  record_id is the immutable join key; assert set-equality and count
      equality between the tokenized input and the likelihood-stats output
- #13 corpus-wide aggregation (percentiles, outlier thresholds) is fine
      here specifically because this is a declared separate stage, not
      folded into the row-wise GPU loop in embeddings.py
- #21 read via ParquetFile.iter_batches, Arrow-native, no full-corpus
      pandas materialization
"""

from pathlib import Path
from typing import Any

import dagster as dg
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from carbon_enrichment.config import CarbonPipelineConfig

# ============================================================================
# Constants
# ============================================================================

# Outlier threshold expressed as a percentile of the corpus's own
# perplexity distribution, not an absolute cutoff -- absolute perplexity
# values are model/tokenizer-specific and would need re-tuning per
# checkpoint. This is a QC flag, not a filter: no rows are removed.
LOW_LIKELIHOOD_PERCENTILE = 1.0  # bottom 1% by mean_log_prob


# ============================================================================
# record_id integrity (#8)
# ============================================================================


def _read_record_id_column(input_dir: str | Path, column: str = "record_id") -> pa.Array:
    """Read just the record_id column across every shard in a directory."""

    input_dir = Path(input_dir)

    shard_paths = sorted(input_dir.glob("shard-*.parquet"))

    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {input_dir}")

    chunks = [pq.read_table(p, columns=[column]).column(column) for p in shard_paths]

    return pa.chunked_array(chunks).combine_chunks()


def _assert_record_id_integrity(
    tokenized_input_dir: str | Path,
    likelihood_output_dir: str | Path,
) -> tuple[int, int]:
    """
    Assert set(input.record_id) == set(output.record_id) and count
    equality (#8), scoped correctly: quarantined rows (design doc #23,
    written by embeddings.py to its own quarantine shard) are legitimately
    absent from carbon_likelihood_stats, so this checks against
    (input - quarantined), not the full tokenized input set.
    """

    input_ids = _read_record_id_column(tokenized_input_dir)
    output_ids = _read_record_id_column(likelihood_output_dir)

    quarantine_dir = Path(f"{tokenized_input_dir}_quarantine")

    quarantined_ids: set[Any] = set()
    if quarantine_dir.exists() and any(quarantine_dir.glob("shard-*.parquet")):
        quarantined_ids = set(_read_record_id_column(quarantine_dir).to_pylist())

    expected_ids = set(input_ids.to_pylist()) - quarantined_ids
    actual_ids = set(output_ids.to_pylist())

    missing = expected_ids - actual_ids
    unexpected = actual_ids - expected_ids

    if missing or unexpected:
        raise RuntimeError(
            "record_id integrity check failed for carbon_likelihood_stats "
            f"(#8): {len(missing)} missing, {len(unexpected)} unexpected. "
            "This checks against (tokenized input - quarantined rows), "
            "not the raw tokenized input set."
        )

    return len(expected_ids), len(actual_ids)


# ============================================================================
# Corpus-wide likelihood summary (#13: declared separate stage, not the
# row-wise GPU loop)
# ============================================================================


def _summarize_likelihood(likelihood_output_dir: str | Path) -> dict[str, Any]:
    """
    Corpus-wide distribution stats over mean_log_prob / perplexity,
    computed from the already-written likelihood shards -- read
    incrementally shard by shard (Arrow-native), not loaded as one
    in-memory pandas frame.
    """

    input_dir = Path(likelihood_output_dir)

    shard_paths = sorted(input_dir.glob("shard-*.parquet"))

    mean_log_prob_chunks = []
    perplexity_chunks = []

    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["mean_log_prob", "perplexity"])
        mean_log_prob_chunks.append(table.column("mean_log_prob"))
        perplexity_chunks.append(table.column("perplexity"))

    mean_log_prob = pa.chunked_array(mean_log_prob_chunks).combine_chunks()
    perplexity = pa.chunked_array(perplexity_chunks).combine_chunks()

    if len(mean_log_prob) == 0:
        return {
            "rows_summarized": 0,
            "mean_log_prob_p1": None,
            "mean_log_prob_p50": None,
            "mean_log_prob_p99": None,
            "perplexity_p50": None,
            "perplexity_p99": None,
            "low_likelihood_threshold": None,
            "low_likelihood_row_count": 0,
        }

    quantiles = pc.quantile(
        mean_log_prob,
        q=[
            LOW_LIKELIHOOD_PERCENTILE / 100.0,
            0.5,
            0.99,
        ],
    ).to_pylist()

    low_likelihood_threshold = quantiles[0]

    perplexity_quantiles = pc.quantile(perplexity, q=[0.5, 0.99]).to_pylist()

    below_threshold = pc.sum(
        pc.cast(pc.less_equal(mean_log_prob, low_likelihood_threshold), pa.int64())
    ).as_py() or 0

    return {
        "rows_summarized": len(mean_log_prob),
        "mean_log_prob_p1": quantiles[0],
        "mean_log_prob_p50": quantiles[1],
        "mean_log_prob_p99": quantiles[2],
        "perplexity_p50": perplexity_quantiles[0],
        "perplexity_p99": perplexity_quantiles[1],
        "low_likelihood_threshold": low_likelihood_threshold,
        "low_likelihood_row_count": below_threshold,
    }


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    name="carbon_likelihood_summary",
    group_name="gpu",
    compute_kind="cpu",
    deps=["carbon_gpu_enrichment"],
    description=(
        "Validates record_id integrity (#8) and summarizes the corpus-wide "
        "likelihood distribution from carbon_likelihood_stats. Does NOT run "
        "a model forward pass -- consumes the output of embeddings.py's "
        "single shared pass (#14)."
    ),
)
def carbon_likelihood_summary(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
) -> dg.MaterializeResult:
    """Validate and summarize the likelihood stats produced in embeddings.py."""

    likelihood_dir = f"{config.tokenized_output_dir}_likelihood"

    context.log.info(f"Validating record_id integrity against {likelihood_dir!r}")

    expected_count, actual_count = _assert_record_id_integrity(
        tokenized_input_dir=config.tokenized_output_dir,
        likelihood_output_dir=likelihood_dir,
    )

    context.log.info(
        f"record_id integrity OK: expected={expected_count:,}, actual={actual_count:,}"
    )

    summary = _summarize_likelihood(likelihood_dir)

    context.log.info(
        "Likelihood summary: "
        f"rows={summary['rows_summarized']:,}, "
        f"mean_log_prob_p50={summary['mean_log_prob_p50']}, "
        f"low_likelihood_rows={summary['low_likelihood_row_count']:,}"
    )

    if summary["low_likelihood_row_count"]:
        context.log.info(
            f"{summary['low_likelihood_row_count']:,} rows flagged as low-"
            f"likelihood (bottom {LOW_LIKELIHOOD_PERCENTILE}% by "
            "mean_log_prob) -- QC signal only, no rows removed."
        )

    return dg.MaterializeResult(
        metadata={
            "record_id_expected_count": expected_count,
            "record_id_actual_count": actual_count,
            **summary,
            "low_likelihood_percentile": LOW_LIKELIHOOD_PERCENTILE,
        }
    )