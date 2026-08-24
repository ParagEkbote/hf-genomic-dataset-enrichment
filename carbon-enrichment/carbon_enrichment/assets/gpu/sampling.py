"""
M2.5 — Stratified pilot-corpus sampling (design doc #22.5).

Sits between CPU enrichment and tokenize_and_tag:

    carbon_cpu_enriched_sequences (32M rows, full corpus)
                │
                ▼
    carbon_pilot_corpus  (this module — ~25% subset, stratified)
                │
                ▼
    carbon_tokenized_corpus (#14.5)

WHY A UNIFORM PER-ROW RATE ACHIEVES PROPORTIONAL STRATIFICATION
------------------------------------------------------------------
#22.5 explicitly warns against "first 25% by row order" and unstratified
random sampling. The key fact this module relies on: a per-row inclusion
probability that is IDENTICAL across every stratum, applied
independently per row, IS proportional stratified sampling -- it is
mathematically equivalent to sampling each stratum separately at the
same rate. What #22.5 actually forbids is correlation between inclusion
and any stratification variable (row order correlates with ingestion
order, which correlates with taxonomy/species clustering in the source
corpus) -- not uniformity of rate per se.

This lets sampling happen in ONE streaming pass, consistent with every
other stage in this pipeline (bounded memory, no corpus materialization):
inclusion is decided per row via a deterministic hash of record_id, and
per-stratum counts for BOTH the full corpus and the sampled subset are
accumulated in the same pass -- no second pass to learn stratum sizes
first, no large in-memory reservoir.

STRATIFICATION VARIABLES -- ONE DELIBERATE SUBSTITUTION
----------------------------------------------------------
#1/#22.5 call for stratifying by token-length BUCKET. Token length is not
knowable at this point in the pipeline -- it depends on <dna> tag-wrapping
and 6-mer tokenization behavior (#14.5), which hasn't run yet; running it
just to stratify a sample would defeat the point of subsampling before
the expensive tokenize/GPU stages. This module stratifies on
`sequence_length` (raw bp, already present from CPU enrichment) binned
into buckets, as an approximate proxy -- flagged explicitly here rather
than silently presented as the real thing. bp length and 6-mer token
count are highly correlated (~length/6 before tag overhead), so this is
a reasonable proxy, not an equivalent.

Other stratification axes -- CDS flag, strand, taxonomy rank -- use the
real CPU-enriched columns directly (`is_coding_region`, `strand`,
`taxonomy_domain`), no approximation needed there.

DETERMINISM AND REPRODUCIBILITY
----------------------------------
Inclusion uses `hashlib.sha256(f"{seed}:{record_id}")`, not Python's
built-in `hash()` -- built-in string hashing is randomized per-process
(PYTHONHASHSEED) unless explicitly disabled, which would make the sample
non-reproducible across runs. sha256 is deterministic and cheap enough
per-row at this scale (tens of seconds for 32M rows, negligible next to
GPU/tokenization cost). Sampling method, stratification variables, and
seed are recorded in asset output metadata per #12/#22.5's provenance
requirement.
"""

import hashlib
from collections import defaultdict
from typing import Any, Literal

import dagster as dg
import pyarrow as pa

from carbon_enrichment.assets.cpu.streaming import (
    ParquetShardWriter,
    _read_hub_dataset,
    _read_local_parquet,
    iter_batches,
)
from carbon_enrichment.config import CarbonPipelineConfig
from carbon_enrichment.schema import VALIDATION_LEVEL_ROWS

# ============================================================================
# Constants
# ============================================================================

# bp-length bucket boundaries used as a token-length proxy (see module
# docstring). Not the real M3.5 bucket table (#15) -- that's defined in
# token space, post-tokenization, and consumed by embeddings.py. This is
# a coarser, earlier-stage stand-in for stratification purposes only.
_LENGTH_PROXY_BUCKETS = [512, 2048, 8192, 32768, float("inf")]

# Drift tolerance for the post-hoc representativeness check: if any
# stratum's sampled share deviates from its bounded-corpus share by more
# than this (in percentage points), it's logged as a warning, not a
# failure -- #22.5 asks this be validated and recorded, not enforced as
# a hard gate at this stage.
_DRIFT_WARNING_THRESHOLD_PCT = 2.0


# ============================================================================
# Stratification key
# ============================================================================


def _length_bucket_proxy(sequence_length: int) -> int:
    """Return the upper bound of the bp-length bucket sequence_length falls into."""

    for boundary in _LENGTH_PROXY_BUCKETS:
        if sequence_length <= boundary:
            return int(boundary) if boundary != float("inf") else -1

    return -1  # unreachable, last boundary is inf


def _stratum_key(
    sequence_length: int,
    is_coding_region: bool,
    strand: Any,
    taxonomy_domain: Any,
) -> tuple:
    """
    Return the stratification key.

    The key consists of:

    - sequence-length bucket proxy
    - CDS/coding-region flag
    - strand
    - taxonomy domain

    taxonomy_domain is the top-level rank already produced by enrichment,
    rather than the complete taxonomy lineage. Using the full lineage would
    create far too many near-singleton strata at 32M-row scale.
    """

    return (
        _length_bucket_proxy(sequence_length),
        bool(is_coding_region),
        strand,
        taxonomy_domain,
    )


# ============================================================================
# Deterministic inclusion decision
# ============================================================================


def _include_row(record_id: str, seed: int, fraction: float) -> bool:
    """
    Return a deterministic, reproducible per-row inclusion decision.

    Uses SHA-256 rather than Python's built-in hash() because built-in string
    hashing is randomized per process unless PYTHONHASHSEED is explicitly
    controlled. SHA-256 therefore guarantees that the same seed and
    record_id produce the same decision across runs and processes.
    """

    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()

    # First 8 hex chars -> approximately uniform value in [0, 1).
    fractional_value = int(digest[:8], 16) / 0xFFFFFFFF

    return fractional_value < fraction


# ============================================================================
# Single-pass sampling + representativeness accounting
# ============================================================================


def sample_pilot_corpus(
    input_source: str,
    input_type: Literal["local", "hub"],
    output_dir: str,
    *,
    fraction: float,
    seed: int,
    max_rows: int | None,
    batch_size: int,
    rows_per_shard: int,
    compression: str,
    context: dg.AssetExecutionContext | None = None,
) -> dict[str, Any]:
    """
    Sample a pilot corpus in a single streaming pass.

    The input may be either:

    - a local CPU-enriched Parquet corpus, or
    - a Hugging Face Hub CPU-enriched dataset.

    ``max_rows`` is a hard upper bound on the number of source rows consumed
    from the input. It is normally derived from the configured validation
    level, e.g.:

        dev            -> 1,000
        dev_gpu_small  -> 3,000
        dev_gpu_medium -> 30,000
        integration    -> 1,000,000
        auth           -> 32,410,000

    Within that bounded source, each row receives a deterministic inclusion
    decision based on its record_id, seed, and sampling fraction.

    The same pass:

    1. reads the bounded source corpus,
    2. computes the stratification key,
    3. decides whether each row is sampled,
    4. writes sampled rows,
    5. accumulates bounded-source and sampled stratum counts.

    This avoids a second pass solely for representativeness validation.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"Sampling fraction must be between 0.0 and 1.0, got {fraction!r}"
        )

    if max_rows is not None and max_rows < 0:
        raise ValueError(f"max_rows must be non-negative or None, got {max_rows!r}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")

    if rows_per_shard <= 0:
        raise ValueError(f"rows_per_shard must be positive, got {rows_per_shard!r}")

    if input_type == "hub":
        row_iter = _read_hub_dataset(
            input_source,
            batch_size=batch_size,
        )
    else:
        row_iter = _read_local_parquet(
            input_source,
            batch_size=batch_size,
        )

    # These counts describe the bounded validation corpus, not necessarily
    # the entire upstream Hub dataset.
    bounded_corpus_counts: dict[tuple, int] = defaultdict(int)
    sampled_counts: dict[tuple, int] = defaultdict(int)

    rows_read = 0
    rows_sampled = 0
    batches_processed = 0
    shards_written = 0

    with ParquetShardWriter(
        output_dir,
        rows_per_shard,
        compression,
    ) as writer:
        for record_batch in iter_batches(
            row_iter,
            batch_size=batch_size,
        ):
            # ------------------------------------------------------------
            # Enforce the validation-level row ceiling.
            #
            # The final batch may contain more rows than remain under
            # max_rows, so slice it before any statistics or sampling are
            # performed.
            # ------------------------------------------------------------

            if max_rows is not None:
                remaining_rows = max_rows - rows_read

                if remaining_rows <= 0:
                    break

                if record_batch.num_rows > remaining_rows:
                    record_batch = record_batch.slice(
                        0,
                        remaining_rows,
                    )

            if record_batch.num_rows == 0:
                break

            record_ids = record_batch.column("record_id").to_pylist()
            sequence_lengths = record_batch.column("sequence_length").to_pylist()
            is_coding = record_batch.column("is_coding_region").to_pylist()
            strands = record_batch.column("strand").to_pylist()
            taxonomy_domains = record_batch.column("taxonomy_domain").to_pylist()

            include_mask: list[bool] = []

            for rid, seq_len, cds, strand, domain in zip(
                record_ids,
                sequence_lengths,
                is_coding,
                strands,
                taxonomy_domains,
            ):
                key = _stratum_key(
                    seq_len,
                    cds,
                    strand,
                    domain,
                )

                bounded_corpus_counts[key] += 1

                included = _include_row(
                    rid,
                    seed,
                    fraction,
                )

                include_mask.append(included)

                if included:
                    sampled_counts[key] += 1

            rows_read += record_batch.num_rows

            mask_array = pa.array(include_mask)

            sampled_batch = record_batch.filter(mask_array)

            if sampled_batch.num_rows:
                writer.write_batch(sampled_batch)
                rows_sampled += sampled_batch.num_rows

            batches_processed += 1

            if context is not None and batches_processed % 100 == 0:
                context.log.info(
                    "Pilot sampling progress: "
                    f"{rows_read:,} rows read, "
                    f"{rows_sampled:,} sampled "
                    f"({rows_sampled / max(rows_read, 1):.1%})"
                )

            # Explicitly stop once the validation-level ceiling has been
            # reached. This is mostly redundant with the next-loop check,
            # but makes the control flow obvious and prevents an additional
            # iterator request from the Hub.
            if max_rows is not None and rows_read >= max_rows:
                break

        shards_written = writer.shards_written

    drift_report = _representativeness_drift(
        bounded_corpus_counts,
        sampled_counts,
    )

    return {
        "rows_read": rows_read,
        "rows_sampled": rows_sampled,
        "actual_fraction": (rows_sampled / rows_read if rows_read else 0.0),
        "target_fraction": fraction,
        "seed": seed,
        "max_rows": max_rows,
        "shards_written": shards_written,
        "stratum_count": len(bounded_corpus_counts),
        "drift_report": drift_report,
    }


# ============================================================================
# Representativeness validation (#22.5)
# ============================================================================


def _representativeness_drift(
    bounded_corpus_counts: dict[tuple, int],
    sampled_counts: dict[tuple, int],
) -> list[dict[str, Any]]:
    """
    Compare each stratum's share of the bounded source corpus against its
    share of the sampled corpus.

    Returns strata whose drift exceeds
    ``_DRIFT_WARNING_THRESHOLD_PCT``, sorted by drift magnitude.

    This validates the composition of the pilot against the exact bounded
    source used for the current validation run.

    It does not claim that a 3,000-row ``dev_gpu_small`` run has validated
    representativeness against the entire upstream 32M-row corpus.
    """

    total_bounded = sum(bounded_corpus_counts.values())
    total_sampled = sum(sampled_counts.values())

    if total_bounded == 0 or total_sampled == 0:
        return []

    drifted: list[dict[str, Any]] = []

    for key, bounded_count in bounded_corpus_counts.items():
        bounded_share = bounded_count / total_bounded
        sampled_share = sampled_counts.get(key, 0) / total_sampled

        drift_pct = abs(sampled_share - bounded_share) * 100.0

        if drift_pct > _DRIFT_WARNING_THRESHOLD_PCT:
            drifted.append(
                {
                    "stratum": str(key),
                    "bounded_corpus_share_pct": round(
                        bounded_share * 100,
                        3,
                    ),
                    "sampled_share_pct": round(
                        sampled_share * 100,
                        3,
                    ),
                    "drift_pct": round(
                        drift_pct,
                        3,
                    ),
                }
            )

    return sorted(
        drifted,
        key=lambda d: -d["drift_pct"],
    )


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    name="carbon_pilot_corpus",
    group_name="cpu",
    compute_kind="cpu",
    description=(
        "Stratified subset of the CPU-enriched corpus for GPU enrichment "
        "(design doc #22.5) -- proportional sampling via a deterministic "
        "per-row hash, not first-N-by-order. Stratifies on a bp-length "
        "proxy (real token length isn't known pre-tokenization), CDS "
        "flag, strand, and taxonomy rank. Consumes the CPU-enriched "
        "corpus (local or Hub) and feeds carbon_tokenized_corpus instead "
        "of the full 32M-row corpus."
    ),
    # No static deps declaration here because the CPU-enriched input may
    # come either from the local carbon_cpu_enriched_sequences asset or
    # from the externally materialized Hugging Face Hub dataset.
)
def carbon_pilot_corpus(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
) -> dg.MaterializeResult:
    """Sample a stratified pilot subset of the CPU-enriched corpus."""

    if config.cpu_enriched_input_type == "hub":
        if not config.cpu_enriched_dataset:
            raise ValueError(
                "cpu_enriched_dataset is required when cpu_enriched_input_type='hub'"
            )

        input_source = config.cpu_enriched_dataset

    else:
        input_source = config.output_dir

    try:
        max_rows = VALIDATION_LEVEL_ROWS[config.validation_level]
    except KeyError as exc:
        raise ValueError(
            f"Unknown validation level: {config.validation_level!r}. "
            f"Expected one of: {tuple(VALIDATION_LEVEL_ROWS)}"
        ) from exc

    context.log.info(
        "Sampling pilot corpus: "
        f"source_type={config.cpu_enriched_input_type!r}, "
        f"source={input_source!r}, "
        f"validation_level={config.validation_level!r}, "
        f"max_rows={max_rows:,}, "
        f"fraction={config.pilot_sample_fraction!r}, "
        f"seed={config.pilot_sample_seed!r}"
    )

    result = sample_pilot_corpus(
        input_source=input_source,
        input_type=config.cpu_enriched_input_type,
        output_dir=config.pilot_output_dir,
        fraction=config.pilot_sample_fraction,
        seed=config.pilot_sample_seed,
        max_rows=max_rows,
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        context=context,
    )

    if result["drift_report"]:
        context.log.warning(
            f"{len(result['drift_report'])} strata exceeded "
            f"{_DRIFT_WARNING_THRESHOLD_PCT}pp drift between bounded "
            "source and sampled share -- review before trusting "
            "M3.5/M5.5 numbers to generalize (#22.5): "
            f"{result['drift_report'][:5]}"
        )

    context.log.info(
        "Pilot sampling completed: "
        f"rows_sampled={result['rows_sampled']:,} / "
        f"{result['rows_read']:,} "
        f"({result['actual_fraction']:.2%}), "
        f"validation_limit={max_rows:,}, "
        f"strata={result['stratum_count']:,}, "
        f"shards={result['shards_written']:,}"
    )

    return dg.MaterializeResult(
        metadata={
            "input_source_type": config.cpu_enriched_input_type,
            "input_dataset": (
                config.cpu_enriched_dataset
                if config.cpu_enriched_input_type == "hub"
                else None
            ),
            "input_dir": (
                config.output_dir if config.cpu_enriched_input_type == "local" else None
            ),
            "validation_level": config.validation_level,
            "max_rows": max_rows,
            "rows_read": result["rows_read"],
            "rows_sampled": result["rows_sampled"],
            "actual_fraction": round(
                result["actual_fraction"],
                4,
            ),
            "target_fraction": result["target_fraction"],
            "sampling_seed": result["seed"],
            "sampling_method": ("deterministic_per_row_hash_stratified"),
            "stratification_variables": [
                "sequence_length_bucket_proxy",
                "is_coding_region",
                "strand",
                "taxonomy_domain",
            ],
            "stratum_count": result["stratum_count"],
            "parquet_shards": result["shards_written"],
            "drifted_strata_count": len(result["drift_report"]),
            "drift_report": dg.MetadataValue.json(result["drift_report"]),
        }
    )
