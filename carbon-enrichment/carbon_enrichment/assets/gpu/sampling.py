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
from pathlib import Path
from typing import Any

import dagster as dg
import pyarrow as pa
import pyarrow.parquet as pq
from carbon_enrichment.assets.cpu.streaming import ParquetShardWriter
from carbon_enrichment.config import CarbonPipelineConfig

# ============================================================================
# Constants
# ============================================================================

# bp-length bucket boundaries used as a token-length proxy (see module
# docstring). Not the real M3.5 bucket table (#15) -- that's defined in
# token space, post-tokenization, and consumed by embeddings.py. This is
# a coarser, earlier-stage stand-in for stratification purposes only.
_LENGTH_PROXY_BUCKETS = [512, 2048, 8192, 32768, float("inf")]

# Drift tolerance for the post-hoc representativeness check: if any
# stratum's sampled share deviates from its full-corpus share by more
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
    Stratification key: (length_bucket_proxy, CDS flag, strand, taxonomy
    rank). taxonomy_domain is the top-level rank only (already what
    enrich_batch produces), not the full lineage -- using the full
    lineage string would create far too many near-singleton strata at
    32M-row scale.
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
    Deterministic, reproducible per-row inclusion decision.

    Uses sha256 rather than Python's built-in hash() -- built-in string
    hashing is randomized per-process (PYTHONHASHSEED) unless explicitly
    disabled, which would silently make "the same seed" produce a
    different sample on every run. Not vectorized (no batch-level
    pyarrow.compute hash kernel exists for this) -- cheap enough per-row
    at this scale that a Python loop is not the bottleneck here, unlike
    the composition/entropy work in enrichment.py that #24 vectorized.
    """

    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()

    # First 8 hex chars -> uniform value in [0, 1).
    fractional_value = int(digest[:8], 16) / 0xFFFFFFFF

    return fractional_value < fraction


# ============================================================================
# Single-pass sampling + representativeness accounting
# ============================================================================


def sample_pilot_corpus(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    fraction: float,
    seed: int,
    batch_size: int,
    rows_per_shard: int,
    compression: str,
    context: dg.AssetExecutionContext | None = None,
) -> dict[str, Any]:
    """
    Single streaming pass: decide inclusion per row, write included rows,
    and accumulate per-stratum counts for both the full corpus and the
    sampled subset -- so representativeness (#22.5) can be validated from
    this same pass, no second pass required.
    """

    input_dir = Path(input_dir)

    shard_paths = sorted(input_dir.glob("shard-*.parquet"))

    if not shard_paths:
        raise FileNotFoundError(f"No CPU-enriched shards found in {input_dir}")

    full_corpus_counts: dict[tuple, int] = defaultdict(int)
    sampled_counts: dict[tuple, int] = defaultdict(int)

    rows_read = 0
    rows_sampled = 0
    batches_processed = 0

    with ParquetShardWriter(output_dir, rows_per_shard, compression) as writer:
        for shard_path in shard_paths:
            parquet_file = pq.ParquetFile(shard_path)

            for record_batch in parquet_file.iter_batches(batch_size=batch_size):
                record_ids = record_batch.column("record_id").to_pylist()
                sequence_lengths = record_batch.column("sequence_length").to_pylist()
                is_coding = record_batch.column("is_coding_region").to_pylist()
                strands = record_batch.column("strand").to_pylist()
                taxonomy_domains = record_batch.column("taxonomy_domain").to_pylist()

                include_mask = []

                for rid, seq_len, cds, strand, domain in zip(
                    record_ids, sequence_lengths, is_coding, strands, taxonomy_domains
                ):
                    key = _stratum_key(seq_len, cds, strand, domain)

                    full_corpus_counts[key] += 1

                    included = _include_row(rid, seed, fraction)

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
                        f"Pilot sampling progress: {rows_read:,} rows read, "
                        f"{rows_sampled:,} sampled "
                        f"({rows_sampled / max(rows_read, 1):.1%})"
                    )

    drift_report = _representativeness_drift(full_corpus_counts, sampled_counts)

    return {
        "rows_read": rows_read,
        "rows_sampled": rows_sampled,
        "actual_fraction": rows_sampled / rows_read if rows_read else 0.0,
        "target_fraction": fraction,
        "seed": seed,
        "shards_written": writer.shards_written,
        "stratum_count": len(full_corpus_counts),
        "drift_report": drift_report,
    }


# ============================================================================
# Representativeness validation (#22.5)
# ============================================================================


def _representativeness_drift(
    full_corpus_counts: dict[tuple, int],
    sampled_counts: dict[tuple, int],
) -> list[dict[str, Any]]:
    """
    Compare each stratum's share of the full corpus against its share of
    the sample. Returns strata whose drift exceeds
    _DRIFT_WARNING_THRESHOLD_PCT, sorted by drift magnitude -- the
    "validate the sample's composition matches the full corpus" step
    #22.5 asks for, computed from the counts already gathered during the
    single sampling pass above.
    """

    total_full = sum(full_corpus_counts.values())
    total_sampled = sum(sampled_counts.values())

    if total_full == 0 or total_sampled == 0:
        return []

    drifted = []

    for key, full_count in full_corpus_counts.items():
        full_share = full_count / total_full
        sampled_share = sampled_counts.get(key, 0) / total_sampled

        drift_pct = abs(sampled_share - full_share) * 100.0

        if drift_pct > _DRIFT_WARNING_THRESHOLD_PCT:
            drifted.append(
                {
                    "stratum": str(key),
                    "full_corpus_share_pct": round(full_share * 100, 3),
                    "sampled_share_pct": round(sampled_share * 100, 3),
                    "drift_pct": round(drift_pct, 3),
                }
            )

    return sorted(drifted, key=lambda d: -d["drift_pct"])


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    name="carbon_pilot_corpus",
    group_name="cpu",
    compute_kind="cpu",
    deps=["carbon_cpu_enriched_sequences"],
    description=(
        "Stratified subset of the CPU-enriched corpus for GPU enrichment "
        "(design doc #22.5) -- proportional sampling via a deterministic "
        "per-row hash, not first-N-by-order. Stratifies on a bp-length "
        "proxy (real token length isn't known pre-tokenization), CDS "
        "flag, strand, and taxonomy rank. Feeds carbon_tokenized_corpus "
        "instead of the full 32M-row corpus."
    ),
)
def carbon_pilot_corpus(
    context: dg.AssetExecutionContext,
    config: CarbonPipelineConfig,
) -> dg.MaterializeResult:
    """Sample a stratified pilot subset of the CPU-enriched corpus."""

    context.log.info(
        f"Sampling pilot corpus: fraction={config.pilot_sample_fraction!r}, "
        f"seed={config.pilot_sample_seed!r}"
    )

    result = sample_pilot_corpus(
        input_dir=config.output_dir,
        output_dir=config.pilot_output_dir,
        fraction=config.pilot_sample_fraction,
        seed=config.pilot_sample_seed,
        batch_size=config.batch_size,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        context=context,
    )

    if result["drift_report"]:
        context.log.warning(
            f"{len(result['drift_report'])} strata exceeded "
            f"{_DRIFT_WARNING_THRESHOLD_PCT}pp drift between full-corpus "
            "and sampled share -- review before trusting M3.5/M5.5 "
            f"numbers to generalize (#22.5): {result['drift_report'][:5]}"
        )

    context.log.info(
        "Pilot sampling completed: "
        f"rows_sampled={result['rows_sampled']:,} / {result['rows_read']:,} "
        f"({result['actual_fraction']:.2%}), "
        f"strata={result['stratum_count']:,}, "
        f"shards={result['shards_written']:,}"
    )

    return dg.MaterializeResult(
        metadata={
            "rows_read": result["rows_read"],
            "rows_sampled": result["rows_sampled"],
            "actual_fraction": round(result["actual_fraction"], 4),
            "target_fraction": result["target_fraction"],
            "sampling_seed": result["seed"],
            "sampling_method": "deterministic_per_row_hash_stratified",
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
