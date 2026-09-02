from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from carbon_enrichment.resources.clickhouse import (
    ClickHouseResource,
    ClickHouseConfig,
)


# ============================================================================
# Constants
# ============================================================================

EMBEDDINGS_URL = (
    "https://huggingface.co/datasets/AINovice2005/carbon-embeddings/"
    "resolve/main/*.parquet"
)

LIKELIHOOD_URL = (
    "https://huggingface.co/datasets/AINovice2005/carbon-likelihood-stats/"
    "resolve/main/*.parquet"
)

CPU_URL = (
    "https://huggingface.co/datasets/AINovice2005/carbon-cpu-enriched-sequences/"
    "resolve/main/*.parquet"
)

DEFAULT_OUTPUT = Path("results/case_study/phase45")
DEFAULT_TAXON_INDEX = -1
DEFAULT_OUTLIER_Z = 3.0
DEFAULT_TOP_DIMS = 10

# `start` and `end` are reserved words in ClickHouse's SQL grammar
# (`END` closes CASE/interval expressions, `START` appears in
# transaction syntax). Every reference to these columns MUST be
# backtick-quoted, or the parser can silently desync and blame an
# unrelated identifier (this is what produced the original
# "Unknown expression identifier `record_id`" error).
COL_START = "`start`"
COL_END = "`end`"


# ============================================================================
# Helpers
# ============================================================================

def sql_quote(value: str) -> str:
    """Escape a string for SQL."""
    return "'" + value.replace("'", "''") + "'"


def source_for(resource: ClickHouseResource, name: str) -> str:
    """Get the ClickHouse source expression for a registered dataset."""
    return resource.source_expr(name)


def table_to_parquet(table: pa.Table, path: Path) -> None:
    """Write a PyArrow table to Parquet with Zstd compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def table_to_csv(table: pa.Table, path: Path) -> None:
    """Safely write a PyArrow table to CSV, including list columns."""
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = []
    fields = []

    for i, field in enumerate(table.schema):
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            arrays.append(table.column(i).cast(pa.string()))
            fields.append(pa.field(field.name, pa.string()))
        else:
            arrays.append(table.column(i))
            fields.append(field)

    safe_table = pa.Table.from_arrays(
        arrays,
        schema=pa.schema(fields),
    )

    pc.write_csv(safe_table, path)


def safe_float(value: Any) -> float | None:
    """Safely convert a value to finite float."""
    if value is None:
        return None

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def inspect_columns(ch: ClickHouseResource, source: str) -> set[str]:
    """Get source column names."""
    table = ch.query_arrow(f"DESCRIBE TABLE {source}")
    return {str(x) for x in table.column("name").to_pylist()}


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Phase 4.5: 8 non-KNN metrics for the common cohort."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for Phase 4.5 results.",
    )

    parser.add_argument(
        "--taxon-index",
        type=int,
        default=DEFAULT_TAXON_INDEX,
        help=(
            "Zero-based component of semicolon-delimited taxonomy to use "
            "for intra-taxon ranking. -1 uses the final component."
        ),
    )

    parser.add_argument(
        "--outlier-z",
        type=float,
        default=DEFAULT_OUTLIER_Z,
        help="Absolute z-score threshold for comparative outlier flags.",
    )

    parser.add_argument(
        "--top-dims",
        type=int,
        default=DEFAULT_TOP_DIMS,
        help="Number of top embedding dimensions to retain per record.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help=(
            "ClickHouse max_threads. Defaults to 4 to reduce peak memory "
            "during remote scans and window operations."
        ),
    )

    parser.add_argument(
        "--max-memory-bytes",
        type=int,
        default=0,
        help=(
            "Optional ClickHouse max_memory_usage cap in bytes for this "
            "query (0 disables the override and uses server default)."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    common_cohort_path = Path(
        "/teamspace/studios/this_studio/"
        "hf-genomic-dataset-enrichment/results/derived/common_cohort/"
        "common_cohort.csv"
    )

    if not common_cohort_path.exists():
        raise FileNotFoundError(
            f"Common cohort file not found: {common_cohort_path}"
        )

    # ------------------------------------------------------------------
    # Load common cohort
    # ------------------------------------------------------------------

    common_cohort_table = pc.read_csv(common_cohort_path)

    required_columns = {"record_id", "start", "end"}
    available_columns = set(common_cohort_table.column_names)

    missing = required_columns - available_columns
    if missing:
        raise ValueError(
            f"Common cohort is missing required columns: {sorted(missing)}"
        )

    record_ids = common_cohort_table["record_id"].to_pylist()
    starts = common_cohort_table["start"].to_pylist()
    ends = common_cohort_table["end"].to_pylist()

    # Deduplicate composite keys in Python.
    unique_keys = set(zip(record_ids, starts, ends))

    print("=" * 78)
    print("PHASE 4.5 — 8 NON-KNN METRICS FOR COMMON COHORT")
    print("=" * 78)
    print(f"Common cohort source: {common_cohort_path}")
    print(f"Unique composite keys: {len(unique_keys):,}")
    print(f"Output directory: {args.output}")
    print(f"ClickHouse threads: {args.threads}")
    print()

    # ------------------------------------------------------------------
    # Create compact cohort-key Parquet.
    #
    # This file is used as a membership set. The analytical query does
    # NOT JOIN the huge HF sources directly against this table.
    #
    # Sorted by (record_id, start, end) so that when ClickHouse reads
    # this as a MergeTree-backed / file() source, the IN-subquery's
    # right side is already ordered — this lets the engine build a
    # smaller, more cache-friendly hash set and can speed up the
    # semi-join probe against each large HF source.
    # ------------------------------------------------------------------

    sorted_keys = sorted(unique_keys, key=lambda k: (k[0], k[1], k[2]))

    keys_table = pa.Table.from_arrays(
        [
            pa.array([k[0] for k in sorted_keys], type=pa.string()),
            pa.array([k[1] for k in sorted_keys], type=pa.int64()),
            pa.array([k[2] for k in sorted_keys], type=pa.int64()),
        ],
        schema=pa.schema(
            [
                ("record_id", pa.string()),
                ("start", pa.int64()),
                ("end", pa.int64()),
            ]
        ),
    )

    keys_path = args.output / "_cohort_keys.parquet"
    record_path = args.output / "_record_metrics.parquet"

    table_to_parquet(keys_table, keys_path)

    config_kwargs = {
        "threads": args.threads,
    }

    started = time.time()

    try:
        with ClickHouseResource(ClickHouseConfig(**config_kwargs)) as ch:

            # ==========================================================
            # Register sources
            # ==========================================================

            ch.register_hf_dataset("embeddings", EMBEDDINGS_URL)
            ch.register_hf_dataset("likelihood", LIKELIHOOD_URL)
            ch.register_hf_dataset("cpu", CPU_URL)
            ch.register_dataset("cohort_keys", keys_path)

            embeddings = source_for(ch, "embeddings")
            likelihood = source_for(ch, "likelihood")
            cpu = source_for(ch, "cpu")
            cohort_keys = source_for(ch, "cohort_keys")

            # ==========================================================
            # Inspect optional schema fields
            # ==========================================================

            likelihood_cols = inspect_columns(ch, likelihood)
            cpu_cols = inspect_columns(ch, cpu)

            has_token_array = "per_token_logprob" in likelihood_cols

            boundary_candidates = [
                ("gene_start", "gene_end"),
                ("gene_start_position", "gene_end_position"),
                (
                    "begin_of_gene_position",
                    "end_of_gene_position",
                ),
            ]

            boundary_pair = next(
                (
                    pair
                    for pair in boundary_candidates
                    if pair[0] in cpu_cols and pair[1] in cpu_cols
                ),
                None,
            )

            # ==========================================================
            # Taxonomy expression
            # ==========================================================

            taxonomy_expr = (
                (
                    "arrayElement("
                    "splitByChar(';', taxonomy), "
                    f"{args.taxon_index + 1}"
                    ")"
                )
                if args.taxon_index != -1
                else "arrayElement(splitByChar(';', taxonomy), -1)"
            )

            # ==========================================================
            # Optional likelihood token metrics
            # ==========================================================

            token_fields = ""

            if has_token_array:
                token_fields = """
                    ,length(per_token_logprob) AS token_profile_length
                    ,arrayMin(per_token_logprob) AS token_profile_min
                    ,arrayMax(per_token_logprob) AS token_profile_max
                    ,arrayAvg(per_token_logprob) AS token_profile_mean
                """

            # ==========================================================
            # Optional gene boundary fields
            # ==========================================================

            boundary_fields = ""

            if boundary_pair is not None:
                b0, b1 = boundary_pair

                boundary_fields = f"""
                    ,toFloat64({b0}) AS gene_start_position
                    ,toFloat64({b1}) AS gene_end_position
                """

            # ==========================================================
            # IMPORTANT MEMORY OPTIMIZATION
            #
            # Every large HF source is independently restricted to the
            # common cohort BEFORE the joins happen.
            #
            # We also extract the top embedding dimensions immediately,
            # so the full embedding array does not survive into `base`.
            # ==========================================================

            top_dims = max(1, int(args.top_dims))

            # Build boundary SQL separately so nested SQL quotes cannot
            # break the Python f-string containing the main query.
            if boundary_pair is not None:
                final_boundary_fields = """
                ,gene_start_position
                ,gene_end_position
                ,(
                    argmin_position >= gene_start_position
                    AND argmin_position <= gene_end_position
                ) AS argmin_inside_gene_boundary
                ,least(
                    abs(
                        toFloat64(argmin_position)
                        - gene_start_position
                    ),
                    abs(
                        toFloat64(argmin_position)
                        - gene_end_position
                    )
                ) AS distance_to_nearest_gene_boundary
                ,'exact_numeric_gene_boundaries'
                    AS boundary_method
                """
            else:
                final_boundary_fields = """
                ,CAST(NULL AS Nullable(Float64))
                    AS gene_start_position
                ,CAST(NULL AS Nullable(Float64))
                    AS gene_end_position
                ,CAST(NULL AS Nullable(UInt8))
                    AS argmin_inside_gene_boundary
                ,CAST(NULL AS Nullable(Float64))
                    AS distance_to_nearest_gene_boundary
                ,'sequence_boundary_proxy_only'
                    AS boundary_method
                """

            memory_setting = (
                f", max_memory_usage = {args.max_memory_bytes}"
                if args.max_memory_bytes > 0
                else ""
            )

            sql = f"""
            WITH

            /* ---------------------------------------------------------
             * Cohort membership set.
             *
             * This is deliberately used as an IN/set filter rather than
             * as the right side of a normal hash JOIN. `start`/`end`
             * are ClickHouse reserved words and MUST be backtick-quoted
             * everywhere they appear as identifiers.
             * --------------------------------------------------------- */

            cohort AS
            (
                SELECT
                    record_id,
                    {COL_START},
                    {COL_END}
                FROM {cohort_keys}
            ),

            /* ---------------------------------------------------------
             * CPU: filter FIRST.
             * --------------------------------------------------------- */

            filtered_cpu AS
            (
                SELECT
                    record_id,
                    {COL_START},
                    {COL_END},
                    sequence_length,
                    gc_content,
                    shannon_entropy,
                    toUInt8(is_coding_region) AS is_coding_region,
                    gene_length,
                    taxonomy,
                    {taxonomy_expr} AS taxon_group
                    {boundary_fields}
                FROM {cpu}
                WHERE (record_id, {COL_START}, {COL_END}) IN (
                    SELECT record_id, {COL_START}, {COL_END} FROM cohort
                )
            ),

            /* ---------------------------------------------------------
             * Likelihood: filter FIRST.
             * --------------------------------------------------------- */

            filtered_likelihood AS
            (
                SELECT
                    record_id,
                    {COL_START},
                    {COL_END},
                    mean_log_prob,
                    sum_log_prob,
                    perplexity,
                    supervised_position_count,
                    min_token_logprob,
                    argmin_position,
                    per_token_logprob_std
                    {token_fields}
                FROM {likelihood}
                WHERE (record_id, {COL_START}, {COL_END}) IN (
                    SELECT record_id, {COL_START}, {COL_END} FROM cohort
                )
            ),

            /* ---------------------------------------------------------
             * Embeddings: filter FIRST.
             *
             * Critically, the original embedding array is NOT retained.
             * Only embedding_norm and top-K diagnostic information
             * survive.
             *
             * arrayPartialReverseSort avoids sorting the complete vector
             * when only the top K coordinates are needed.
             * --------------------------------------------------------- */

            filtered_embeddings AS
            (
                SELECT
                    record_id,
                    {COL_START},
                    {COL_END},
                    embedding_norm,

                    if(
                        length(embedding) = 0,
                        CAST([], 'Array(UInt16)'),

                        arrayMap(
                            x -> tupleElement(x, 2),

                            arrayResize(
                                arrayPartialReverseSort(
                                    x -> tupleElement(x, 1),
                                    {top_dims},

                                    arrayZip(
                                        arrayMap(
                                            x -> abs(x),
                                            embedding
                                        ),
                                        arrayEnumerate(embedding)
                                    )
                                ),

                                {top_dims}
                            )
                        )
                    ) AS top_embedding_dimensions,

                    if(
                        length(embedding) = 0,
                        CAST([], 'Array(Float32)'),

                        arrayMap(
                            x -> tupleElement(x, 1),

                            arrayResize(
                                arrayPartialReverseSort(
                                    x -> tupleElement(x, 1),
                                    {top_dims},

                                    arrayZip(
                                        arrayMap(
                                            x -> abs(x),
                                            embedding
                                        ),
                                        arrayEnumerate(embedding)
                                    )
                                ),

                                {top_dims}
                            )
                        )
                    ) AS top_embedding_abs_values

                FROM {embeddings}
                WHERE (record_id, {COL_START}, {COL_END}) IN (
                    SELECT record_id, {COL_START}, {COL_END} FROM cohort
                )
            ),

            /* ---------------------------------------------------------
             * Join only the already-filtered cohort-sized relations.
             * --------------------------------------------------------- */

            base AS
            (
                SELECT
                    record_id,
                    {COL_START},
                    {COL_END},

                    cpu.sequence_length,
                    cpu.gc_content,
                    cpu.shannon_entropy,
                    cpu.is_coding_region,
                    cpu.gene_length,
                    cpu.taxonomy,
                    cpu.taxon_group,

                    lk.mean_log_prob,
                    lk.sum_log_prob,
                    lk.perplexity,
                    lk.supervised_position_count,
                    lk.min_token_logprob,
                    lk.argmin_position,
                    lk.per_token_logprob_std

                    {", lk.token_profile_length" if has_token_array else ""}
                    {", lk.token_profile_min" if has_token_array else ""}
                    {", lk.token_profile_max" if has_token_array else ""}
                    {", lk.token_profile_mean" if has_token_array else ""}

                    ,emb.embedding_norm
                    ,emb.top_embedding_dimensions
                    ,emb.top_embedding_abs_values

                    {", cpu.gene_start_position" if boundary_pair is not None else ""}
                    {", cpu.gene_end_position" if boundary_pair is not None else ""}

                FROM filtered_cpu AS cpu

                INNER JOIN filtered_likelihood AS lk
                    USING (record_id, {COL_START}, {COL_END})

                INNER JOIN filtered_embeddings AS emb
                    USING (record_id, {COL_START}, {COL_END})
            ),

            /* ---------------------------------------------------------
             * Global statistics.
             *
             * This relation contains exactly one row.
             * --------------------------------------------------------- */

            stats AS
            (
                SELECT
                    avg(gc_content) AS gc_mu,
                    stddevPop(gc_content) AS gc_sd,

                    avg(sequence_length) AS length_mu,
                    stddevPop(sequence_length) AS length_sd,

                    avg(perplexity) AS ppl_mu,
                    stddevPop(perplexity) AS ppl_sd,

                    avg(embedding_norm) AS emb_mu,
                    stddevPop(embedding_norm) AS emb_sd,

                    avg(shannon_entropy) AS entropy_mu,
                    stddevPop(shannon_entropy) AS entropy_sd,

                    avg(min_token_logprob) AS min_lp_mu,
                    stddevPop(min_token_logprob) AS min_lp_sd,

                    avg(per_token_logprob_std) AS token_std_mu,
                    stddevPop(per_token_logprob_std) AS token_std_sd

                FROM base
            ),

            scored AS
            (
                SELECT
                    b.*,

                    if(
                        s.gc_sd = 0,
                        0.,
                        (b.gc_content - s.gc_mu) / s.gc_sd
                    ) AS gc_z,

                    if(
                        s.length_sd = 0,
                        0.,
                        (
                            b.sequence_length - s.length_mu
                        ) / s.length_sd
                    ) AS length_z,

                    if(
                        s.ppl_sd = 0,
                        0.,
                        (b.perplexity - s.ppl_mu) / s.ppl_sd
                    ) AS perplexity_z,

                    if(
                        s.emb_sd = 0,
                        0.,
                        (
                            b.embedding_norm - s.emb_mu
                        ) / s.emb_sd
                    ) AS embedding_norm_z,

                    if(
                        s.entropy_sd = 0,
                        0.,
                        (
                            b.shannon_entropy - s.entropy_mu
                        ) / s.entropy_sd
                    ) AS entropy_z,

                    if(
                        s.min_lp_sd = 0,
                        0.,
                        (
                            b.min_token_logprob - s.min_lp_mu
                        ) / s.min_lp_sd
                    ) AS min_token_logprob_z,

                    if(
                        s.token_std_sd = 0,
                        0.,
                        (
                            b.per_token_logprob_std - s.token_std_mu
                        ) / s.token_std_sd
                    ) AS token_std_z

                FROM base AS b
                CROSS JOIN stats AS s
            ),

            enriched AS
            (
                SELECT
                    *,

                    (
                        abs(gc_z)
                        + abs(length_z)
                        + abs(perplexity_z)
                        + abs(embedding_norm_z)
                    ) / 4.0 AS multi_layer_anomaly_score,

                    abs(gc_z) >= {args.outlier_z}
                        AS is_gc_outlier,

                    abs(length_z) >= {args.outlier_z}
                        AS is_length_outlier,

                    abs(perplexity_z) >= {args.outlier_z}
                        AS is_likelihood_outlier,

                    abs(embedding_norm_z) >= {args.outlier_z}
                        AS is_embedding_outlier,

                    (
                        toUInt8(abs(gc_z) >= {args.outlier_z})
                        + toUInt8(abs(length_z) >= {args.outlier_z})
                        + toUInt8(abs(perplexity_z) >= {args.outlier_z})
                        + toUInt8(abs(embedding_norm_z) >= {args.outlier_z})
                    ) AS outlier_layer_count,

                    if(
                        sequence_length = 0,
                        NULL,
                        argmin_position / toFloat64(sequence_length)
                    ) AS normalized_argmin_position,

                    least(
                        greatest(
                            toFloat64(argmin_position),
                            0.0
                        ),
                        greatest(
                            toFloat64(sequence_length - 1),
                            0.0
                        )
                    ) AS bounded_argmin_position,

                    least(
                        abs(toFloat64(argmin_position)),
                        abs(
                            toFloat64(
                                sequence_length - argmin_position
                            )
                        )
                    ) AS distance_to_sequence_boundary,

                    (
                        abs(gc_z)
                        + abs(length_z)
                        + abs(perplexity_z)
                    ) / 3.0 AS non_embedding_anomaly,

                    (
                        abs(perplexity_z)
                        + abs(min_token_logprob_z)
                        + abs(token_std_z)
                    ) / 3.0 AS likelihood_signal,

                    (
                        abs(gc_z)
                        + abs(length_z)
                        + abs(entropy_z)
                        + abs(
                            toFloat64(is_coding_region) - 0.5
                        ) * 2
                    ) / 4.0 AS cpu_signal,

                    abs(embedding_norm_z) AS embedding_signal,

                    /* -------------------------------------------------
                     * Percentile ranks replace rank()+count().
                     * ------------------------------------------------- */

                    percent_rank() OVER (
                        PARTITION BY taxon_group
                        ORDER BY gc_content
                    ) AS intra_taxon_gc_percentile,

                    percent_rank() OVER (
                        PARTITION BY taxon_group
                        ORDER BY sequence_length
                    ) AS intra_taxon_length_percentile,

                    percent_rank() OVER (
                        PARTITION BY taxon_group
                        ORDER BY perplexity
                    ) AS intra_taxon_perplexity_percentile,

                    percent_rank() OVER (
                        PARTITION BY taxon_group
                        ORDER BY embedding_norm
                    ) AS intra_taxon_embedding_norm_percentile

                FROM scored
            )

            SELECT
                record_id,
                {COL_START},
                {COL_END},

                sequence_length,
                gc_content,
                shannon_entropy,
                is_coding_region,
                gene_length,
                taxonomy,
                taxon_group,

                mean_log_prob,
                sum_log_prob,
                perplexity,
                supervised_position_count,
                min_token_logprob,
                argmin_position,
                per_token_logprob_std

                {", token_profile_length" if has_token_array else ""}
                {", token_profile_min" if has_token_array else ""}
                {", token_profile_max" if has_token_array else ""}
                {", token_profile_mean" if has_token_array else ""}

                ,embedding_norm

                ,gc_z
                ,length_z
                ,perplexity_z
                ,embedding_norm_z
                ,entropy_z
                ,min_token_logprob_z
                ,token_std_z

                ,multi_layer_anomaly_score

                ,is_gc_outlier
                ,is_length_outlier
                ,is_likelihood_outlier
                ,is_embedding_outlier
                ,outlier_layer_count

                ,normalized_argmin_position
                ,bounded_argmin_position
                ,distance_to_sequence_boundary

                ,non_embedding_anomaly
                ,likelihood_signal
                ,cpu_signal
                ,embedding_signal

                ,intra_taxon_gc_percentile
                ,intra_taxon_length_percentile
                ,intra_taxon_perplexity_percentile
                ,intra_taxon_embedding_norm_percentile

                ,top_embedding_dimensions
                ,top_embedding_abs_values

                {", gene_start_position" if boundary_pair is not None else ""}
                {", gene_end_position" if boundary_pair is not None else ""}

                {final_boundary_fields}

            FROM enriched

            SETTINGS
                max_threads = {args.threads},
                max_bytes_before_external_group_by = 1073741824,
                max_bytes_before_external_sort = 1073741824
                {memory_setting}
            """

            # ==========================================================
            # Execute analytical query
            # ==========================================================

            print("Executing optimized cohort-first ClickHouse query...")
            print()
            query_started = time.time()

            result = ch.query_arrow(sql)

            query_elapsed = time.time() - query_started

            print(
                f"Rows in analytical cohort: {result.num_rows:,}"
            )
            print(
                f"Analytical query elapsed: {query_elapsed:,.1f}s"
            )

            # ==========================================================
            # Persist record-level result
            # ==========================================================

            record_csv = args.output / "record_metrics.csv"

            table_to_csv(result, record_csv)

            # Temporary Parquet is retained only while downstream
            # ClickHouse calculations execute.
            table_to_parquet(result, record_path)

            # ==========================================================
            # Summary
            # ==========================================================

            summary_sql = f"""
            SELECT
                count() AS n,

                avg(multi_layer_anomaly_score)
                    AS anomaly_mean,

                quantileTDigest(0.50)(
                    multi_layer_anomaly_score
                ) AS anomaly_median,

                quantileTDigest(0.95)(
                    multi_layer_anomaly_score
                ) AS anomaly_p95,

                quantileTDigest(0.99)(
                    multi_layer_anomaly_score
                ) AS anomaly_p99,

                max(multi_layer_anomaly_score)
                    AS anomaly_max,

                avg(gc_z) AS gc_z_mean,
                stddevPop(gc_z) AS gc_z_sd,

                avg(length_z) AS length_z_mean,
                stddevPop(length_z) AS length_z_sd,

                avg(perplexity_z) AS perplexity_z_mean,
                stddevPop(perplexity_z) AS perplexity_z_sd,

                avg(embedding_norm_z) AS embedding_z_mean,
                stddevPop(embedding_norm_z) AS embedding_z_sd,

                sum(toUInt64(is_likelihood_outlier))
                    AS likelihood_outliers,

                sum(toUInt64(is_embedding_outlier))
                    AS embedding_outliers,

                sum(toUInt64(is_gc_outlier))
                    AS gc_outliers,

                sum(toUInt64(is_length_outlier))
                    AS length_outliers,

                avg(intra_taxon_gc_percentile)
                    AS mean_taxon_gc_percentile,

                avg(intra_taxon_length_percentile)
                    AS mean_taxon_length_percentile,

                avg(intra_taxon_perplexity_percentile)
                    AS mean_taxon_perplexity_percentile,

                avg(intra_taxon_embedding_norm_percentile)
                    AS mean_taxon_embedding_norm_percentile,

                avg(normalized_argmin_position)
                    AS mean_normalized_argmin_position,

                avg(distance_to_sequence_boundary)
                    AS mean_distance_to_sequence_boundary,

                {sql_quote(
                    "available"
                    if has_token_array
                    else "summary_only_no_per_token_array"
                )} AS token_profile_status,

                {sql_quote(
                    "exact_numeric_gene_boundaries"
                    if boundary_pair is not None
                    else "not_available_current_schema"
                )} AS boundary_metric_status

            FROM file(
                {sql_quote(str(record_path))},
                Parquet
            )
            """

            summary = ch.query_arrow(summary_sql)

            table_to_csv(
                summary,
                args.output / "metric_summary.csv",
            )

            # ==========================================================
            # Information gain
            # ==========================================================

            gain_sql = f"""
            WITH data AS
            (
                SELECT
                    cpu_signal,
                    likelihood_signal,
                    embedding_signal,
                    multi_layer_anomaly_score
                FROM file(
                    {sql_quote(str(record_path))},
                    Parquet
                )
            ),

            corrs AS
            (
                SELECT
                    corr(
                        cpu_signal,
                        likelihood_signal
                    ) AS rho_cpu_likelihood,

                    corr(
                        likelihood_signal,
                        embedding_signal
                    ) AS rho_likelihood_embedding,

                    corr(
                        cpu_signal,
                        embedding_signal
                    ) AS rho_cpu_embedding,

                    corr(
                        multi_layer_anomaly_score,
                        cpu_signal
                    ) AS rho_score_cpu,

                    corr(
                        multi_layer_anomaly_score,
                        likelihood_signal
                    ) AS rho_score_likelihood,

                    corr(
                        multi_layer_anomaly_score,
                        embedding_signal
                    ) AS rho_score_embedding

                FROM data
            )

            SELECT
                'CPU → Likelihood' AS transition,

                rho_cpu_likelihood AS correlation,

                greatest(
                    0.,
                    1. - rho_cpu_likelihood * rho_cpu_likelihood
                ) AS incremental_information_gain_proxy,

                greatest(
                    0.,
                    1. - rho_cpu_likelihood * rho_cpu_likelihood
                ) AS cumulative_information_gain_proxy

            FROM corrs

            UNION ALL

            SELECT
                'Likelihood → Embedding',

                rho_likelihood_embedding,

                greatest(
                    0.,
                    1. -
                    rho_likelihood_embedding
                    * rho_likelihood_embedding
                ),

                greatest(
                    0.,
                    1. -
                    rho_cpu_likelihood
                    * rho_cpu_likelihood
                )
                +
                greatest(
                    0.,
                    1. -
                    rho_likelihood_embedding
                    * rho_likelihood_embedding
                )

            FROM corrs

            UNION ALL

            SELECT
                'CPU → Embedding',

                rho_cpu_embedding,

                greatest(
                    0.,
                    1. -
                    rho_cpu_embedding
                    * rho_cpu_embedding
                ),

                greatest(
                    0.,
                    1. -
                    rho_cpu_embedding
                    * rho_cpu_embedding
                )

            FROM corrs
            """

            gain = ch.query_arrow(gain_sql)

            table_to_csv(
                gain,
                args.output / "information_gain.csv",
            )

            # ==========================================================
            # Top anomalies
            # ==========================================================

            top_sql = f"""
            SELECT
                record_id,
                {COL_START},
                {COL_END},
                taxon_group,

                sequence_length,
                gc_content,
                perplexity,
                embedding_norm,

                multi_layer_anomaly_score,

                gc_z,
                length_z,
                perplexity_z,
                embedding_norm_z,

                is_gc_outlier,
                is_length_outlier,
                is_likelihood_outlier,
                is_embedding_outlier,

                outlier_layer_count,

                normalized_argmin_position,
                distance_to_sequence_boundary,

                boundary_method,

                top_embedding_dimensions,
                top_embedding_abs_values

            FROM file(
                {sql_quote(str(record_path))},
                Parquet
            )

            ORDER BY
                multi_layer_anomaly_score DESC,
                outlier_layer_count DESC

            LIMIT 1000

            SETTINGS
                max_threads = {args.threads},
                max_bytes_before_external_sort = 1073741824
            """

            top = ch.query_arrow(top_sql)

            table_to_csv(
                top,
                args.output / "top_anomalies.csv",
            )

            # ==========================================================
            # Metadata
            # ==========================================================

            metadata = {
                "phase": "4.5",

                "metric_set": [
                    "multi_layer_anomaly_score",
                    "cumulative_information_gain_proxy",
                    "token_level_likelihood_profile_diagnostics",
                    "layer_by_layer_comparison",
                    "intra_taxon_rank",
                    "boundary_crossing_detection_diagnostics",
                    "feature_contribution_analysis",
                    "comparative_outlier_status",
                ],

                "knn_used": False,
                "pretraining_corpus_used": False,

                "cohort_key": [
                    "record_id",
                    "start",
                    "end",
                ],

                "cohort_source": str(common_cohort_path),
                "cohort_rows": len(unique_keys),

                "taxon_expression": taxonomy_expr,
                "taxon_index": args.taxon_index,

                "outlier_z_threshold": args.outlier_z,
                "top_embedding_dimensions": top_dims,

                "clickhouse_threads": args.threads,
                "clickhouse_max_memory_bytes": (
                    args.max_memory_bytes or None
                ),

                "optimization": {
                    "cohort_filtering": (
                        "Each large HF source is independently "
                        "restricted using composite-key IN membership "
                        "before joining."
                    ),
                    "cohort_key_sorted": (
                        "Cohort keys parquet is written pre-sorted by "
                        "(record_id, start, end) to keep the IN-subquery "
                        "hash-set build cache-friendly."
                    ),
                    "reserved_word_quoting": (
                        "`start`/`end` are ClickHouse reserved words "
                        "and are backtick-quoted everywhere they are "
                        "used as identifiers."
                    ),
                    "embedding_memory": (
                        "Full embedding array is discarded immediately "
                        "after top-K dimension extraction."
                    ),
                    "embedding_sort": (
                        "arrayPartialReverseSort is used for top-K "
                        "embedding coordinates."
                    ),
                    "taxonomic_percentiles": (
                        "percent_rank window functions replace "
                        "rank plus count window calculations."
                    ),
                    "external_spill_threshold_bytes": 1073741824,
                },

                "likelihood_has_per_token_logprob": has_token_array,

                "boundary_numeric_pair": boundary_pair,

                "boundary_metric_note": (
                    "Exact gene-boundary crossing available."
                    if boundary_pair
                    else (
                        "Current CPU schema does not expose numeric "
                        "gene boundary coordinates; sequence-boundary "
                        "proxy fields are emitted."
                    )
                ),

                "information_gain_note": (
                    "Uses 1-rho^2 as an incremental novelty proxy "
                    "between layer signals. It is not Shannon "
                    "mutual information."
                ),

                "embedding_feature_contribution_note": (
                    "Top dimensions are ranked by absolute embedding "
                    "coordinate magnitude. They are not claimed to "
                    "be biologically interpretable features."
                ),

                "elapsed_seconds": time.time() - started,
            }

            (args.output / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            # ==========================================================
            # Cleanup temporary files
            # ==========================================================

            keys_path.unlink(missing_ok=True)
            record_path.unlink(missing_ok=True)

            # ==========================================================
            # Final report
            # ==========================================================

            print()
            print("Phase 4.5 Outputs:")
            print(f"  {args.output / 'record_metrics.csv'}")
            print(f"  {args.output / 'metric_summary.csv'}")
            print(f"  {args.output / 'information_gain.csv'}")
            print(f"  {args.output / 'top_anomalies.csv'}")
            print(f"  {args.output / 'run_metadata.json'}")
            print()
            print(
                f"Total elapsed: {time.time() - started:,.1f}s"
            )

            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

        # Keep the temporary cohort keys if debugging is necessary,
        # but remove them on normal failures where possible.
        return 1


if __name__ == "__main__":
    sys.exit(main())