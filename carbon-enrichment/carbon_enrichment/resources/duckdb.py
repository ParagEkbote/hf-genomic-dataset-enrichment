"""
DuckDB analytics and provenance-validation layer for the Carbon pipeline.

Faceberg.PipelineCatalog owns the Iceberg/catalog access layer.
This module owns SQL-based Phase 1 lineage checks and reusable Phase 2+
feature statistics.

Declared lineage:

    pretraining corpus
        -> pretraining split
        -> cpu_enriched
        -> sampled_cpu
        -> tokenized
        -> {likelihood_stats, embeddings}

The CPU sampling edge is:

    cpu_enriched
        -> sampled_cpu

Sampling implementation, as defined by sampling.py:

    deterministic SHA-256 per-row hash
    + uniform inclusion fraction

with representativeness measured across:

    - sequence-length bucket proxy
    - is_coding_region
    - strand
    - taxonomy_domain

The four sampling dimensions already exist in cpu_enriched because they
are materialized by enrichment.py:

    sequence_length
    is_coding_region
    strand
    taxonomy_domain

Therefore DuckDB validates the same physical columns rather than
inventing alternate schema names.

The length proxy buckets are:

    <= 512
    <= 2048
    <= 8192
    <= 32768
    > 32768

The sampler's representativeness check uses a 2.0 percentage-point
warning threshold. It is a warning rather than a hard sampling gate.

Important bounded-input caveat:
sampling.py validates the composition of the bounded source actually
consumed by a run. A small validation run does not establish
representativeness of the complete ~32.4M-row CPU-enriched population.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from faceberg import (
    PIPELINE_TABLES,
    PipelineCatalog,
    SAMPLING_DRIFT_WARNING_THRESHOLD_PCT,
    SAMPLING_STRATIFICATION_VARIABLES,
)


# ============================================================================
# Constants
# ============================================================================

PRETRAINING_SPLIT_ROW_COUNT = 46_300_000
CPU_ENRICHED_EXPECTED_FRACTION = 0.70

LENGTH_PROXY_BUCKET_LABELS: tuple[str, ...] = (
    "<=512",
    "513-2048",
    "2049-8192",
    "8193-32768",
    ">32768",
)


# ============================================================================
# Connection helper
# ============================================================================


def get_connection(
    config: dict[str, Any] | None = None,
) -> Any:
    """Return an independent DuckDB connection with Iceberg loaded."""

    import duckdb

    con =  cast(Any, duckdb).connect(config=config or {})
    con.execute("INSTALL iceberg; LOAD iceberg")
    return con


# ============================================================================
# Table helper
# ============================================================================


def _table(node_id: str) -> str:
    """Return the fully-qualified attached catalog table name."""

    if node_id not in PIPELINE_TABLES:
        raise KeyError(
            f"{node_id!r} is not a catalog-managed table"
        )

    return f"cat.{PIPELINE_TABLES[node_id]['table']}"


# ============================================================================
# Sampling-derived expressions
# ============================================================================


def length_bucket_expression(
    field: str = "sequence_length",
) -> str:
    """Return the exact SQL equivalent of sampling.py's proxy buckets.

    sampling.py uses upper bounds 512, 2048, 8192 and 32768, and -1 for
    the final infinity bucket. Here we use human-readable labels while
    preserving the same boundaries.
    """

    return f"""
        CASE
            WHEN {field} <= 512 THEN '<=512'
            WHEN {field} <= 2048 THEN '513-2048'
            WHEN {field} <= 8192 THEN '2049-8192'
            WHEN {field} <= 32768 THEN '8193-32768'
            ELSE '>32768'
        END
    """


def sampling_dimensions_sql(
    *,
    table: str,
) -> str:
    """Return a SELECT that exposes the exact sampling dimensions."""

    return f"""
        SELECT
            {length_bucket_expression("sequence_length")}
                AS length_bucket,
            is_coding_region,
            strand,
            taxonomy_domain
        FROM {table}
    """


# ============================================================================
# Proportion helpers
# ============================================================================


def _independent_stratum_proportions_sql(
    *,
    table: str,
) -> str:
    """Build independent proportion distributions for each sampler field."""

    return f"""
        WITH derived AS (
            {sampling_dimensions_sql(table=table)}
        ),

        strata AS (
            SELECT
                'length_bucket' AS stratum_name,
                CAST(length_bucket AS VARCHAR) AS stratum
            FROM derived

            UNION ALL

            SELECT
                'is_coding_region' AS stratum_name,
                CAST(is_coding_region AS VARCHAR) AS stratum
            FROM derived

            UNION ALL

            SELECT
                'strand' AS stratum_name,
                CAST(strand AS VARCHAR) AS stratum
            FROM derived

            UNION ALL

            SELECT
                'taxonomy_domain' AS stratum_name,
                CAST(taxonomy_domain AS VARCHAR) AS stratum
            FROM derived
        ),

        counts AS (
            SELECT
                stratum_name,
                stratum,
                COUNT(*) AS n
            FROM strata
            GROUP BY stratum_name, stratum
        )

        SELECT
            stratum_name,
            stratum,
            n,
            n * 1.0
                / SUM(n) OVER (PARTITION BY stratum_name)
                AS proportion
        FROM counts
        ORDER BY stratum_name, stratum
    """


def stratum_proportions(
    catalog: PipelineCatalog,
    *,
    node_id: str,
) -> dict[str, dict[str, float]]:
    """Return independent proportions for all declared sampling strata."""

    df = catalog.query(
        _independent_stratum_proportions_sql(
            table=_table(node_id),
        )
    )

    result: dict[str, dict[str, float]] = {}

    for field in (
        "length_bucket",
        "is_coding_region",
        "strand",
        "taxonomy_domain",
    ):
        subset = df[df["stratum_name"] == field]

        result[field] = {
            str(key): float(value)
            for key, value in zip(
                subset["stratum"],
                subset["proportion"],
            )
        }

    return result


def sampling_stratum_counts(
    catalog: PipelineCatalog,
    *,
    node_id: str,
) -> Any:
    """Return joint counts across the four sampling dimensions."""

    table = _table(node_id)

    return catalog.query(
        f"""
        WITH derived AS (
            {sampling_dimensions_sql(table=table)}
        )
        SELECT
            length_bucket,
            is_coding_region,
            strand,
            taxonomy_domain,
            COUNT(*) AS n
        FROM derived
        GROUP BY
            length_bucket,
            is_coding_region,
            strand,
            taxonomy_domain
        ORDER BY
            length_bucket,
            is_coding_region,
            strand,
            taxonomy_domain
        """
    )


# ============================================================================
# Phase 1 — lineage checks
# ============================================================================


@dataclass(frozen=True)
class EdgeCheck:
    """Result of one lineage-edge consistency check."""

    edge: str
    expected: dict[str, Any]
    observed: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge": self.edge,
            "expected": self.expected,
            "observed": self.observed,
        }


def pretraining_split_to_cpu_enriched(
    catalog: PipelineCatalog,
    *,
    pretraining_split_row_count: int = PRETRAINING_SPLIT_ROW_COUNT,
    expected_fraction: float = CPU_ENRICHED_EXPECTED_FRACTION,
) -> EdgeCheck:
    """Validate CPU-enriched coverage of the pretraining split.

    The 46.3M-row source is the pretraining split. The expected
    cpu_enriched population is approximately 70% of that split.

    The 75% corpus-to-pretraining-split relationship is a separate
    lineage edge and is not measured by this catalog function.
    """

    df = catalog.query(
        f"""
        SELECT COUNT(*) AS n
        FROM {_table("cpu_enriched")}
        """
    )

    cpu_rows = int(df["n"].iloc[0])

    observed_fraction = (
        cpu_rows / pretraining_split_row_count
        if pretraining_split_row_count
        else None
    )

    return EdgeCheck(
        edge="pretraining_split_to_cpu_enriched",
        expected={
            "fraction": expected_fraction,
            "source": "pretraining split",
            "pretraining_split_row_count": pretraining_split_row_count,
        },
        observed={
            "pretraining_split_row_count": pretraining_split_row_count,
            "cpu_enriched_row_count": cpu_rows,
            "observed_fraction": observed_fraction,
        },
    )


def sampling_proportion_drift(
    catalog: PipelineCatalog,
) -> EdgeCheck:
    """Compare CPU-enriched and sampled proportions.

    This reproduces the conceptual representativeness check in
    sampling.py, but against the currently cataloged datasets.

    Unlike sampling.py's in-run check, this compares the published
    cpu_enriched and sampled_cpu artifacts.
    """

    cpu = stratum_proportions(
        catalog,
        node_id="cpu_enriched",
    )

    sampled = stratum_proportions(
        catalog,
        node_id="sampled_cpu",
    )

    drift: dict[str, float | None] = {}

    for field in (
        "length_bucket",
        "is_coding_region",
        "strand",
        "taxonomy_domain",
    ):
        cpu_dist = cpu[field]
        sampled_dist = sampled[field]

        keys = set(cpu_dist) | set(sampled_dist)

        drift[field] = (
            max(
                abs(
                    cpu_dist.get(key, 0.0)
                    - sampled_dist.get(key, 0.0)
                )
                for key in keys
            )
            if keys
            else None
        )

    warning_fields = [
        field
        for field, value in drift.items()
        if value is not None
        and value * 100.0 > SAMPLING_DRIFT_WARNING_THRESHOLD_PCT
    ]

    return EdgeCheck(
        edge="cpu_enriched_to_sampled_cpu_sampling",
        expected={
            "method": "deterministic_per_row_hash",
            "sampling_strata": list(SAMPLING_STRATIFICATION_VARIABLES),
            "drift_warning_threshold_pct": (
                SAMPLING_DRIFT_WARNING_THRESHOLD_PCT
            ),
        },
        observed={
            "max_absolute_proportion_drift": drift,
            "warning_fields": warning_fields,
            "cpu": cpu,
            "sampled": sampled,
        },
    )


def row_count_relationship(
    catalog: PipelineCatalog,
    *,
    upstream: str,
    downstream: str,
    expected_relationship: str = "downstream <= upstream",
) -> EdgeCheck:
    """Compare upstream/downstream row counts."""

    df = catalog.query(
        f"""
        SELECT
            (
                SELECT COUNT(*)
                FROM {_table(upstream)}
            ) AS upstream_n,

            (
                SELECT COUNT(*)
                FROM {_table(downstream)}
            ) AS downstream_n
        """
    )

    upstream_n = int(df["upstream_n"].iloc[0])
    downstream_n = int(df["downstream_n"].iloc[0])

    return EdgeCheck(
        edge=f"{upstream}_to_{downstream}",
        expected={
            "relationship": expected_relationship,
        },
        observed={
            "upstream_n": upstream_n,
            "downstream_n": downstream_n,
            "ratio": (
                downstream_n / upstream_n
                if upstream_n
                else None
            ),
            "looks_like_full_passthrough": (
                downstream_n == upstream_n
            ),
        },
    )


def fanout_consistency(
    catalog: PipelineCatalog,
    *,
    parent: str = "tokenized",
    children: tuple[str, ...] = (
        "likelihood_stats",
        "embeddings",
    ),
) -> EdgeCheck:
    """Check that GPU-output row counts agree."""

    counts: dict[str, int] = {}

    for child in children:
        df = catalog.query(
            f"""
            SELECT COUNT(*) AS n
            FROM {_table(child)}
            """
        )
        counts[child] = int(df["n"].iloc[0])

    match = len(set(counts.values())) == 1

    return EdgeCheck(
        edge=f"{parent}_fanout",
        expected={
            "children_row_counts_should_match": True,
        },
        observed={
            "counts": counts,
            "match": match,
        },
    )


# ============================================================================
# Phase 1 suite
# ============================================================================


def run_phase1_checks(
    catalog: PipelineCatalog,
) -> dict[str, Any]:
    """Run the Phase 1 lineage-validation suite."""

    checks: list[EdgeCheck] = [
        pretraining_split_to_cpu_enriched(catalog),
        sampling_proportion_drift(catalog),
        row_count_relationship(
            catalog,
            upstream="cpu_enriched",
            downstream="sampled_cpu",
            expected_relationship=(
                "sampled CPU population <= CPU-enriched population "
                "under deterministic sampling"
            ),
        ),
        row_count_relationship(
            catalog,
            upstream="sampled_cpu",
            downstream="tokenized",
            expected_relationship=(
                "tokenized population <= sampled CPU population "
                "under downstream processing/selection"
            ),
        ),
        fanout_consistency(catalog),
    ]

    return {
        check.edge: check.to_dict()
        for check in checks
    }


# ============================================================================
# Phase 2+ — reusable feature statistics
# ============================================================================


def feature_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str,
    numeric_fields: tuple[str, ...],
    group_by: str | None = None,
) -> Any:
    """Compute AVG/MIN/MAX/STDDEV for numeric fields."""

    aggs = ", ".join(
        (
            f"AVG({field}) AS avg_{field}, "
            f"MIN({field}) AS min_{field}, "
            f"MAX({field}) AS max_{field}, "
            f"STDDEV({field}) AS stddev_{field}"
        )
        for field in numeric_fields
    )

    group_clause = (
        f"GROUP BY {group_by}"
        if group_by
        else ""
    )

    select_group = (
        f"{group_by}, "
        if group_by
        else ""
    )

    return catalog.query(
        f"""
        SELECT
            {select_group}
            {aggs}
        FROM {_table(node_id)}
        {group_clause}
        """
    )


# ============================================================================
# Output helper
# ============================================================================


def write_json(
    data: dict[str, Any],
    path: str,
) -> None:
    """Write a small Python dictionary as JSON."""

    import json

    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )
