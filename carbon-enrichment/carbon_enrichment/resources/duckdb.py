"""
DuckDB analytics layer for the Carbon pipeline.

This module owns SQL-based provenance validation and reusable feature
statistics. Faceberg.PipelineCatalog owns the catalog / Iceberg access
layer.

Phase 1 validates the declared lineage:

    pretraining split
        -> cpu_enriched
        -> sampled_cpu
        -> tokenized
        -> {likelihood_stats, embeddings}

The important sampling edge is:

    cpu_enriched
        -> sampled_cpu

where sampled_cpu is the stratified CPU-enriched population used as the
GPU input corpus.

The sampling assertion is based on the declared strata:

    - length bucket proxy
    - coding status
    - strand
    - taxonomy domain

The physical biological columns are not renamed. Derived sampling
dimensions are computed in DuckDB.

NOTE:
The exact length-bucket boundaries and exact coding-status rule must
match the implementation that produced pilot_corpus_dedup. They are
therefore explicit configuration below rather than silently invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faceberg import PIPELINE_TABLES, PipelineCatalog


# ============================================================================
# Sampling configuration
# ============================================================================

# These are the four conceptual strata used by the sampling operation.
SAMPLING_STRATA: tuple[str, ...] = (
    "length_bucket",
    "coding_status",
    "strand",
    "taxonomy_domain",
)


# ---------------------------------------------------------------------------
# Derived expressions
# ---------------------------------------------------------------------------

# The source schema contains start/end coordinates, so sequence length can
# be derived without reading the sequence payload.
#
# This expression intentionally returns the coordinate span:
#
#     end - start
#
# The bucket boundaries themselves are NOT invented here.
#
# Replace LENGTH_BUCKET_EXPR with the exact bucket expression used by the
# original deterministic sampling job once those boundaries are confirmed.

LENGTH_EXPR = """
    GREATEST(
        CAST(end AS BIGINT) - CAST(start AS BIGINT),
        0
    )
"""


# IMPORTANT:
# Replace this with the exact coding-status rule used by the sampler.
#
# This default is deliberately based on the existing CPU-enriched field
# `is_coding_region`, because the current pipeline already references that
# field. If the original sampler instead used gene_type, change only this
# expression.
CODING_STATUS_EXPR = """
    CASE
        WHEN is_coding_region IS TRUE THEN 'coding'
        WHEN is_coding_region IS FALSE THEN 'non_coding'
        ELSE 'unknown'
    END
"""


# Taxonomy domain is derived from the physical taxonomy string.
#
# Example:
#
#     Eukaryota;Fungi;Dikarya;...
#
# becomes:
#
#     Eukaryota
#
# This preserves the canonical `taxonomy` column rather than renaming it.
TAXONOMY_DOMAIN_EXPR = """
    NULLIF(
        TRIM(
            SPLIT_PART(
                COALESCE(taxonomy, ''),
                ';',
                1
            )
        ),
        ''
    )
"""


# ============================================================================
# Length bucket configuration
# ============================================================================

# Set these to the exact boundaries used by the original sampling job.
#
# Example shape only:
#
#     LENGTH_BUCKET_BOUNDARIES = (
#         (100, "lt_100"),
#         (1000, "100_999"),
#         ...
#     )
#
# They are intentionally empty until the actual sampling configuration
# is available. This prevents the provenance validator from claiming
# that an arbitrary bucketing scheme reproduces the historical sample.

LENGTH_BUCKET_BOUNDARIES: tuple[tuple[int, str], ...] = ()


def length_bucket_sql() -> str:
    """Return the exact SQL expression for the sampling length bucket.

    The bucket definition must be supplied from the actual sampling
    implementation before the Phase 1 sampling assertion is considered
    fully reproducible.
    """

    if not LENGTH_BUCKET_BOUNDARIES:
        raise RuntimeError(
            "LENGTH_BUCKET_BOUNDARIES is not configured. "
            "Set it to the exact length-bucket boundaries used by "
            "the original stratified sampling job."
        )

    clauses: list[str] = []

    for upper_bound, label in LENGTH_BUCKET_BOUNDARIES:
        clauses.append(
            f"WHEN {LENGTH_EXPR} < {upper_bound} "
            f"THEN '{label}'"
        )

    last_label = LENGTH_BUCKET_BOUNDARIES[-1][1]

    return (
        "CASE\n"
        + "\n".join(f"        {clause}" for clause in clauses)
        + f"\n        ELSE '{last_label}_plus'\n"
        + "    END"
    )


# ============================================================================
# Connection helper
# ============================================================================


def get_connection(
    config: dict[str, Any] | None = None,
) -> Any:
    """Return an independent DuckDB connection with Iceberg loaded."""

    import duckdb

    con = duckdb.connect(config=config or {})

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
# Derived sampling dimensions
# ============================================================================


def sampling_projection(
    *,
    source_table: str,
) -> str:
    """Build the SQL SELECT projection containing sampling dimensions.

    The physical dataset columns remain untouched. The sampling fields
    are derived only for validation/analytics.
    """

    return f"""
        SELECT
            *,
            {length_bucket_sql()} AS length_bucket,
            {CODING_STATUS_EXPR} AS coding_status,
            {TAXONOMY_DOMAIN_EXPR} AS taxonomy_domain
        FROM {source_table}
    """


def sampling_stratum_proportions(
    catalog: PipelineCatalog,
    *,
    node_id: str,
) -> dict[str, Any]:
    """Calculate proportions for the four declared sampling strata."""

    table = _table(node_id)

    sql = f"""
        WITH derived AS (
            {sampling_projection(source_table=table)}
        )
        SELECT
            length_bucket,
            coding_status,
            strand,
            taxonomy_domain,
            COUNT(*) AS n,
            COUNT(*) * 1.0
                / SUM(COUNT(*)) OVER () AS proportion
        FROM derived
        GROUP BY
            length_bucket,
            coding_status,
            strand,
            taxonomy_domain
        ORDER BY
            length_bucket,
            coding_status,
            strand,
            taxonomy_domain
    """

    return {
        "joint": catalog.query(sql)
    }


def stratum_proportions(
    catalog: PipelineCatalog,
    *,
    node_id: str,
) -> dict[str, Any]:
    """Return independent distribution proportions for each sampler stratum.

    This is the primary Phase 1 representation for comparing
    cpu_enriched against sampled_cpu.
    """

    table = _table(node_id)

    sql = f"""
        WITH derived AS (
            {sampling_projection(source_table=table)}
        )
        SELECT
            stratum_name,
            stratum,
            COUNT(*) AS n,
            COUNT(*) * 1.0
                / SUM(COUNT(*)) OVER (
                    PARTITION BY stratum_name
                ) AS proportion
        FROM (
            SELECT
                'length_bucket' AS stratum_name,
                CAST(length_bucket AS VARCHAR) AS stratum
            FROM derived

            UNION ALL

            SELECT
                'coding_status' AS stratum_name,
                CAST(coding_status AS VARCHAR) AS stratum
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
        )
        GROUP BY
            stratum_name,
            stratum
        ORDER BY
            stratum_name,
            stratum
    """

    df = catalog.query(sql)

    result: dict[str, dict[Any, float]] = {}

    for field in SAMPLING_STRATA:
        subset = df[df["stratum_name"] == field]

        result[field] = (
            subset
            .set_index("stratum")["proportion"]
            .to_dict()
        )

    return result


# ============================================================================
# Phase 1 — lineage checks
# ============================================================================


@dataclass(frozen=True)
class EdgeCheck:
    """Result of a declared lineage-edge consistency check."""

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
    pretraining_split_row_count: int = 46_300_000,
    expected_fraction: float = 0.70,
) -> EdgeCheck:
    """Validate CPU enrichment coverage of the pretraining split.

    The ~46.3M source population is the pretraining split, not the
    complete pretraining corpus.

    The expected CPU-enriched population is ~70% of that split.
    """

    split_rows = catalog.query(
        f"""
        SELECT COUNT(*) AS n
        FROM {_table("cpu_enriched")}
        """
    )["n"].iloc[0]

    split_rows = int(split_rows)

    observed_fraction = (
        split_rows / pretraining_split_row_count
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
            "cpu_enriched_row_count": split_rows,
            "observed_fraction": observed_fraction,
        },
    )


def sampling_proportion_drift(
    catalog: PipelineCatalog,
) -> EdgeCheck:
    """Compare declared sampling strata before and after sampling.

    The intended assertion is that the sampled CPU population preserves
    the CPU-enriched distribution according to the four sampling strata.

    This is different from asserting that deduplication alone preserved
    distributions.
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

    for field in SAMPLING_STRATA:
        cpu_dist = cpu[field]
        sampled_dist = sampled[field]

        keys = set(cpu_dist) | set(sampled_dist)

        if not keys:
            drift[field] = None
            continue

        drift[field] = max(
            abs(
                cpu_dist.get(key, 0.0)
                - sampled_dist.get(key, 0.0)
            )
            for key in keys
        )

    return EdgeCheck(
        edge="cpu_enriched_to_sampled_cpu",
        expected={
            "method": (
                "deterministic per-row hash + "
                "proportional stratified sampling"
            ),
            "strata": list(SAMPLING_STRATA),
        },
        observed={
            "max_absolute_proportion_drift": drift,
            "cpu": cpu,
            "sampled": sampled,
        },
    )


def row_count_relationship(
    catalog: PipelineCatalog,
    *,
    upstream: str,
    downstream: str,
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
            "relationship": (
                "downstream <= upstream "
                "under sampling/token-budget selection"
            ),
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
    """Check that the two GPU output datasets have matching row counts."""

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
    """Run the complete Phase 1 lineage-validation suite."""

    checks: list[EdgeCheck] = []

    checks.append(
        pretraining_split_to_cpu_enriched(catalog)
    )

    checks.append(
        sampling_proportion_drift(catalog)
    )

    checks.append(
        row_count_relationship(
            catalog,
            upstream="cpu_enriched",
            downstream="sampled_cpu",
        )
    )

    checks.append(
        row_count_relationship(
            catalog,
            upstream="sampled_cpu",
            downstream="tokenized",
        )
    )

    checks.append(
        fanout_consistency(catalog)
    )

    return {
        check.edge: check.to_dict()
        for check in checks
    }


# ============================================================================
# Phase 2+ — feature statistics
# ============================================================================


def feature_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str,
    numeric_fields: tuple[str, ...],
    group_by: str | None = None,
) -> Any:
    """Compute reusable numeric feature statistics."""

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