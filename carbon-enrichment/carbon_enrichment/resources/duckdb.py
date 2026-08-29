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

from typing import Any, cast

from carbon_enrichment.resources.faceberg import PIPELINE_TABLES

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
# Connection
# ============================================================================


def get_connection(
    config: dict[str, Any] | None = None,
) -> Any:
    """Return an independent DuckDB connection with Iceberg loaded."""

    import duckdb

    con = cast(Any, duckdb).connect(config=config or {})

    con.execute("INSTALL iceberg")
    con.execute("LOAD iceberg")

    con.execute(
        """
        CREATE SECRET IF NOT EXISTS hf_token (
            TYPE huggingface,
            PROVIDER credential_chain
        )
        """
    )

    # Create an in-memory DuckDB database named `cat`.
    con.execute("ATTACH ':memory:' AS cat")

    return con


# ============================================================================
# Table helpers
# ============================================================================


def _table(node_id: str) -> str:
    """Return the logical SQL table name for a catalog-managed node."""

    if node_id not in PIPELINE_TABLES:
        raise KeyError(f"{node_id!r} is not a catalog-managed table")

    return f"cat.{PIPELINE_TABLES[node_id]['table']}"


def quote_identifier(identifier: str) -> str:
    """Safely quote a DuckDB SQL identifier."""

    return '"' + identifier.replace('"', '""') + '"'


def quote_string(value: str) -> str:
    """Safely quote a DuckDB SQL string literal."""

    return "'" + value.replace("'", "''") + "'"


# ============================================================================
# Iceberg registration
# ============================================================================


def register_iceberg_view(
    con: Any,
    *,
    database: str,
    schema: str,
    table: str,
    metadata_location: str,
) -> None:
    """Register an Iceberg metadata file as a DuckDB view."""

    con.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS
        {quote_identifier(database)}.{quote_identifier(schema)}
        """
    )

    qualified_table = (
        f"{quote_identifier(database)}."
        f"{quote_identifier(schema)}."
        f"{quote_identifier(table)}"
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW {qualified_table} AS
        SELECT *
        FROM iceberg_scan({quote_string(metadata_location)})
        """
    )


def register_catalog_table(
    con: Any,
    *,
    node_id: str,
    metadata_location: str,
) -> str:
    """Expose one Faceberg table through DuckDB."""

    if node_id not in PIPELINE_TABLES:
        raise KeyError(
            f"{node_id!r} is not a catalog-managed table"
        )

    identifier = PIPELINE_TABLES[node_id]["table"]

    try:
        schema, table = identifier.split(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid catalog table identifier: {identifier!r}"
        ) from exc

    register_iceberg_view(
        con,
        database="cat",
        schema=schema,
        table=table,
        metadata_location=metadata_location,
    )

    return f"cat.{identifier}"


# ============================================================================
# Sampling-derived expressions
# ============================================================================


def length_bucket_expression(
    field: str = "sequence_length",
) -> str:
    """Return the exact SQL equivalent of sampling.py's proxy buckets."""

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
    """Return a SELECT exposing the exact sampling dimensions."""

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
    catalog: Any,
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
    catalog: Any,
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
