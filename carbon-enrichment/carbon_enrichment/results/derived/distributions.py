"""
Phase 2 distribution analysis for the Carbon corpus.

Architecture
------------

    Faceberg / Iceberg
            |
          DuckDB
            |
      distributions.py
            |
      analysis DataFrames
            |
       visualization

This module is an analytical consumer of already-materialized datasets.
It is deliberately NOT a Dagster asset and does not create another heavy
corpus artifact.

For the published CPU-enriched corpus, DuckDB performs aggregate
statistics over the complete catalog table. Only visualization-oriented
rows are bounded in memory.

The existing local-Parquet entry point is retained for development and
offline validation.

Phase 2 scope
-------------

CPU-enriched population:

    numeric distributions
    categorical distributions
    missingness
    sampling-stratum distributions
    bounded visualization samples

The four sampling dimensions established by enrichment.py/sampling.py
are treated as canonical:

    sequence_length bucket proxy
    is_coding_region
    strand
    taxonomy_domain
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
from tqdm.auto import tqdm

from carbon_enrichment.assets.cpu.streaming import _read_local_parquet
from carbon_enrichment.resources.faceberg import (
    PIPELINE_TABLES,
    PipelineCatalog,
)

# ============================================================================
# Phase 2 configuration
# ============================================================================

NUMERIC_COLUMNS = (
    "sequence_length",
    "gene_length",
    "gc_content",
    "gc_skew",
    "shannon_entropy",
    "taxonomy_depth",
)

CATEGORICAL_COLUMNS = (
    "gene_type",
    "species_type",
    "strand",
    "molecule_type",
    "topology",
    "taxonomy",
    "taxonomy_domain",
    "is_coding_region",
    "qc_flag",
)

VISUALIZATION_COLUMNS = (
    "record_id",
    "sequence_length",
    "gene_length",
    "gc_content",
    "gc_skew",
    "shannon_entropy",
    "taxonomy_depth",
    "gene_type",
    "species_type",
    "strand",
    "molecule_type",
    "topology",
    "taxonomy",
    "taxonomy_domain",
    "is_coding_region",
    "qc_flag",
)

DEFAULT_VISUALIZATION_SAMPLE_SIZE = 100_000
DEFAULT_RANDOM_SEED = 42

SAMPLING_STRATA = (
    "length_bucket",
    "is_coding_region",
    "strand",
    "taxonomy_domain",
)

SAMPLING_DRIFT_WARNING_THRESHOLD_PCT = 2.0


# ============================================================================
# SQL helpers
# ============================================================================


def _table(catalog: PipelineCatalog, node_id: str) -> str:
    """Return a DuckDB-visible catalog table name."""

    if node_id not in PIPELINE_TABLES:
        raise KeyError(
            f"{node_id!r} is not a catalog-managed table. "
            f"Available: {sorted(PIPELINE_TABLES)}"
        )

    table = PIPELINE_TABLES[node_id]["table"]

    # PipelineCatalog.query() exposes the Faceberg catalog as `cat`.
    return f"cat.{table}"


def length_bucket_expression(field: str = "sequence_length") -> str:
    """Return the exact sampling.py length-bucket proxy."""

    return f"""
        CASE
            WHEN {field} <= 512 THEN '<=512'
            WHEN {field} <= 2048 THEN '513-2048'
            WHEN {field} <= 8192 THEN '2049-8192'
            WHEN {field} <= 32768 THEN '8193-32768'
            ELSE '>32768'
        END
    """


# ============================================================================
# Catalog-backed numeric statistics
# ============================================================================


def catalog_numeric_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
    columns: tuple[str, ...] = NUMERIC_COLUMNS,
) -> Any:
    """Compute complete-population numeric statistics through DuckDB.

    Every valid value in the catalog table contributes. No visualization
    sample is involved.
    """

    if not columns:
        raise ValueError("columns must not be empty")

    expressions: list[str] = []

    for column in columns:
        expressions.extend(
            [
                f"COUNT({column}) AS count_{column}",
                f"COUNT(*) - COUNT({column}) AS missing_{column}",
                f"AVG({column}) AS mean_{column}",
                f"STDDEV_POP({column}) AS std_{column}",
                f"MIN({column}) AS min_{column}",
                f"MAX({column}) AS max_{column}",
            ]
        )

    sql = f"""
        SELECT
            COUNT(*) AS rows_processed,
            {", ".join(expressions)}
        FROM {_table(catalog, node_id)}
    """

    return catalog.query(sql)


# ============================================================================
# Catalog-backed categorical statistics
# ============================================================================


def catalog_categorical_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
    columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
) -> dict[str, Any]:
    """Compute complete-population categorical frequencies.

    Each categorical field is queried independently. This avoids the
    combinatorial explosion that would result from grouping all
    categorical columns together.
    """

    result: dict[str, Any] = {}

    for column in columns:
        sql = f"""
            SELECT
                CAST({column} AS VARCHAR) AS value,
                COUNT(*) AS count,
                COUNT(*) * 100.0
                    / SUM(COUNT(*)) OVER () AS percentage
            FROM {_table(catalog, node_id)}
            GROUP BY {column}
            ORDER BY count DESC, value
        """

        result[column] = catalog.query(sql)

    return result


# ============================================================================
# Sampling-stratum distributions
# ============================================================================


def catalog_sampling_stratum_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
) -> dict[str, Any]:
    """Compute the four canonical sampling-stratum distributions."""

    table = _table(catalog, node_id)

    queries = {
        "length_bucket": f"""
            SELECT
                {length_bucket_expression()} AS value,
                COUNT(*) AS count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()
                    AS percentage
            FROM {table}
            GROUP BY 1
            ORDER BY
                CASE value
                    WHEN '<=512' THEN 1
                    WHEN '513-2048' THEN 2
                    WHEN '2049-8192' THEN 3
                    WHEN '8193-32768' THEN 4
                    WHEN '>32768' THEN 5
                END
        """,
        "is_coding_region": f"""
            SELECT
                CAST(is_coding_region AS VARCHAR) AS value,
                COUNT(*) AS count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()
                    AS percentage
            FROM {table}
            GROUP BY is_coding_region
            ORDER BY count DESC, value
        """,
        "strand": f"""
            SELECT
                CAST(strand AS VARCHAR) AS value,
                COUNT(*) AS count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()
                    AS percentage
            FROM {table}
            GROUP BY strand
            ORDER BY count DESC, value
        """,
        "taxonomy_domain": f"""
            SELECT
                CAST(taxonomy_domain AS VARCHAR) AS value,
                COUNT(*) AS count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()
                    AS percentage
            FROM {table}
            GROUP BY taxonomy_domain
            ORDER BY count DESC, value
        """,
    }

    return {name: catalog.query(sql) for name, sql in queries.items()}


# ============================================================================
# Complete Phase 2 catalog analysis
# ============================================================================


@dataclass
class CatalogDistributionAnalysis:
    """Complete Phase 2 analysis backed by the published catalog."""

    node_id: str
    rows_processed: int
    numeric: Any
    categorical: dict[str, Any]
    sampling_strata: dict[str, Any]
    visualization_sample: list[dict[str, Any]]
    visualization_sample_size: int
    random_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _catalog_visualization_sample(
    catalog: PipelineCatalog,
    *,
    node_id: str,
    sample_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Return a bounded deterministic-ish sample for visualization.

    The sample is intentionally separate from population statistics.

    DuckDB's hash ordering makes the selection stable for a fixed source
    table and seed while avoiding a full in-memory load. This is not the
    same sampling operation used to construct sampled_cpu; it is solely a
    visualization sample.
    """

    if sample_size < 0:
        raise ValueError("sample_size must be >= 0")

    if sample_size == 0:
        return []

    columns = ", ".join(VISUALIZATION_COLUMNS)

    # DuckDB hash() is used only to bound visualization rows. This must not
    # be confused with sampling.py's SHA-256 corpus-selection procedure.
    sql = f"""
        SELECT
            {columns}
        FROM {_table(catalog, node_id)}
        ORDER BY
            hash(
                CAST(record_id AS VARCHAR) || '|' || '{random_seed}'
            )
        LIMIT {int(sample_size)}
    """

    df = catalog.query(sql)

    return df.to_dict(orient="records")


def analyze_catalog_distributions(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
    visualization_sample_size: int = DEFAULT_VISUALIZATION_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> CatalogDistributionAnalysis:
    """Analyze a catalog-backed CPU-enriched population.

    Population statistics are calculated over the complete table by
    DuckDB. Only the visualization subset is bounded.

    No artifact is written by this function.
    """

    row_count = catalog.query(
        f"""
        SELECT COUNT(*) AS n
        FROM {_table(catalog, node_id)}
        """
    )

    rows_processed = int(row_count["n"].iloc[0])

    numeric = catalog_numeric_statistics(
        catalog,
        node_id=node_id,
    )

    categorical = catalog_categorical_statistics(
        catalog,
        node_id=node_id,
    )

    sampling_strata = catalog_sampling_stratum_statistics(
        catalog,
        node_id=node_id,
    )

    visualization_sample = _catalog_visualization_sample(
        catalog,
        node_id=node_id,
        sample_size=visualization_sample_size,
        random_seed=random_seed,
    )

    return CatalogDistributionAnalysis(
        node_id=node_id,
        rows_processed=rows_processed,
        numeric=numeric,
        categorical=categorical,
        sampling_strata=sampling_strata,
        visualization_sample=visualization_sample,
        visualization_sample_size=len(visualization_sample),
        random_seed=random_seed,
        metadata={
            "population_scope": "complete_catalog_table",
            "population_statistics_are_not_sampled": True,
            "visualization_sample_is_analysis_subset": True,
            "visualization_sample_method": ("bounded DuckDB hash ordering"),
            "visualization_sample_is_not_corpus_sampling": True,
            "sampling_strata": SAMPLING_STRATA,
            "sampling_drift_warning_threshold_pct": (
                SAMPLING_DRIFT_WARNING_THRESHOLD_PCT
            ),
        },
    )


# ============================================================================
# Local streaming analysis
# ============================================================================


@dataclass
class NumericAccumulator:
    """Streaming statistics for one numeric feature."""

    count: int = 0
    missing: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: Iterable[Any]) -> None:
        """Update the accumulator from a sequence of values."""

        for value in values:
            if value is None:
                self.missing += 1
                continue

            try:
                x = float(value)
            except (TypeError, ValueError):
                self.missing += 1
                continue

            if not x == x or x in (float("inf"), float("-inf")):
                self.missing += 1
                continue

            self.count += 1

            if self.minimum is None or x < self.minimum:
                self.minimum = x

            if self.maximum is None or x > self.maximum:
                self.maximum = x

            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self) -> float | None:
        """Return population variance."""

        if self.count == 0:
            return None

        return self.m2 / self.count

    @property
    def standard_deviation(self) -> float | None:
        """Return population standard deviation."""

        variance = self.variance

        if variance is None:
            return None

        return variance**0.5

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable statistics."""

        return {
            "count": self.count,
            "missing": self.missing,
            "mean": self.mean if self.count else None,
            "std": self.standard_deviation,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class CategoricalAccumulator:
    """Streaming frequency counts for one categorical feature."""

    count: int = 0
    missing: int = 0
    values: Counter[str] = field(default_factory=Counter)

    def update(self, values: Iterable[Any]) -> None:
        """Update categorical counts from a sequence of values."""

        for value in values:
            if value is None:
                self.missing += 1
                continue

            value_string = str(value)

            if not value_string:
                self.missing += 1
                continue

            self.count += 1
            self.values[value_string] += 1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable categorical statistics."""

        return {
            "count": self.count,
            "missing": self.missing,
            "unique": len(self.values),
            "values": {
                value: {
                    "count": count,
                    "percentage": (count / self.count * 100 if self.count else 0.0),
                }
                for value, count in self.values.most_common()
            },
        }


class ReservoirSampler:
    """Bounded reservoir sample for local visualization.

    This sample is NOT used for corpus statistics.
    """

    def __init__(
        self,
        sample_size: int,
        *,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        if sample_size < 0:
            raise ValueError("sample_size must be >= 0")

        import random

        self.sample_size = sample_size
        self.random = random.Random(seed)
        self._rows: list[dict[str, Any]] = []
        self._seen = 0

    def update(self, rows: Iterable[dict[str, Any]]) -> None:
        """Consider rows for inclusion in the reservoir."""

        for row in rows:
            self._seen += 1

            if self.sample_size == 0:
                continue

            if len(self._rows) < self.sample_size:
                self._rows.append(row)
                continue

            replacement_index = self.random.randrange(self._seen)

            if replacement_index < self.sample_size:
                self._rows[replacement_index] = row

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    @property
    def seen(self) -> int:
        return self._seen


@dataclass
class DistributionAnalysis:
    """Complete local Phase 2 distribution analysis result."""

    rows_processed: int
    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, Any]]
    visualization_sample: list[dict[str, Any]]
    visualization_sample_size: int
    random_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# Arrow helpers
# ============================================================================


def _column_values(
    batch: pa.RecordBatch,
    column: str,
) -> list[Any]:
    """Return one Arrow column as a Python list."""

    index = batch.schema.get_field_index(column)

    if index < 0:
        raise KeyError(
            f"Required column {column!r} not found in batch. "
            f"Available columns: {batch.schema.names}"
        )

    return batch.column(index).to_pylist()


def _batch_to_visualization_rows(
    batch: pa.RecordBatch,
) -> list[dict[str, Any]]:
    """Extract only plot-relevant fields from an Arrow batch."""

    available = set(batch.schema.names)

    missing = [column for column in VISUALIZATION_COLUMNS if column not in available]

    if missing:
        raise KeyError(f"Missing visualization columns: {missing}")

    selected = batch.select(VISUALIZATION_COLUMNS)

    return selected.to_pylist()


# ============================================================================
# Local streaming analysis
# ============================================================================


def analyze_distributions(
    batches: Iterable[pa.RecordBatch],
    *,
    total_rows: int | None = None,
    visualization_sample_size: int = DEFAULT_VISUALIZATION_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    show_progress: bool = True,
) -> DistributionAnalysis:
    """Analyze a complete local CPU-enriched corpus in streaming mode.

    Every record contributes to numeric and categorical statistics.
    Only the bounded visualization reservoir is sampled.
    """

    numeric = {column: NumericAccumulator() for column in NUMERIC_COLUMNS}

    categorical = {column: CategoricalAccumulator() for column in CATEGORICAL_COLUMNS}

    reservoir = ReservoirSampler(
        visualization_sample_size,
        seed=random_seed,
    )

    rows_processed = 0
    iterator = batches

    if show_progress:
        progress = tqdm(
            total=total_rows,
            desc="Phase 2: CPU feature analysis",
            unit="records",
        )
    else:
        progress = None

    try:
        for batch in iterator:
            if batch.num_rows == 0:
                continue

            batch_rows = batch.num_rows
            rows_processed += batch_rows

            for column, accumulator in numeric.items():
                accumulator.update(_column_values(batch, column))

            for column, accumulator in categorical.items():
                accumulator.update(_column_values(batch, column))

            reservoir.update(_batch_to_visualization_rows(batch))

            if progress is not None:
                progress.update(batch_rows)

    finally:
        if progress is not None:
            progress.close()

    return DistributionAnalysis(
        rows_processed=rows_processed,
        numeric={
            column: accumulator.to_dict() for column, accumulator in numeric.items()
        },
        categorical={
            column: accumulator.to_dict() for column, accumulator in categorical.items()
        },
        visualization_sample=reservoir.rows,
        visualization_sample_size=len(reservoir.rows),
        random_seed=random_seed,
        metadata={
            "streaming": True,
            "population_scope": "complete_cpu_enriched_corpus",
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "visualization_columns": VISUALIZATION_COLUMNS,
            "visualization_sample_is_analysis_subset": True,
            "writes_heavy_artifact": False,
        },
    )


def analyze_cpu_enriched_corpus(
    input_dir: str | Path,
    *,
    batch_size: int,
    total_rows: int | None = None,
    visualization_sample_size: int = DEFAULT_VISUALIZATION_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    show_progress: bool = True,
) -> DistributionAnalysis:
    """Analyze an existing CPU-enriched local Parquet corpus.

    This is retained as a development/offline path. It does not perform
    CPU enrichment and does not write another corpus artifact.
    """

    batches = _read_local_parquet(
        input_dir=input_dir,
        batch_size=batch_size,
    )

    return analyze_distributions(
        batches,
        total_rows=total_rows,
        visualization_sample_size=visualization_sample_size,
        random_seed=random_seed,
        show_progress=show_progress,
    )
