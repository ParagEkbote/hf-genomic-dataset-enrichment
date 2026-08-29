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

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
from tqdm.auto import tqdm

from carbon_enrichment.assets.cpu.streaming import _read_local_parquet
from carbon_enrichment.resources.faceberg import (
    PIPELINE_TABLES,
    PipelineCatalog,
    SAMPLING_DRIFT_WARNING_THRESHOLD_PCT,
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

LENGTH_PROXY_BUCKET_LABELS = (
    "<=512",
    "513-2048",
    "2049-8192",
    "8193-32768",
    ">32768",
)

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
# Catalog-backed shard-wise analysis
# ============================================================================

DEFAULT_DISTRIBUTIONS_OUTPUT_DIR = (
    Path.cwd() / "results" / "derived" / "distributions"
)

DEFAULT_DISTRIBUTIONS_OUTPUT_PATH = (
    DEFAULT_DISTRIBUTIONS_OUTPUT_DIR
    / "cpu_enriched_distributions.json"
)


def _catalog_shards(
    catalog: PipelineCatalog,
    *,
    node_id: str,
) -> list[str]:
    """Discover the Parquet shards represented by a catalog table."""

    if node_id not in PIPELINE_TABLES:
        raise KeyError(
            f"{node_id!r} is not a catalog-managed table. "
            f"Available: {sorted(PIPELINE_TABLES)}"
        )

    from faceberg.catalog import discover_dataset

    spec = PIPELINE_TABLES[node_id]
    info = discover_dataset(
        repo_id=spec["repo"],
        config=spec["config"],
    )

    return [str(parquet_file.uri) for parquet_file in info.files]


def _quote_string(value: str) -> str:
    """Quote a SQL string literal for DuckDB."""

    return "'" + value.replace("'", "''") + "'"


def _merge_numeric(
    accumulator: dict[str, dict[str, Any]],
    row: dict[str, Any],
    column: str,
) -> None:
    """Merge one shard's numeric aggregate into the population."""

    count = int(row[f"count_{column}"] or 0)
    missing = int(row[f"missing_{column}"] or 0)
    current = accumulator[column]
    current["missing"] += missing

    if count == 0:
        return

    mean = float(row[f"mean_{column}"])
    sumsq = float(row[f"sumsq_{column}"] or 0.0)
    shard_m2 = max(0.0, sumsq - count * mean * mean)

    if current["count"] == 0:
        current.update(
            {
                "count": count,
                "mean": mean,
                "m2": shard_m2,
                "min": row[f"min_{column}"],
                "max": row[f"max_{column}"],
            }
        )
        return

    old_count = current["count"]
    old_mean = current["mean"]
    total_count = old_count + count
    delta = mean - old_mean

    current["m2"] += (
        shard_m2
        + delta * delta * old_count * count / total_count
    )
    current["mean"] = (
        old_mean + delta * count / total_count
    )
    current["count"] = total_count

    shard_min = row[f"min_{column}"]
    shard_max = row[f"max_{column}"]

    if shard_min is not None:
        current["min"] = (
            shard_min
            if current["min"] is None
            else min(current["min"], shard_min)
        )

    if shard_max is not None:
        current["max"] = (
            shard_max
            if current["max"] is None
            else max(current["max"], shard_max)
        )


def _finalize_numeric(
    accumulator: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return JSON-serializable numeric statistics."""

    result: dict[str, dict[str, Any]] = {}

    for column, stats in accumulator.items():
        count = int(stats["count"])

        result[column] = {
            "count": count,
            "missing": int(stats["missing"]),
            "mean": stats["mean"] if count else None,
            "std": (
                (stats["m2"] / count) ** 0.5
                if count
                else None
            ),
            "min": stats["min"],
            "max": stats["max"],
        }

    return result


def _merge_histogram(
    accumulator: Counter[str],
    histogram: Any,
) -> None:
    """Merge a DuckDB histogram/map result."""

    if histogram is None or not hasattr(histogram, "items"):
        return

    for value, count in histogram.items():
        if value is None or str(value) == "":
            continue
        accumulator[str(value)] += int(count)


class _CatalogReservoirSampler:
    """Bounded reservoir sample collected during the shard scan."""

    def __init__(self, sample_size: int, seed: int) -> None:
        if sample_size < 0:
            raise ValueError("sample_size must be >= 0")

        import random

        self.sample_size = sample_size
        self.random = random.Random(seed)
        self.rows: list[dict[str, Any]] = []
        self.seen = 0

    def update(self, rows: list[dict[str, Any]]) -> None:
        """Update the reservoir with rows from one shard."""

        for row in rows:
            self.seen += 1

            if self.sample_size == 0:
                continue

            if len(self.rows) < self.sample_size:
                self.rows.append(row)
                continue

            index = self.random.randrange(self.seen)
            if index < self.sample_size:
                self.rows[index] = row


def _counts_to_records(
    counts: Counter[str],
) -> list[dict[str, Any]]:
    """Convert counts to visualization/JSON records."""

    total = sum(counts.values())

    return [
        {
            "value": value,
            "count": count,
            "percentage": (
                count * 100.0 / total
                if total
                else 0.0
            ),
        }
        for value, count in counts.most_common()
    ]


def _catalog_shard_aggregate_sql(shard_uri: str) -> str:
    """Build the single aggregate query executed per remote shard."""

    expressions: list[str] = ["COUNT(*) AS rows_processed"]

    for column in NUMERIC_COLUMNS:
        expressions.extend(
            [
                f"COUNT({column}) AS count_{column}",
                f"COUNT(*) - COUNT({column}) AS missing_{column}",
                f"AVG({column}) AS mean_{column}",
                f"SUM({column} * {column}) AS sumsq_{column}",
                f"MIN({column}) AS min_{column}",
                f"MAX({column}) AS max_{column}",
            ]
        )

    for column in CATEGORICAL_COLUMNS:
        expressions.append(
            f"histogram(CAST({column} AS VARCHAR)) "
            f"AS histogram_{column}"
        )

    expressions.append(
        f"histogram({length_bucket_expression()}) "
        "AS histogram_length_bucket"
    )

    for column in (
        "is_coding_region",
        "strand",
        "taxonomy_domain",
    ):
        expressions.append(
            f"histogram(CAST({column} AS VARCHAR)) "
            f"AS histogram_{column}_sampling"
        )

    return f"""
        SELECT
            {", ".join(expressions)}
        FROM {_quote_string(shard_uri)}
    """


def _catalog_shard_analysis(
    catalog: PipelineCatalog,
    *,
    node_id: str,
    visualization_sample_size: int = DEFAULT_VISUALIZATION_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[
    int,
    int,
    dict[str, dict[str, Any]],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    list[dict[str, Any]],
]:
    """Scan every remote shard once and merge all population statistics."""

    shards = _catalog_shards(catalog, node_id=node_id)

    numeric = {
        column: {
            "count": 0,
            "missing": 0,
            "mean": 0.0,
            "m2": 0.0,
            "min": None,
            "max": None,
        }
        for column in NUMERIC_COLUMNS
    }

    categorical = {
        column: Counter()
        for column in CATEGORICAL_COLUMNS
    }

    sampling = {
        name: Counter()
        for name in SAMPLING_STRATA
    }

    reservoir = _CatalogReservoirSampler(
        visualization_sample_size,
        random_seed,
    )

    from carbon_enrichment.resources.duckdb import get_connection

    con = get_connection()
    total_rows = 0
    processed_shards = 0

    try:
        with tqdm(
            shards,
            desc="CPU-enriched shards",
            unit="shard",
        ) as progress:
            for shard_uri in progress:
                from time import perf_counter

                shard_start = perf_counter()

                result = con.execute(
                    _catalog_shard_aggregate_sql(shard_uri)
                ).fetchdf()

                row = result.iloc[0].to_dict()
                rows = int(row["rows_processed"])

                if visualization_sample_size > 0:
                    sample_sql = f"""
                        SELECT {", ".join(VISUALIZATION_COLUMNS)}
                        FROM {_quote_string(shard_uri)}
                        USING SAMPLE reservoir({max(1, visualization_sample_size // max(1, len(shards)))} ROWS) 
                        REPEATABLE ({random_seed})
                    """
                    sample_df = con.execute(sample_sql).fetchdf()
                    reservoir.update(
                        sample_df.to_dict(orient="records")
                    )

                total_rows += rows
                processed_shards += 1

                for column in NUMERIC_COLUMNS:
                    _merge_numeric(
                        numeric,
                        row,
                        column,
                    )

                for column in CATEGORICAL_COLUMNS:
                    _merge_histogram(
                        categorical[column],
                        row[f"histogram_{column}"],
                    )

                _merge_histogram(
                    sampling["length_bucket"],
                    row["histogram_length_bucket"],
                )

                for column in (
                    "is_coding_region",
                    "strand",
                    "taxonomy_domain",
                ):
                    _merge_histogram(
                        sampling[column],
                        row[f"histogram_{column}_sampling"],
                    )

                progress.set_postfix(
                    shard_time=(
                        f"{perf_counter() - shard_start:.2f}s"
                    ),
                    rows=f"{rows:,}",
                    total=f"{total_rows:,}",
                )
    finally:
        con.close()

    return (
        total_rows,
        processed_shards,
        numeric,
        categorical,
        sampling,
        reservoir.rows,
    )


# ============================================================================
# Catalog-backed public statistics
# ============================================================================


def catalog_numeric_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
    columns: tuple[str, ...] = NUMERIC_COLUMNS,
) -> Any:
    """Compute complete-population numeric statistics."""

    if not columns:
        raise ValueError("columns must not be empty")

    (
        rows_processed,
        _,
        numeric,
        _,
        _,
        _,
    ) = _catalog_shard_analysis(
        catalog,
        node_id=node_id,
    )

    import pandas as pd

    finalized = _finalize_numeric(numeric)
    row: dict[str, Any] = {
        "rows_processed": rows_processed,
    }

    for column in columns:
        for key, value in finalized[column].items():
            row[f"{key}_{column}"] = value

    return pd.DataFrame([row])


def catalog_categorical_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
    columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
) -> dict[str, Any]:
    """Compute complete-population categorical frequencies."""

    if not columns:
        raise ValueError("columns must not be empty")

    (
        _,
        _,
        _,
        categorical,
        _,
        _,
    ) = _catalog_shard_analysis(
        catalog,
        node_id=node_id,
    )

    import pandas as pd

    return {
        column: pd.DataFrame(
            _counts_to_records(categorical[column]),
            columns=["value", "count", "percentage"],
        )
        for column in columns
    }


def catalog_sampling_stratum_statistics(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
) -> dict[str, Any]:
    """Compute the four canonical sampling-stratum distributions."""

    (
        _,
        _,
        _,
        _,
        sampling,
        _,
    ) = _catalog_shard_analysis(
        catalog,
        node_id=node_id,
    )

    import pandas as pd

    result = {
        name: pd.DataFrame(
            _counts_to_records(sampling[name]),
            columns=["value", "count", "percentage"],
        )
        for name in SAMPLING_STRATA
    }

    order = {
        label: index
        for index, label in enumerate(
            LENGTH_PROXY_BUCKET_LABELS
        )
    }

    length_df = result["length_bucket"]

    if not length_df.empty:
        length_df["_order"] = (
            length_df["value"]
            .map(lambda x: order.get(x, len(order)))
        )
        result["length_bucket"] = (
            length_df
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    return result


# ============================================================================
# Complete Phase 2 catalog analysis
# ============================================================================


@dataclass
class CatalogDistributionAnalysis:
    """Complete Phase 2 analysis backed by the published catalog."""

    node_id: str
    rows_processed: int
    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, list[dict[str, Any]]]
    sampling_strata: dict[str, list[dict[str, Any]]]
    visualization_sample: list[dict[str, Any]]
    visualization_sample_size: int
    random_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _catalog_visualization_sample(
    catalog: PipelineCatalog,
    *,
    node_id: str,
    sample_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Compatibility helper; primary analysis collects the sample in-pass."""

    del catalog, node_id, random_seed
    if sample_size < 0:
        raise ValueError("sample_size must be >= 0")
    return []


def _json_default(value: Any) -> Any:
    """Serialize common NumPy/Pandas scalar values."""

    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def save_catalog_distribution_analysis(
    analysis: CatalogDistributionAnalysis,
    *,
    output_path: str | Path = DEFAULT_DISTRIBUTIONS_OUTPUT_PATH,
) -> Path:
    """Persist distribution results for later visualization."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            analysis.to_dict(),
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    return path


def analyze_catalog_distributions(
    catalog: PipelineCatalog,
    *,
    node_id: str = "cpu_enriched",
    visualization_sample_size: int = DEFAULT_VISUALIZATION_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    output_path: str | Path | None = (
        DEFAULT_DISTRIBUTIONS_OUTPUT_PATH
    ),
) -> CatalogDistributionAnalysis:
    """Analyze the catalog population and persist a visualization artifact.

    Population distributions are computed from one aggregate query per
    remote Parquet shard. The visualization sample is bounded separately.
    """

    (
        rows_processed,
        processed_shards,
        numeric_accumulator,
        categorical_accumulator,
        sampling_accumulator,
        visualization_sample,
    ) = _catalog_shard_analysis(
        catalog,
        node_id=node_id,
        visualization_sample_size=visualization_sample_size,
        random_seed=random_seed,
    )

    table = catalog.load_table(node_id)

    analysis = CatalogDistributionAnalysis(
        node_id=node_id,
        rows_processed=rows_processed,
        numeric=_finalize_numeric(numeric_accumulator),
        categorical={
            column: _counts_to_records(
                categorical_accumulator[column]
            )
            for column in CATEGORICAL_COLUMNS
        },
        sampling_strata={
            name: _counts_to_records(
                sampling_accumulator[name]
            )
            for name in SAMPLING_STRATA
        },
        visualization_sample=visualization_sample,
        visualization_sample_size=len(visualization_sample),
        random_seed=random_seed,
        metadata={
            "population_scope": "complete_catalog_table",
            "population_statistics_are_not_sampled": True,
            "visualization_sample_is_analysis_subset": True,
            "visualization_sample_method": (
                "bounded DuckDB hash ordering"
            ),
            "visualization_sample_is_not_corpus_sampling": True,
            "sampling_strata": SAMPLING_STRATA,
            "sampling_drift_warning_threshold_pct": (
                SAMPLING_DRIFT_WARNING_THRESHOLD_PCT
            ),
            "catalog_scan_mode": (
                "one_aggregate_query_per_remote_shard"
            ),
            "shards_processed": processed_shards,
            "dataset_repo": PIPELINE_TABLES[node_id]["repo"],
            "dataset_config": PIPELINE_TABLES[node_id]["config"],
            "dataset_revision": str(
                table.properties.get(
                    "hf.dataset.revision",
                    "",
                )
            ),
            "analysis_timestamp_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
        },
    )

    if output_path is not None:
        path = save_catalog_distribution_analysis(
            analysis,
            output_path=output_path,
        )

        print()
        print("=" * 60)
        print("Distribution analysis complete")
        print("=" * 60)
        print(f"Node:             {node_id}")
        print(f"Shards processed: {processed_shards:,}")
        print(f"Rows processed:   {rows_processed:,}")
        print(f"Output:           {path}")

    return analysis


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


def main() -> None:
    """Run the complete catalog-backed distribution analysis."""

    catalog = PipelineCatalog.local("./carbon-catalog")

    analyze_catalog_distributions(
        catalog,
        node_id="cpu_enriched",
        visualization_sample_size=DEFAULT_VISUALIZATION_SAMPLE_SIZE,
        random_seed=DEFAULT_RANDOM_SEED,
        output_path=DEFAULT_DISTRIBUTIONS_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()