from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
from tqdm.auto import tqdm

from carbon_enrichment.assets.cpu.streaming import _read_local_parquet


# ============================================================================
# Phase 2 configuration
# ============================================================================

# These are the CPU-enriched numeric features we want to characterize.
NUMERIC_COLUMNS = (
    "sequence_length",
    "gene_length",
    "gc_content",
    "gc_skew",
    "shannon_entropy",
    "taxonomy_depth",
)

# These are the categorical features we want to characterize.
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

# Columns retained for the bounded visualization sample.
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


# ============================================================================
# Streaming numeric statistics
# ============================================================================


@dataclass
class NumericAccumulator:
    """Streaming statistics for one numeric feature.

    Statistics are calculated over every valid value encountered in the
    complete CPU-enriched corpus.
    """

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

            # Welford's online algorithm.
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


# ============================================================================
# Streaming categorical statistics
# ============================================================================


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
                    "percentage": (
                        count / self.count * 100
                        if self.count
                        else 0.0
                    ),
                }
                for value, count in self.values.most_common()
            },
        }


# ============================================================================
# Deterministic visualization reservoir
# ============================================================================


class ReservoirSampler:
    """Bounded reservoir sample for visualization.

    This sample is NOT used for corpus statistics. It exists only to provide
    a manageable set of records for scatter/joint plots.
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


# ============================================================================
# Analysis result
# ============================================================================


@dataclass
class DistributionAnalysis:
    """Complete Phase 2 distribution analysis result."""

    rows_processed: int

    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, Any]]

    visualization_sample: list[dict[str, Any]]

    visualization_sample_size: int
    random_seed: int

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

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

    missing = [
        column
        for column in VISUALIZATION_COLUMNS
        if column not in available
    ]

    if missing:
        raise KeyError(
            "Missing visualization columns: "
            f"{missing}"
        )

    selected = batch.select(VISUALIZATION_COLUMNS)

    return selected.to_pylist()


# ============================================================================
# Main Phase 2 analysis
# ============================================================================


def analyze_distributions(
    batches: Iterable[pa.RecordBatch],
    *,
    total_rows: int | None = None,
    visualization_sample_size: int = DEFAULT_VISUALIZATION_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    show_progress: bool = True,
) -> DistributionAnalysis:
    """Analyze the complete CPU-enriched corpus in streaming mode.

    Every record contributes to numeric and categorical statistics.

    Only the bounded visualization reservoir is sampled.

    Parameters
    ----------
    batches:
        Iterable of Arrow RecordBatches from the CPU-enriched corpus.

    total_rows:
        Optional total number of CPU-enriched records. When supplied,
        tqdm reports percentage completion and ETA.

    visualization_sample_size:
        Maximum number of records retained for downstream visualization.

    random_seed:
        Seed controlling the reproducible visualization reservoir.

    show_progress:
        Display a tqdm progress bar when True.
    """

    numeric = {
        column: NumericAccumulator()
        for column in NUMERIC_COLUMNS
    }

    categorical = {
        column: CategoricalAccumulator()
        for column in CATEGORICAL_COLUMNS
    }

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

            # ---------------------------------------------------------------
            # Numeric statistics — ALL records
            # ---------------------------------------------------------------

            for column, accumulator in numeric.items():
                accumulator.update(
                    _column_values(batch, column)
                )

            # ---------------------------------------------------------------
            # Categorical statistics — ALL records
            # ---------------------------------------------------------------

            for column, accumulator in categorical.items():
                accumulator.update(
                    _column_values(batch, column)
                )

            # ---------------------------------------------------------------
            # Visualization reservoir — bounded sample only
            # ---------------------------------------------------------------

            reservoir.update(
                _batch_to_visualization_rows(batch)
            )

            if progress is not None:
                progress.update(batch_rows)

    finally:
        if progress is not None:
            progress.close()

    return DistributionAnalysis(
        rows_processed=rows_processed,
        numeric={
            column: accumulator.to_dict()
            for column, accumulator in numeric.items()
        },
        categorical={
            column: accumulator.to_dict()
            for column, accumulator in categorical.items()
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
            "visualization_sample_is_analysis_subset": False,
        },
    )


# ============================================================================
# Convenience entry point for the local CPU-enriched corpus
# ============================================================================


def analyze_cpu_enriched_corpus(
    input_dir: str | Path,
    *,
    batch_size: int,
    total_rows: int | None = None,
    visualization_sample_size: int = DEFAULT_VISUALIZATION_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    show_progress: bool = True,
) -> DistributionAnalysis:
    """Analyze an existing CPU-enriched Parquet corpus.

    The corpus is read through the project's existing bounded-memory
    Parquet reader. No CPU enrichment is performed here.
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