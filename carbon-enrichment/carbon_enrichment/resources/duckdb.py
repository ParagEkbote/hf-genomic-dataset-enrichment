from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from carbon_enrichment.resources.resource_logging import (
    get_logger,
    timed_operation,
)


logger = get_logger("duckdb")


# ----------------------------------------------------------------------
# Local execution defaults
# ----------------------------------------------------------------------

DEFAULT_THREADS = max(
    1,
    (os.cpu_count() or 1) - 1,
)

DEFAULT_MEMORY_LIMIT = None

DEFAULT_TEMP_DIRECTORY = Path(
    "data/tmp/duckdb"
)


@dataclass(frozen=True)
class DuckDBConfig:
    """Configuration for a local DuckDB resource."""

    database_path: str | Path = ":memory:"
    threads: int | None = DEFAULT_THREADS
    memory_limit: str | None = DEFAULT_MEMORY_LIMIT
    temp_directory: str | Path | None = DEFAULT_TEMP_DIRECTORY
    read_only: bool = False


class DuckDBResource:
    """
    Thin resource wrapper around a local DuckDB connection.

    Responsibilities:
      - connection lifecycle
      - DuckDB execution configuration
      - Parquet view registration
      - analytical query execution

    Analytical logic belongs in consuming modules such as
    distributions.py.
    """

    def __init__(
        self,
        config: DuckDBConfig | None = None,
    ) -> None:
        self.config = config or DuckDBConfig()

        self._connection: (
            duckdb.DuckDBPyConnection | None
        ) = None

        self._registered_datasets: dict[
            str,
            Path,
        ] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def get_connection(
        self,
    ) -> duckdb.DuckDBPyConnection:
        """Create and return the DuckDB connection, initializing once."""
        if self._connection is not None:
            return self._connection

        with timed_operation(
            logger,
            "duckdb",
            "connection_initialize",
            database=str(
                self.config.database_path
            ),
            read_only=self.config.read_only,
        ):
            self._connection = duckdb.connect(
                database=str(
                    self.config.database_path
                ),
                read_only=self.config.read_only,
            )

            self._configure_connection()

        return self._connection

    def _configure_connection(self) -> None:
        """Apply resource-level execution settings."""
        connection = self._require_connection()

        if self.config.threads is not None:
            if self.config.threads < 1:
                raise ValueError(
                    "threads must be >= 1"
                )

            connection.execute(
                f"SET threads = {self.config.threads}"
            )

        if self.config.memory_limit is not None:
            connection.execute(
                "SET memory_limit = ?",
                [self.config.memory_limit],
            )

        if self.config.temp_directory is not None:
            temp_directory = (
                Path(
                    self.config.temp_directory
                )
                .expanduser()
            )

            temp_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            connection.execute(
                "SET temp_directory = ?",
                [str(temp_directory)],
            )

    def close(self) -> None:
        """Close the DuckDB connection if it is open."""
        if self._connection is None:
            return

        with timed_operation(
            logger,
            "duckdb",
            "connection_close",
        ):
            self._connection.close()
            self._connection = None

    def __enter__(self) -> DuckDBResource:
        self.get_connection()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        path: str | Path,
    ) -> None:
        """Register a source Parquet dataset as a DuckDB view."""
        self._validate_relation_name(name)

        dataset_path = (
            Path(path)
            .expanduser()
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Parquet dataset does not exist: "
                f"{dataset_path}"
            )

        connection = self.get_connection()

        parquet_expression = (
            self._parquet_expression(
                dataset_path
            )
        )

        with timed_operation(
            logger,
            "duckdb",
            "register_dataset",
            name=name,
            path=str(dataset_path),
        ):
            connection.execute(
                f"""
                CREATE OR REPLACE VIEW "{name}" AS
                SELECT *
                FROM read_parquet({parquet_expression})
                """
            )

        self._registered_datasets[name] = dataset_path

    def register_enriched_glob(
        self,
        name: str,
        directory: str | Path,
        pattern: str = "batch_*.parquet",
    ) -> None:
        """
        Register a directory of enriched batch Parquet files
        as one DuckDB view.
        """
        self._validate_relation_name(name)

        directory = (
            Path(directory)
            .expanduser()
        )

        if not directory.exists():
            raise FileNotFoundError(
                f"Enriched output directory does not exist: "
                f"{directory}"
            )

        glob_path = str(
            directory / pattern
        )

        connection = self.get_connection()

        with timed_operation(
            logger,
            "duckdb",
            "register_enriched_glob",
            name=name,
            directory=str(directory),
            pattern=pattern,
        ):
            connection.execute(
                f"""
                CREATE OR REPLACE VIEW "{name}" AS
                SELECT *
                FROM read_parquet(?)
                """,
                [glob_path],
            )

        self._registered_datasets[name] = directory

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_arrow(
        self,
        sql: str,
        parameters: list[Any]
        | tuple[Any, ...]
        | None = None,
    ):
        """Execute SQL and return the result as an Arrow table."""
        connection = self.get_connection()

        with timed_operation(
            logger,
            "duckdb",
            "query_arrow",
        ) as timing:
            result = (
                connection
                .execute(
                    sql,
                    parameters or [],
                )
                .fetch_arrow_table()
            )

            timing.metadata[
                "rows_returned"
            ] = result.num_rows

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def registered_datasets(
        self,
    ) -> dict[str, Path]:
        """Return a copy of currently registered datasets."""
        return dict(
            self._registered_datasets
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connection(
        self,
    ) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            raise RuntimeError(
                "DuckDB connection has not been initialized."
            )

        return self._connection

    @staticmethod
    def _validate_relation_name(
        name: str,
    ) -> None:
        if not name:
            raise ValueError(
                "Dataset name cannot be empty."
            )

        if not name.replace(
            "_",
            "",
        ).isalnum():
            raise ValueError(
                f"Invalid DuckDB relation name: "
                f"{name!r}"
            )

    @staticmethod
    def _parquet_expression(
        path: Path,
    ) -> str:
        escaped = str(path).replace(
            "'",
            "''",
        )

        return f"'{escaped}'"