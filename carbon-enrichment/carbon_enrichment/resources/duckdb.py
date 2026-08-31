"""
DuckDB resource for local Parquet datasets.

Responsibilities
----------------
- Create and configure a local DuckDB connection.
- Register local Parquet datasets as logical relations.
- Execute analytical SQL.
- Return compact query results to callers.
- Provide schema inspection.
- Record operation timing and throughput.

This module intentionally contains no Phase 2/3/4 analytical logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from carbon_enrichment.resources.resource_logging import get_logger, timed_operation


logger = get_logger("duckdb")


@dataclass(frozen=True)
class DuckDBConfig:
    """Configuration for a local DuckDB resource."""

    database_path: str | Path = ":memory:"
    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: str | Path | None = None
    read_only: bool = False


class DuckDBResource:
    """
    Thin resource wrapper around a local DuckDB connection.

    The resource provides access primitives only. Analytical logic
    belongs in the derived layer.
    """

    def __init__(
        self,
        config: DuckDBConfig | None = None,
    ) -> None:
        self.config = config or DuckDBConfig()
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._registered_datasets: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Create and return the configured DuckDB connection.

        The connection is created lazily and reused for the lifetime
        of this resource instance.
        """
        if self._connection is not None:
            return self._connection

        with timed_operation(
            logger,
            "duckdb",
            "connection_initialize",
        ):
            self._connection = duckdb.connect(
                database=str(self.config.database_path),
                read_only=self.config.read_only,
            )

            self._configure_connection()

        return self._connection

    def _configure_connection(self) -> None:
        """Apply resource-level DuckDB configuration."""
        connection = self._require_connection()

        if self.config.threads is not None:
            if self.config.threads < 1:
                raise ValueError("threads must be >= 1")

            connection.execute(
                f"SET threads = {self.config.threads}"
            )

        if self.config.memory_limit is not None:
            connection.execute(
                "SET memory_limit = ?",
                [self.config.memory_limit],
            )

        if self.config.temp_directory is not None:
            temp_directory = Path(
                self.config.temp_directory
            ).expanduser()

            temp_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            connection.execute(
                "SET temp_directory = ?",
                [str(temp_directory)],
            )

        logger.info(
            "connection_configured | database=%s | threads=%s | "
            "memory_limit=%s | temp_directory=%s | read_only=%s",
            self.config.database_path,
            self.config.threads,
            self.config.memory_limit,
            self.config.temp_directory,
            self.config.read_only,
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
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
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
        """
        Register a local Parquet dataset as a DuckDB view.

        Parameters
        ----------
        name:
            Logical relation name, e.g. ``cpu``.
        path:
            Parquet file or directory/glob containing Parquet files.
        """
        self._validate_relation_name(name)

        dataset_path = Path(path).expanduser()

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Parquet dataset does not exist: {dataset_path}"
            )

        connection = self.get_connection()

        parquet_expression = self._parquet_expression(
            dataset_path
        )

        with timed_operation(
            logger,
            "duckdb",
            "register_dataset",
            metadata_name=name,
        ):
            connection.execute(
                f"""
                CREATE OR REPLACE VIEW "{name}" AS
                SELECT *
                FROM read_parquet({parquet_expression})
                """
            )

        self._registered_datasets[name] = dataset_path

        logger.info(
            "dataset_registered | name=%s | path=%s",
            name,
            dataset_path,
        )

    def register_all_datasets(
        self,
        datasets: dict[str, str | Path],
    ) -> None:
        """
        Register explicitly supplied datasets.

        This is a convenience method. It does not discover datasets
        automatically and does not open resources that were not supplied.
        """
        for name, path in datasets.items():
            self.register_dataset(name, path)

    # ------------------------------------------------------------------
    # Query / execution
    # ------------------------------------------------------------------

    def query(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None = None,
    ):
        """
        Execute a SQL query and return an Arrow-compatible result.

        The result is intentionally left as a DuckDB relation/result
        object rather than automatically converting the full result
        into pandas.
        """
        connection = self.get_connection()

        with timed_operation(
            logger,
            "duckdb",
            "query",
        ) as timing:
            result = connection.execute(
                sql,
                parameters or [],
            )

        timing.metadata["rows_returned"] = self._safe_row_count(
            result
        )

        return result

    def query_arrow(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None = None,
    ):
        """
        Execute SQL and return the result as an Arrow table.

        This should be used when the caller explicitly needs an
        Arrow representation.
        """
        connection = self.get_connection()

        with timed_operation(
            logger,
            "duckdb",
            "query_arrow",
        ) as timing:
            result = connection.execute(
                sql,
                parameters or [],
            ).fetch_arrow_table()

        timing.metadata["rows_returned"] = result.num_rows

        return result

    def execute(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None = None,
    ) -> duckdb.DuckDBPyConnection:
        """
        Execute SQL without forcing result materialization.
        """
        connection = self.get_connection()

        with timed_operation(
            logger,
            "duckdb",
            "execute",
        ):
            connection.execute(
                sql,
                parameters or [],
            )

        return connection

    # ------------------------------------------------------------------
    # Schema / metadata
    # ------------------------------------------------------------------

    def table_exists(self, name: str) -> bool:
        """Return whether a registered relation exists."""
        connection = self.get_connection()

        result = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [name],
        ).fetchone()

        return bool(result and result[0])

    def describe_table(self, name: str):
        """
        Return the schema of a registered relation.

        Analytical profiling such as null percentages,
        distributions, and cardinality analysis belongs elsewhere.
        """
        if not self.table_exists(name):
            raise ValueError(
                f"DuckDB relation does not exist: {name}"
            )

        connection = self.get_connection()

        with timed_operation(
            logger,
            "duckdb",
            "describe_table",
        ):
            return connection.execute(
                f'DESCRIBE "{name}"'
            ).fetch_arrow_table()

    def registered_datasets(self) -> dict[str, Path]:
        """Return a copy of the currently registered datasets."""
        return dict(self._registered_datasets)

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
    def _validate_relation_name(name: str) -> None:
        """
        Validate logical relation names before interpolating them
        into SQL identifiers.
        """
        if not name:
            raise ValueError(
                "Dataset name cannot be empty."
            )

        if not name.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid DuckDB relation name: {name!r}"
            )

    @staticmethod
    def _parquet_expression(path: Path) -> str:
        """
        Produce a SQL-safe Parquet path expression.

        DuckDB's read_parquet() accepts a string path or glob.
        """
        escaped = str(path).replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def _safe_row_count(result: Any) -> int | None:
        """Best-effort result cardinality without materializing data."""
        try:
            return result.rowcount
        except Exception:
            return None