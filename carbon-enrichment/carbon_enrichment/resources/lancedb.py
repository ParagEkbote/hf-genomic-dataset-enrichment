from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import lancedb
import pyarrow as pa


@dataclass(frozen=True)
class LanceDBConfig:
    """Configuration for a local LanceDB resource."""

    # LanceDB uses a local directory URI (e.g., "./.lancedb") or S3 URI
    uri: str = "./data/lancedb_store"
    
    table_name: str = "carbon_embeddings"

    # Default distance metric ('cosine', 'l2', or 'dot')
    distance: str = "cosine"

    # ------------------------------------------------------------------
    # Retry configuration (Useful for OS-level file lock contention)
    # ------------------------------------------------------------------
    max_retries: int = 5
    retry_base_delay: float = 2.0


class LanceDBResource:
    """
    Thin resource wrapper around a LanceDB connection.

    Provides table management, storage, and retrieval primitives.
    Retains the retry logic to handle potential OS-level file locking 
    contentions during high-concurrency local writes.
    """

    def __init__(self, config: LanceDBConfig | None = None) -> None:
        self.config = config or LanceDBConfig()
        self._db: lancedb.DBConnection | None = None

    # ------------------------------------------------------------------
    # Client / Connection
    # ------------------------------------------------------------------

    def get_db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._db = lancedb.connect(self.config.uri)
        return self._db

    def close(self) -> None:
        # LanceDB connections are lightweight and don't strictly require 
        # closing like a gRPC client, but we clear the reference for parity.
        self._db = None

    def __enter__(self) -> LanceDBResource:
        self.get_db()
        return self

    def __exit__(
        self,
        _exc_type,
        _exc_value,
        _traceback,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _with_retry(self, operation, operation_name: str):
        max_attempts = max(1, self.config.max_retries)
        base_delay = max(0.0, self.config.retry_base_delay)

        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                if attempt >= max_attempts:
                    print(
                        f"\nERROR: {operation_name} failed after "
                        f"{max_attempts} attempts."
                    )
                    print(f"Last error: {exc}")
                    raise

                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"\nWARNING: {operation_name} failed "
                    f"(attempt {attempt}/{max_attempts})."
                )
                print(f"Error: {exc}")
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def table_exists(self, table_name: str | None = None) -> bool:
        """Return whether a table exists."""
        name = table_name or self.config.table_name
        return name in self.get_db().table_names()

    def ensure_table(
        self,
        schema: pa.Schema,
        *,
        table_name: str | None = None,
    ) -> lancedb.table.Table:
        """
        Create the table if it does not already exist.
        LanceDB requires a PyArrow schema or a Pydantic model.
        """
        name = table_name or self.config.table_name
        
        return self._with_retry(
            lambda: self.get_db().create_table(
                name=name, 
                schema=schema, 
                exist_ok=True
            ),
            f"ensure_table('{name}')",
        )

    def recreate_table(
        self,
        schema: pa.Schema,
        *,
        table_name: str | None = None,
    ) -> lancedb.table.Table:
        """Delete and recreate a table for reproducible runs."""
        name = table_name or self.config.table_name

        return self._with_retry(
            lambda: self.get_db().create_table(
                name=name, 
                schema=schema, 
                mode="overwrite"
            ),
            f"recreate_table('{name}')",
        )

    def count(self, table_name: str | None = None) -> int:
        """Return the number of rows in a table."""
        name = table_name or self.config.table_name
        table = self.get_db().open_table(name)
        
        return self._with_retry(
            lambda: len(table),
            f"count('{name}')",
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def upsert_points(
        self,
        data: List[Dict[str, Any]] | pa.Table,
        *,
        table_name: str | None = None,
    ) -> None:
        """
        Insert or update points.
        Unlike Qdrant's PointStruct, LanceDB natively accepts a list of 
        dictionaries, a Pandas DataFrame, or a PyArrow Table.
        """
        name = table_name or self.config.table_name
        table = self.get_db().open_table(name)

        # LanceDB supports merge/upsert via merge_insert, but basic 
        # append is 'add'. We use add here assuming unique data ingestion.
        self._with_retry(
            lambda: table.add(data),
            f"upsert({len(data)} points)",
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        table_name: str | None = None,
        query_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Search for nearest vectors."""
        name = table_name or self.config.table_name
        table = self.get_db().open_table(name)

        def _do_search():
            query = table.search(vector).metric(self.config.distance).limit(limit)
            if query_filter:
                # LanceDB uses standard SQL where-clauses for filtering
                query = query.where(query_filter)
            return query.to_list()

        return self._with_retry(_do_search, f"search('{name}')")

    def search_batch(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        limit: int = 10,
        table_name: str | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Since LanceDB is local, batching queries avoids network latency inherently.
        However, you can execute loops locally with minimal overhead, or 
        process directly on a pyarrow batch.
        """
        name = table_name or self.config.table_name
        table = self.get_db().open_table(name)

        def _do_batch():
            results = []
            for vec in vectors:
                results.append(
                    table.search(vec)
                    .metric(self.config.distance)
                    .limit(limit)
                    .to_list()
                )
            return results

        return self._with_retry(_do_batch, f"search_batch({len(vectors)} queries)")

    # ------------------------------------------------------------------
    # Faceting & Aggregation
    # ------------------------------------------------------------------

    def facet(
        self,
        key: str,
        *,
        table_name: str | None = None,
    ) -> pa.Table:
        """
        LanceDB uses DuckDB integration for complex faceting and aggregation
        without moving data out of the Arrow format.
        """
        import duckdb
        name = table_name or self.config.table_name
        
        # Expose the Lance dataset to DuckDB
        ds = self.get_db().open_table(name).to_lance() 
        
        def _do_facet():
            return duckdb.sql(
                f"SELECT {key}, COUNT(*) as count FROM ds GROUP BY {key} ORDER BY count DESC"
            ).arrow()

        return self._with_retry(_do_facet, f"facet('{name}', key='{key}')")