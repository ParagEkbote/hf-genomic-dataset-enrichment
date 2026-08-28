"""
Faceberg access layer for the Carbon pipeline.

Faceberg maps existing Hugging Face datasets to Apache Iceberg table
metadata without copying data.

This module is the single read path for the catalog-managed lineage
datasets used across Phase 1 (provenance) through Phase 4 (retrieval
evaluation):

    cpu_enriched,
    sampled_cpu,
    tokenized,
    likelihood_stats,
    embeddings

The pretraining corpus / pretraining split remain streaming inputs and
are intentionally not cataloged here.

Lineage semantics
-----------------

    pretraining_split
        -> cpu_enriched
        -> sampled_cpu
        -> tokenized
        -> {likelihood_stats, embeddings}

`sampled_cpu` is the stratified CPU-enriched population used as the GPU
input corpus. Its published HF artifact is named
`carbon-pilot-corpus-dedup`, but the lineage operation represented by this
node is deterministic per-row-hash sampling with stratum
representativeness validation.

The physical biological schema is preserved. Sampling dimensions already
materialized by CPU enrichment are not renamed or recomputed here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, cast,  TypedDict

try:
    from faceberg import LocalCatalog, catalog as _catalog_factory
    import faceberg
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faceberg is required: pip install faceberg"
    ) from exc

_LocalCatalog = cast(Any, faceberg.LocalCatalog)
_catalog_factory = cast(Any, faceberg.catalog)
# ============================================================================
# Pipeline table registry
# ============================================================================

class PipelineTableSpec(TypedDict):
    table: str
    repo: str
    config: str | None

PIPELINE_TABLES: dict[str, PipelineTableSpec] = {
    "cpu_enriched": {
        "table": "carbon.cpu_enriched_sequences",
        "repo": "AINovice2005/carbon-cpu-enriched-sequences",
        "config": None,
    },
    "sampled_cpu": {
        # Stratified CPU-enriched population used as the GPU input corpus.
        # The HF artifact name contains "dedup", but the lineage edge is
        # deterministic per-row-hash sampling with representativeness
        # validation.
        "table": "carbon.pilot_corpus_dedup",
        "repo": "AINovice2005/carbon-pilot-corpus-dedup",
        "config": None,
    },
    "tokenized": {
        "table": "carbon.tokenized_corpus",
        "repo": "AINovice2005/carbon-tokenized-corpus",
        "config": None,
    },
    "likelihood_stats": {
        "table": "carbon.likelihood_stats",
        "repo": "AINovice2005/carbon-likelihood-stats",
        "config": None,
    },
    "embeddings": {
        "table": "carbon.embeddings",
        "repo": "AINovice2005/carbon-embeddings",
        "config": None,
    },
}


# ============================================================================
# Sampling metadata
# ============================================================================

SAMPLING_METHOD = "deterministic_per_row_hash"
SAMPLING_STRATIFICATION_VARIABLES: tuple[str, ...] = (
    "sequence_length_bucket_proxy",
    "is_coding_region",
    "strand",
    "taxonomy_domain",
)

# Exact proxy buckets implemented by sampling.py.
SAMPLING_LENGTH_PROXY_BUCKETS: tuple[int, ...] = (
    512,
    2048,
    8192,
    32768,
    -1,  # >32768; sampling.py uses -1 for the infinity bucket
)

SAMPLING_DRIFT_WARNING_THRESHOLD_PCT = 2.0


# ============================================================================
# Table description
# ============================================================================


@dataclass(frozen=True)
class TableDescription:
    """Live schema/location facts for one catalog table."""

    table: str
    location: str
    schema: dict[str, str]
    partition_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "location": self.location,
            "schema": self.schema,
            "partition_fields": list(self.partition_fields),
        }


# ============================================================================
# Pipeline catalog wrapper
# ============================================================================


class PipelineCatalog:
    """Thin wrapper over the Faceberg catalog.

    Use local() during development and verification, and remote() for the
    published HF Space-backed catalog.
    """

    def __init__(self, cat: Any, *, mode: str, uri: str) -> None:
        self._cat = cat
        self.mode = mode
        self.uri = uri

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def local(cls, path: str = "./carbon-catalog") -> "PipelineCatalog":
        """Create a filesystem-backed catalog for local verification."""

        uri = f"file://{path}"
        cat = LocalCatalog(name="carbon", uri=uri)
        return cls(cat, mode="local", uri=uri)

    @classmethod
    def remote(
        cls,
        catalog_id: str,
        *,
        hf_token: str | None = None,
    ) -> "PipelineCatalog":
        """Connect to the published HF Space-backed catalog."""
        cat = _catalog_factory(
            catalog_id,
            hf_token=hf_token or os.environ.get("HF_TOKEN"),
        )

        return cls(cat, mode="remote",uri=catalog_id,)

    # ------------------------------------------------------------------
    # Catalog lifecycle
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Initialize the catalog and register missing pipeline tables."""

        self._cat.init()
        existing = {str(t) for t in self._cat.list_tables("carbon")}

        for node_id, spec in PIPELINE_TABLES.items():
            if spec["table"] not in existing:
                self._cat.add_dataset(
                    spec["table"],
                    spec["repo"],
                    config=spec["config"],
                )

    def sync(self, node_id: str | None = None) -> None:
        """Sync one table or all catalog-managed tables."""

        if node_id is not None:
            if node_id not in PIPELINE_TABLES:
                raise KeyError(
                    f"{node_id!r} is not a catalog-managed table"
                )
            self._cat.sync_dataset(PIPELINE_TABLES[node_id]["table"])
            return

        self._cat.sync_datasets()

    def table_exists(self, node_id: str) -> bool:
        """Return whether a pipeline table exists."""

        if node_id not in PIPELINE_TABLES:
            raise KeyError(
                f"{node_id!r} is not a catalog-managed table"
            )

        return self._cat.table_exists(
            PIPELINE_TABLES[node_id]["table"]
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def load_table(self, node_id: str) -> Any:
        """Return the PyIceberg table handle for a pipeline node."""

        if node_id not in PIPELINE_TABLES:
            raise KeyError(
                f"{node_id!r} is not a catalog-managed pipeline table. "
                f"Available: {sorted(PIPELINE_TABLES)}. "
                "(pretraining_corpus/pretraining_split are streaming-only.)"
            )

        return self._cat.load_table(
            PIPELINE_TABLES[node_id]["table"]
        )

    def scan(
        self,
        node_id: str,
        *,
        columns: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> Any:
        """Unfiltered scan, optionally column-pruned and row-limited."""

        table = self.load_table(node_id)
        kwargs: dict[str, Any] = {}

        if columns is not None:
            kwargs["selected_fields"] = tuple(columns)

        if limit is not None:
            kwargs["limit"] = limit

        return table.scan(**kwargs).to_pandas()

    def scan_stratum(
        self,
        node_id: str,
        *,
        field: str,
        value: Any,
        columns: Iterable[str] | None = None,
    ) -> Any:
        """Row-filtered scan for one field/value pair."""

        from pyiceberg.expressions import EqualTo

        table = self.load_table(node_id)
        kwargs: dict[str, Any] = {
            "row_filter": EqualTo(field, value),
        }

        if columns is not None:
            kwargs["selected_fields"] = tuple(columns)

        return table.scan(**kwargs).to_pandas()

    def scan_strata(
        self,
        node_id: str,
        *,
        field: str,
        values: Iterable[Any],
        columns: Iterable[str] | None = None,
    ) -> dict[Any, Any]:
        """Scan one DataFrame per requested field value."""

        return {
            value: self.scan_stratum(
                node_id,
                field=field,
                value=value,
                columns=columns,
            )
            for value in values
        }

    # ------------------------------------------------------------------
    # DuckDB SQL surface
    # ------------------------------------------------------------------

    def query(self, sql: str) -> Any:
        """Run SQL against the attached Faceberg catalog.

        SQL should reference catalog tables as:

            cat.carbon.<table>
        """

        import duckdb

        conn = cast(Any, duckdb).connect()

        try:
            conn.execute("INSTALL iceberg; LOAD iceberg")
            conn.execute(self._attach_stmt(alias="cat"))
            return conn.execute(sql).fetchdf()
        finally:
            conn.close()

    def _attach_stmt(self, alias: str = "cat") -> str:
        if self.mode == "local":
            path = self.uri.replace("file://", "")
            return (
                f"ATTACH '{path}' AS {alias} "
                "(TYPE ICEBERG, AUTHORIZATION_TYPE 'none')"
            )

        return f"ATTACH '{self.uri}' AS {alias} (TYPE ICEBERG)"

    # ------------------------------------------------------------------
    # Schema / metadata
    # ------------------------------------------------------------------

    def describe(self, node_id: str) -> TableDescription:
        """Return live schema and partition metadata."""

        table = self.load_table(node_id)
        iceberg_schema = table.schema()

        schema = {
            field.name: str(field.field_type)
            for field in iceberg_schema.fields
        }

        partition_fields = (
            tuple(field.name for field in table.spec().fields)
            if table.spec().fields
            else ()
        )

        return TableDescription(
            table=PIPELINE_TABLES[node_id]["table"],
            location=table.location(),
            schema=schema,
            partition_fields=partition_fields,
        )

    def describe_all(self) -> dict[str, TableDescription]:
        """Return live descriptions for all catalog-managed datasets."""

        return {
            node_id: self.describe(node_id)
            for node_id in PIPELINE_TABLES
        }

    # ------------------------------------------------------------------
    # Sharing
    # ------------------------------------------------------------------

    def attach_sql(self, alias: str = "cat") -> str:
        """Return the exact DuckDB statements needed to attach the catalog."""

        return (
            "INSTALL iceberg; LOAD iceberg;\n"
            f"{self._attach_stmt(alias)};"
        )


# ============================================================================
# Module-level convenience
# ============================================================================


def get_catalog(
    mode: str = "local",
    **kwargs: Any,
) -> PipelineCatalog:
    """Construct a local or remote PipelineCatalog."""

    if mode == "local":
        return PipelineCatalog.local(**kwargs)

    if mode == "remote":
        return PipelineCatalog.remote(**kwargs)

    raise ValueError(
        f"mode must be 'local' or 'remote', got {mode!r}"
    )
