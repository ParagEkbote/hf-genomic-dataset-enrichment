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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import faceberg
import faceberg.iceberg as _iceberg_mod
import pyiceberg.avro.encoder as _enc
import pyiceberg.conversions as _conv
from faceberg import catalog as _catalog_factory
from faceberg.iceberg import (
    Lock,
    ManifestEntry,
    ThreadPoolExecutor,
    _write_manifest,
    create_data_file,
)
from pyiceberg.types import StringType



LINEAGE_MANIFEST_FILENAME = "lineage.yml"


def _lineage_manifest_path(catalog_path: Path) -> Path:
    return catalog_path / LINEAGE_MANIFEST_FILENAME


def write_lineage_manifest(catalog_path: str | Path) -> Path:
    """Write the full PIPELINE_TABLES registry (streaming + catalog nodes)
    as a sibling YAML file next to faceberg.yml.

    Unlike faceberg.yml, this is plain data we own outright — it is never
    read or rewritten by _LocalCatalog, so it's safe to include nodes with
    no Iceberg table (access_mode == "streaming").
    """

    import yaml

    resolved_path = Path(catalog_path).expanduser().resolve()
    resolved_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        node_id: {
            "table": spec["table"],
            "repo": spec["repo"],
            "config": spec["config"],
            "upstream": spec["upstream"],
            "access_mode": spec["access_mode"],
            "description": spec["description"],
        }
        for node_id, spec in PIPELINE_TABLES.items()
    }

    manifest_path = _lineage_manifest_path(resolved_path)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    return manifest_path

# ============================================================================
# Monkey-patching
# ============================================================================
DEFAULT_CATALOG_PATH = Path("carbon-catalog")
_LocalCatalog = cast(Any, faceberg.LocalCatalog)
_catalog_factory = cast(Any, faceberg.catalog)
_orig_write_utf8 = _enc.BinaryEncoder.write_utf8
_orig_to_bytes_str = _conv.to_bytes.registry[StringType]


def _patched_write_utf8(self, s):
    return _orig_write_utf8(self, str(s))


_enc.BinaryEncoder.write_utf8 = _patched_write_utf8


def _patched_to_bytes_str(primitive_type, value):
    return _orig_to_bytes_str(primitive_type, str(value))


_conv.to_bytes.register(StringType, _patched_to_bytes_str)


def _patched_write_manifest(
    files,
    metadata,
    schema,
    spec,
    snapshot_id,
    sequence_number,
    io,
    output_file,
    manifest_uri,
    include_split_column,
    progress_callback,
    max_workers=None,
):
    progress_callback(state="in_progress", percent=20, stage="Creating metadata files")
    total_files = len(files)
    completed_files = 0
    lock = Lock()

    def convert_file(parquet_file):
        nonlocal completed_files
        result = create_data_file(
            io=io,
            table_metadata=metadata,
            parquet_file=parquet_file,
            include_split_column=include_split_column,
        )
        with lock:
            completed_files += 1
            percent = 20 + int((completed_files / total_files) * 70)
            progress_callback(
                state="in_progress",
                percent=percent,
                stage=f"{parquet_file.path} ({completed_files}/{total_files})",
            )
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        parquet_files = [pf for _, pf in files]
        data_files = list(executor.map(convert_file, parquet_files))

    data_file_map = {pf.uri: df for pf, df in zip(parquet_files, data_files)}
    entries = []
    with _write_manifest(
        format_version=2,
        spec=spec,
        schema=schema,
        output_file=output_file,
        snapshot_id=snapshot_id,
        avro_compression="deflate",
    ) as writer:
        for status, parquet_file in files:
            data_file = data_file_map[parquet_file.uri]
            entry = ManifestEntry.from_args(
                status=status,
                snapshot_id=snapshot_id,
                sequence_number=sequence_number,
                file_sequence_number=sequence_number,
                data_file=data_file,
            )
            writer.add_entry(entry)
            entries.append(entry)
    # moved outside the `with` block so the writer has flushed/closed first
    manifest = writer.to_manifest_file()
    manifest[0] = manifest_uri
    return manifest, entries


_iceberg_mod.write_manifest = _patched_write_manifest

# ============================================================================
# Pipeline table registry
# ============================================================================

# ============================================================================
# Pipeline table registry
# ============================================================================

class PipelineTableSpec(TypedDict):
    table: str | None
    """Iceberg-qualified table name (e.g. "carbon.cpu_enriched_sequences").
    None for streaming-only nodes that are never materialized as a
    catalog table."""

    repo: str
    """HF Hub repo id this node's data lives in."""

    config: str | None
    """HF dataset config name, if the repo has more than one."""

    upstream: str | None
    """node_id of this node's immediate lineage parent within this
    registry, or None for root sources with no upstream tracked here."""

    access_mode: str
    """"catalog" if Faceberg/Iceberg-backed and DuckDB-queryable via
    cat.<table>, or "streaming" if only accessible as an IterableDataset."""

    description: str
    """One-line human-readable purpose. Replaces ad hoc inline comments."""


PIPELINE_TABLES: dict[str, PipelineTableSpec] = {
    "pretraining_corpus": {
        "table": None,
        "repo": "HuggingFaceBio/carbon-pretraining-corpus",
        "config": None,
        "upstream": None,
        "access_mode": "streaming",
        "description": (
            "Root source corpus (eukaryote_generator/train split). "
            "Streaming-only — never cataloged."
        ),
    },
    "cpu_enriched": {
        "table": "carbon.cpu_enriched_sequences",
        "repo": "AINovice2005/carbon-cpu-enriched-sequences",
        "config": None,
        "upstream": "pretraining_corpus",
        "access_mode": "catalog",
        "description": "CPU-derived sequence features from the 75% pretraining split.",
    },
    "sampled_cpu": {
        "table": "carbon.pilot_corpus_dedup",
        "repo": "AINovice2005/carbon-pilot-corpus-dedup",
        "config": None,
        "upstream": "cpu_enriched",
        "access_mode": "catalog",
        "description": (
            "Stratified CPU-enriched population used as the GPU input "
            "corpus. HF artifact name contains 'dedup', but the lineage "
            "edge is deterministic per-row-hash sampling with "
            "representativeness validation, not deduplication."
        ),
    },
    "tokenized": {
        "table": "carbon.tokenized_corpus",
        "repo": "AINovice2005/carbon-tokenized-corpus",
        "config": None,
        "upstream": "sampled_cpu",
        "access_mode": "catalog",
        "description": "Model-ready tokenized input from the sampled CPU population.",
    },
    "likelihood_stats": {
        "table": "carbon.likelihood_stats",
        "repo": "AINovice2005/carbon-likelihood-stats",
        "config": None,
        "upstream": "tokenized",
        "access_mode": "catalog",
        "description": "Per-sequence model likelihood statistics from GPU enrichment.",
    },
    "embeddings": {
        "table": "carbon.embeddings",
        "repo": "AINovice2005/carbon-embeddings",
        "config": None,
        "upstream": "tokenized",
        "access_mode": "catalog",
        "description": "Model-derived sequence embeddings from GPU enrichment.",
    },
}


def catalog_managed_tables() -> dict[str, PipelineTableSpec]:
    """PIPELINE_TABLES filtered to catalog-backed (Iceberg) nodes only."""
    return {
        node_id: spec
        for node_id, spec in PIPELINE_TABLES.items()
        if spec["access_mode"] == "catalog"
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

    def __init__(
        self,
        cat: Any,
        *,
        mode: str,
        uri: str,
    ) -> None:
        self._cat = cat
        self.mode = mode
        self.uri = uri

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def local(
        cls,
        path: str | Path = DEFAULT_CATALOG_PATH,
    ) -> PipelineCatalog:
        """Create a filesystem-backed Faceberg catalog."""

        catalog_path = Path(path).expanduser().resolve()
        catalog_path.mkdir(parents=True, exist_ok=True)

        cat = _LocalCatalog(
            name="carbon",
            uri=f"file://{catalog_path}",
        )

        return cls(
            cat=cat,
            mode="local",
            uri=f"file://{catalog_path}",
        )

    @classmethod
    def remote(
        cls,
        catalog_id: str,
        *,
        hf_token: str | None = None,
    ) -> PipelineCatalog:
        """Connect to the published HF Space-backed catalog."""

        cat = _catalog_factory(
            catalog_id,
            hf_token=hf_token or os.environ.get("HF_TOKEN"),
        )

        return cls(
            cat,
            mode="remote",
            uri=catalog_id,
        )

    # ------------------------------------------------------------------
    # Catalog lifecycle
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Initialize the catalog, register missing catalog-backed tables,
        and (re)write the full lineage manifest including streaming-only
        nodes."""

        self._cat.init()
        self._cat.create_namespace_if_not_exists("carbon")

        existing = {
            str(table)
            for table in self._cat.list_tables("carbon")
        }

        for spec in catalog_managed_tables().values():
            if spec["table"] not in existing:
                self._cat.add_dataset(
                    spec["table"],
                    spec["repo"],
                    config=spec["config"],
                )

        if self.mode == "local":
            # self.uri is "file://{catalog_path}" for local catalogs.
            catalog_path = self.uri.removeprefix("file://")
            write_lineage_manifest(catalog_path)

    def sync(self, node_id: str | None = None) -> None:
        """Sync one table or all catalog-managed tables."""

        if node_id is not None:
            if node_id not in PIPELINE_TABLES:
                raise KeyError(
                    f"{node_id!r} is not a catalog-managed table"
                )

            self._cat.sync_dataset(
                PIPELINE_TABLES[node_id]["table"]
            )
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

        spec = PIPELINE_TABLES.get(node_id)

        if spec is None or spec["access_mode"] != "catalog":
            raise KeyError(
                f"{node_id!r} is not a catalog-managed pipeline table. "
                f"Available: {sorted(catalog_managed_tables())}."
            )

        return self._cat.load_table(spec["table"])

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
        """Run SQL against catalog-managed Iceberg tables.

        SQL should reference pipeline tables as:

            cat.carbon.<table>

        Faceberg resolves the Iceberg metadata location, while DuckDB
        performs the query against the underlying Parquet data.
        """

        from carbon_enrichment.resources.duckdb import (
            get_connection,
            register_catalog_table,
        )

        conn = get_connection()

        try:
            for node_id in PIPELINE_TABLES:
                table = self.load_table(node_id)
                metadata_location = str(table.metadata_location)

                register_catalog_table(
                    conn,
                    node_id=node_id,
                    metadata_location=metadata_location,
                )

            return conn.execute(sql).fetchdf()

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema / metadata
    # ------------------------------------------------------------------

    def describe(self, node_id: str) -> TableDescription:
        """Return live schema and partition metadata."""

        if node_id not in PIPELINE_TABLES:
            raise KeyError(
                f"{node_id!r} is not a catalog-managed table"
            )

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
        """Return the DuckDB setup statements for catalog-managed tables.

        This method is retained for compatibility. It no longer uses
        DuckDB TYPE ICEBERG ATTACH for local catalogs.
        """

        if alias != "cat":
            raise ValueError(
                "attach_sql() only supports the 'cat' alias"
            )

        return (
            "INSTALL iceberg; LOAD iceberg;"
        )
    
    def list_data_files(
        self,
        node_id: str,
    ) -> list[str]:
        """Return the underlying Parquet file URIs for a catalog table."""

        if node_id not in PIPELINE_TABLES:
            raise KeyError(
                f"{node_id!r} is not a catalog-managed pipeline table"
            )

        from faceberg.catalog import discover_dataset

        spec = PIPELINE_TABLES[node_id]

        info = discover_dataset(
            repo_id=spec["repo"],
            config=spec["config"],
        )

        return [str(parquet_file.uri) for parquet_file in info.files]


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