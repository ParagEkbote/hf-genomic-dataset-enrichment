from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


# ============================================================================
# Carbon resources
# ============================================================================

from carbon_enrichment.resources.faceberg import (
    PipelineCatalog,
    catalog_managed_tables,
    get_catalog,
)


# ============================================================================
# Filesystem helpers
# ============================================================================


def _use_same_filesystem_tmpdir(catalog_path: Path) -> None:
    """Point tempfile-based staging at the same filesystem as the catalog.

    faceberg's _commit() stages writes under tempfile.mkdtemp() (default:
    /tmp) then does an atomic os.replace() into the catalog directory.
    On environments where /tmp is a separate mount from the working
    directory (e.g. cloud studio setups), that replace() fails with
    EXDEV/"Invalid cross-device link". Forcing TMPDIR onto the same
    filesystem as the catalog avoids it.
    """

    local_tmp = catalog_path / ".tmp"
    local_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(local_tmp)
    tempfile.tempdir = None  # force tempfile to re-read TMPDIR


def _require_initialized_catalog(catalog: PipelineCatalog) -> None:
    """Fail loudly if the catalog exists but has no registered pipeline tables.

    PipelineCatalog.local() only constructs the catalog handle — it does not
    call ensure_initialized(). This script is read-only by default and will
    not initialize the catalog for you unless --init is passed, so we must
    detect an uninitialized/partially-initialized catalog up front rather
    than let load_table() fail deep inside provenance generation.
    """

    missing = [
        node_id
        for node_id in catalog_managed_tables()
        if not catalog.table_exists(node_id)
    ]

    if missing:
        raise RuntimeError(
            f"Catalog at {catalog.uri!r} is not initialized — missing "
            f"registered tables for: {missing}. Re-run with --init to "
            "initialize the catalog (create namespace + register "
            "PIPELINE_TABLES) before generating provenance."
        )


# ============================================================================
# Canonical lineage nodes
# ============================================================================


@dataclass(frozen=True)
class ProvenanceNode:
    """A logical dataset or derived artifact in the Carbon pipeline."""

    id: str
    name: str
    artifact_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Canonical lineage operations
# ============================================================================


@dataclass(frozen=True)
class ProvenanceOperation:
    """A transformation or selection between two lineage nodes."""

    id: str
    name: str
    source: str
    target: str
    execution_mode: str
    purpose: str
    parameters: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Analytical provenance
# ============================================================================


@dataclass(frozen=True)
class ProvenanceAnalysis:
    """An analytical operation performed against existing pipeline data."""

    id: str
    name: str
    sources: tuple[str, ...]
    execution_mode: str
    purpose: str
    parameters: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Runtime node resolution
# ============================================================================


@dataclass(frozen=True)
class ResolvedProvenanceNode:
    """Concrete runtime identity of a Faceberg/Iceberg pipeline node."""

    node_id: str
    table: str
    repository: str
    config: str | None

    metadata_location: str | None
    snapshot_id: str | None

    location: str | None
    schema: dict[str, str]
    partition_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# Execution provenance
# ============================================================================


@dataclass(frozen=True)
class DuckDBExecution:
    """A concrete query executed during provenance generation.
    (Kept for schema compatibility, though no longer uses DuckDB)."""

    query_id: str
    purpose: str
    sql: str
    tables: tuple[str, ...]

    row_count: int | None
    elapsed_seconds: float

    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# Runtime provenance
# ============================================================================


@dataclass(frozen=True)
class ProvenanceExecution:
    """Runtime environment in which provenance was generated."""

    engine: str
    catalog_type: str
    catalog_uri: str
    generated_at: str
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# Complete provenance document
# ============================================================================


@dataclass(frozen=True)
class PipelineProvenance:
    """Canonical Carbon lineage plus runtime-resolved provenance."""

    pipeline_name: str
    pipeline_version: str | None
    generated_at: str

    nodes: tuple[ProvenanceNode, ...]
    operations: tuple[ProvenanceOperation, ...]
    analyses: tuple[ProvenanceAnalysis, ...]

    resolved_nodes: tuple[ResolvedProvenanceNode, ...]
    executions: tuple[DuckDBExecution, ...]

    execution: ProvenanceExecution

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# Canonical Carbon lineage
# ============================================================================


def build_carbon_provenance(
    *,
    pipeline_version: str | None = None,
) -> tuple[
    tuple[ProvenanceNode, ...],
    tuple[ProvenanceOperation, ...],
    tuple[ProvenanceAnalysis, ...],
]:
    """Build the canonical Carbon lineage definition.

    This function describes logical lineage only. It does not access
    Faceberg, DuckDB, Hugging Face, or the filesystem.
    """

    nodes = (
        ProvenanceNode(
            id="pretraining_corpus",
            name="Pretraining corpus",
            artifact_type="source_corpus",
            metadata={
                "repository": (
                    "HuggingFaceBio/carbon-pretraining-corpus"
                ),
                "split": "eukaryote_generator/train",
                "access_mode": "streaming",
            },
        ),
        ProvenanceNode(
            id="pretraining_split",
            name="Pretraining split",
            artifact_type="dataset_split",
            metadata={
                "fraction": 0.75,
                "selection": "first_75_percent_by_source_order",
                "access_mode": "streaming",
            },
        ),
        ProvenanceNode(
            id="cpu_enriched",
            name="CPU-enriched sequences",
            artifact_type="cpu_enriched",
        ),
        ProvenanceNode(
            id="sampled_cpu",
            name="Sampled CPU-enriched population",
            artifact_type="sampled_population",
        ),
        ProvenanceNode(
            id="tokenized",
            name="Tokenized corpus",
            artifact_type="tokenized_dataset",
        ),
        ProvenanceNode(
            id="gpu_enriched",
            name="GPU-enriched sequences",
            artifact_type="gpu_enriched",
        ),
        ProvenanceNode(
            id="likelihood_stats",
            name="Likelihood statistics",
            artifact_type="derived_statistics",
        ),
        ProvenanceNode(
            id="embeddings",
            name="Embeddings",
            artifact_type="embedding_dataset",
        ),
    )

    operations = (
        ProvenanceOperation(
            id="pretraining_split",
            name="75% pretraining split",
            source="pretraining_corpus",
            target="pretraining_split",
            execution_mode="dataset_operation",
            purpose=(
                "Define the pretraining population as the first 75% "
                "of records in the source split."
            ),
            parameters={
                "fraction": 0.75,
                "selection": "first_75_percent_by_source_order",
            },
        ),
        ProvenanceOperation(
            id="cpu_enrichment",
            name="CPU enrichment",
            source="pretraining_split",
            target="cpu_enriched",
            execution_mode="dagster_asset",
            purpose="Generate CPU-derived sequence features.",
        ),
        ProvenanceOperation(
            id="cpu_sampling",
            name="Stratified corpus sampling",
            source="cpu_enriched",
            target="sampled_cpu",
            execution_mode="python_script",
            purpose=(
                "Select the compute-controlled GPU input population."
            ),
            parameters={
                "method": "deterministic_per_row_hash",
                "allocation": "proportional",
                "strata": (
                    "sequence_length_bucket_proxy",
                    "is_coding_region",
                    "strand",
                    "taxonomy_domain",
                ),
            },
        ),
        ProvenanceOperation(
            id="tokenization",
            name="Tokenization",
            source="sampled_cpu",
            target="tokenized",
            execution_mode="dagster_asset",
            purpose=(
                "Convert sampled sequences into model-ready "
                "tokenized input."
            ),
        ),
        ProvenanceOperation(
            id="gpu_processing",
            name="Token-budgeted GPU processing",
            source="tokenized",
            target="gpu_enriched",
            execution_mode="dagster_asset",
            purpose=(
                "Generate model-derived sequence information "
                "within the configured token budget."
            ),
        ),
        ProvenanceOperation(
            id="likelihood_generation",
            name="Likelihood statistics",
            source="gpu_enriched",
            target="likelihood_stats",
            execution_mode="dagster_asset",
            purpose=(
                "Produce per-sequence model likelihood statistics."
            ),
        ),
        ProvenanceOperation(
            id="embedding_generation",
            name="Embedding generation",
            source="gpu_enriched",
            target="embeddings",
            execution_mode="dagster_asset",
            purpose=(
                "Produce model-derived sequence embeddings."
            ),
        ),
    )

    analyses = (
        ProvenanceAnalysis(
            id="phase1_catalog_validation",
            name="Phase 1 catalog validation",
            sources=(
                "cpu_enriched",
                "sampled_cpu",
                "tokenized",
                "likelihood_stats",
                "embeddings",
            ),
            execution_mode="duckdb",
            purpose=(
                "Resolve and validate the current Faceberg/Iceberg "
                "pipeline artifacts through DuckDB."
            ),
        ),
        ProvenanceAnalysis(
            id="phase1_cpu_population_analysis",
            name="Phase 1 CPU population analysis",
            sources=("cpu_enriched",),
            execution_mode="duckdb",
            purpose=(
                "Characterize the CPU-enriched population across "
                "the dimensions used for downstream sampling."
            ),
            parameters={
                "dimensions": (
                    "sequence_length_bucket_proxy",
                    "is_coding_region",
                    "strand",
                    "taxonomy_domain",
                ),
            },
        ),
        ProvenanceAnalysis(
            id="phase1_sampling_validation",
            name="Phase 1 sampling validation",
            sources=("cpu_enriched", "sampled_cpu"),
            execution_mode="duckdb",
            purpose=(
                "Compare the sampled CPU population against its "
                "source population to validate representativeness."
            ),
            parameters={
                "comparison": (
                    "independent_and_joint_stratum_distributions"
                ),
                "drift_warning_threshold_pct": 2.0,
            },
        ),
    )

    return nodes, operations, analyses


# ============================================================================
# Faceberg runtime resolution
# ============================================================================


def _get_snapshot_id(table: Any) -> str | None:
    """Return the current Iceberg snapshot ID when available."""

    try:
        snapshot = table.current_snapshot()
    except (AttributeError, TypeError):
        return None

    if snapshot is None:
        return None

    snapshot_id = getattr(snapshot, "snapshot_id", None)

    if snapshot_id is None:
        return None

    return str(snapshot_id)


def resolve_faceberg_nodes(
    catalog: PipelineCatalog,
) -> tuple[ResolvedProvenanceNode, ...]:
    """Resolve all catalog-managed nodes through Faceberg."""

    resolved: list[ResolvedProvenanceNode] = []

    for node_id, spec in catalog_managed_tables().items():
        table_name = spec["table"]

        if table_name is None:
            # catalog_managed_tables() should never yield a streaming-only
            # node (access_mode != "catalog" is filtered out), so this is
            # a defensive guard against a future registry/filter bug
            # rather than an expected runtime path.
            raise AssertionError(
                f"{node_id!r} has access_mode='catalog' but no table name"
            )

        print(f"  resolving {node_id} ({table_name})...", flush=True)

        node_started = perf_counter()

        print(f"    load_table({node_id})...", flush=True)
        table = catalog.load_table(node_id)
        print(f"    load_table done ({perf_counter() - node_started:.1f}s)", flush=True)

        describe_started = perf_counter()
        print(f"    describe({node_id})...", flush=True)
        description = catalog.describe(node_id)
        print(f"    describe done ({perf_counter() - describe_started:.1f}s)", flush=True)

        metadata_location = getattr(
            table,
            "metadata_location",
            None,
        )

        resolved.append(
            ResolvedProvenanceNode(
                node_id=node_id,
                table=table_name,
                repository=spec["repo"],
                config=spec["config"],
                metadata_location=(
                    str(metadata_location)
                    if metadata_location is not None
                    else None
                ),
                snapshot_id=_get_snapshot_id(table),
                location=description.location,
                schema=description.schema,
                partition_fields=description.partition_fields,
            )
        )

        print(
            f"  {node_id} resolved ({perf_counter() - node_started:.1f}s total)",
            flush=True,
        )

    return tuple(resolved)


# ============================================================================
# Phase 1 Faceberg validation
# ============================================================================


def run_phase1_validation(
    catalog: PipelineCatalog,
) -> tuple[DuckDBExecution, ...]:
    """Validate catalog state using Faceberg metadata only.

    Provenance generation must not execute analytical DuckDB queries over
    large Iceberg tables. Faceberg resolution already provides the live
    metadata location, snapshot, schema, and partition information.
    """

    executions: list[DuckDBExecution] = []

    for node_id in catalog_managed_tables():
        started = perf_counter()

        table = catalog.load_table(node_id)
        metadata_location = getattr(table, "metadata_location", None)

        executions.append(
            DuckDBExecution(
                query_id=f"catalog_resolution_{node_id}",
                purpose=(
                    "Resolve the live Faceberg/Iceberg catalog entry "
                    "without executing a DuckDB data query."
                ),
                sql="-- metadata-only validation; no DuckDB query executed",
                tables=(node_id,),
                row_count=None,
                elapsed_seconds=perf_counter() - started,
                result={
                    "accessible": True,
                    "validation_mode": "faceberg_metadata",
                    "metadata_location": (
                        str(metadata_location)
                        if metadata_location is not None
                        else None
                    ),
                },
            )
        )

    return tuple(executions)


# ============================================================================
# Complete runtime provenance generation
# ============================================================================


def generate_provenance(
    *,
    catalog: PipelineCatalog,
    pipeline_version: str | None,
    output_path: Path,
) -> PipelineProvenance:
    """Generate complete runtime-resolved provenance."""

    generated_at = datetime.now(UTC).isoformat()

    (
        nodes,
        operations,
        analyses,
    ) = build_carbon_provenance(
        pipeline_version=pipeline_version,
    )

    resolved_nodes = resolve_faceberg_nodes(catalog)

    executions = run_phase1_validation(catalog)

    execution = ProvenanceExecution(
        engine="faceberg_metadata",
        catalog_type=type(catalog).__name__,
        catalog_uri=catalog.uri,
        generated_at=generated_at,
        output_path=str(output_path),
    )

    return PipelineProvenance(
        pipeline_name="carbon-enrichment",
        pipeline_version=pipeline_version,
        generated_at=generated_at,
        nodes=nodes,
        operations=operations,
        analyses=analyses,
        resolved_nodes=resolved_nodes,
        executions=executions,
        execution=execution,
    )


# ============================================================================
# Serialization
# ============================================================================


def write_provenance(
    provenance: PipelineProvenance,
    output_path: Path,
) -> Path:
    """Write the complete provenance document as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            provenance.to_dict(),
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate runtime-resolved Carbon provenance using "
            "the existing Faceberg catalog."
        )
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("carbon-catalog"),
        help=(
            "Local Faceberg catalog path. "
            "Default: carbon-catalog"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output provenance JSON path. "
            "Defaults to carbon-catalog/provenance.json."
        ),
    )

    parser.add_argument(
        "--version",
        default=None,
        help="Optional Carbon pipeline version.",
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help=(
            "Initialize the catalog (create namespace + register "
            "PIPELINE_TABLES) before generating provenance. Off by "
            "default — this script is otherwise strictly read-only "
            "against the existing catalog state."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """Generate runtime provenance from the current Carbon catalog."""

    args = parse_args()

    catalog_path = args.catalog.expanduser().resolve()

    if args.init:
        catalog_path.mkdir(parents=True, exist_ok=True)
        _use_same_filesystem_tmpdir(catalog_path)
    elif not catalog_path.exists():
        raise FileNotFoundError(
            f"Faceberg catalog does not exist: {catalog_path}. "
            "Pass --init to create and initialize it."
        )

    faceberg_config = catalog_path / "faceberg.yml"

    if not args.init and not faceberg_config.exists():
        raise FileNotFoundError(
            f"Faceberg configuration does not exist: "
            f"{faceberg_config}. Pass --init to create it."
        )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else catalog_path / "provenance.json"
    )

    print("=" * 72)
    print("Carbon runtime provenance")
    print("=" * 72)
    print(f"Catalog : {catalog_path}")
    print(f"Output  : {output_path}")
    print()

    # ------------------------------------------------------------------
    # Faceberg
    #
    # IMPORTANT:
    # By default this is a read-only provenance operation against the
    # existing catalog state. Initialization only happens if --init is
    # explicitly passed.
    # ------------------------------------------------------------------

    print("Opening existing Faceberg catalog...")

    catalog = get_catalog(
        mode="local",
        path=catalog_path,
    )

    print("Faceberg catalog opened.")
    print()

    if args.init:
        print("Initializing catalog (--init)...")
        catalog.ensure_initialized()
        print("Catalog initialized.")
        print(f"Lineage manifest : {catalog_path / 'lineage.yml'}")
        print()

    _require_initialized_catalog(catalog)

    # ------------------------------------------------------------------
    # Resolve + Validation
    # ------------------------------------------------------------------

    print("Resolving Faceberg/Iceberg state...")

    try:
        provenance = generate_provenance(
            catalog=catalog,
            pipeline_version=args.version,
            output_path=output_path,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Provenance generation failed against catalog "
            f"{catalog.uri!r}: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Write artifact
    # ------------------------------------------------------------------

    print("Writing provenance artifact...")

    write_provenance(
        provenance,
        output_path,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("Resolved nodes:")

    for node in provenance.resolved_nodes:
        print(
            f"  {node.node_id:18s}"
            f" snapshot={node.snapshot_id or 'unknown'}"
        )

    print()
    print("DuckDB executions (metadata validation):")

    for execution in provenance.executions:
        print(
            f"  {execution.query_id:40s}"
            f" {execution.elapsed_seconds:.3f}s"
        )

    print()
    print(f"Provenance written to: {output_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())