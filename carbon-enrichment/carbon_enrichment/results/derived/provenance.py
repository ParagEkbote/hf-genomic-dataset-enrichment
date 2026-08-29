from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

# ============================================================================
# Lineage nodes
# ============================================================================


@dataclass(frozen=True)
class ProvenanceNode:
    """A dataset or derived artifact in the Carbon pipeline."""

    id: str
    name: str
    artifact_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Lineage operations
# ============================================================================


@dataclass(frozen=True)
class ProvenanceOperation:
    """A transformation or selection operation between two lineage nodes."""

    id: str
    name: str
    source: str
    target: str
    execution_mode: str
    purpose: str
    parameters: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pipeline provenance
# ============================================================================


@dataclass(frozen=True)
class PipelineProvenance:
    """Canonical lineage definition for the Carbon pipeline."""

    pipeline_name: str
    pipeline_version: str | None
    generated_at: str

    nodes: tuple[ProvenanceNode, ...]
    operations: tuple[ProvenanceOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the provenance object as a JSON-serializable dictionary."""
        return asdict(self)


# ============================================================================
# Canonical Carbon lineage
# ============================================================================


def build_carbon_provenance(
    *,
    pipeline_version: str | None = None,
) -> PipelineProvenance:
    """Build the canonical Carbon pipeline lineage object.

    This function describes the established pipeline lineage only.
    It does not access datasets, Dagster, Hugging Face, or the filesystem.
    """

    nodes = (
        ProvenanceNode(
            id="pretraining_corpus",
            name="Pretraining corpus",
            artifact_type="source_corpus",
            metadata={
                "repository": "HuggingFaceBio/carbon-pretraining-corpus",
                "split": "eukaryote_generator/train",
            },
        ),
        ProvenanceNode(
            id="pretraining_split",
            name="Pretraining split",
            artifact_type="dataset_split",
            metadata={
                "fraction": 0.75,
                "selection": "first_75_percent_by_source_order",
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
            purpose="Select the compute-controlled GPU input population.",
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
            purpose="Convert sampled sequences into model-ready tokenized input.",
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
            purpose="Produce per-sequence model likelihood statistics.",
        ),
        ProvenanceOperation(
            id="embedding_generation",
            name="Embedding generation",
            source="gpu_enriched",
            target="embeddings",
            execution_mode="dagster_asset",
            purpose="Produce model-derived sequence embeddings.",
        ),
    )

    return PipelineProvenance(
        pipeline_name="carbon-enrichment",
        pipeline_version=pipeline_version,
        generated_at=datetime.now(UTC).isoformat(),
        nodes=nodes,
        operations=operations,
    )
