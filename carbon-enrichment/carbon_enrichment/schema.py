"""
Shared data contracts for the Carbon enrichment pipeline.

This module is intentionally independent of Dagster assets and compute
resources. It defines:

1. Hugging Face dataset/configuration identifiers.
2. The three validation tiers used by the development workflow.
3. The expected raw Carbon dataset schema.
4. Allowed categorical/token values.
5. Valid gene-boundary token pairs.
6. The accepted IUPAC nucleotide alphabet.

The schema is shared by CPU ingestion/validation assets, tests, and
later enrichment stages. GPU assets should consume the validated dataset
rather than redefine its raw-data assumptions.
"""

from __future__ import annotations

from typing import Final


# ============================================================================
# 1. Hugging Face dataset
# ============================================================================

HF_DATASET_PATH: Final[str] = (
    "HuggingFaceBio/carbon-pretraining-corpus"
)

HF_DATASET_CONFIG: Final[str] = "eukaryote_generator"

HF_DATASET_SPLIT: Final[str] = "train"


# ============================================================================
# 2. Validation tiers
# ============================================================================
#
# These are intentionally centralized so every stage of the pipeline uses
# the same development/integration/authentication scale definitions.
#
# dev:
#     Fast feedback during implementation.
#
# integration:
#     Main development/integration run.
#
# auth:
#     Expensive pre-release authentication run.
# ============================================================================

VALIDATION_LEVEL_SPLITS: Final[dict[str, str]] = {
    "dev": "train[:100]",
    "integration": "train[:1%]",
    "auth": "train[:25%]",
}

DEFAULT_VALIDATION_LEVEL: Final[str] = "dev"

VALIDATION_LEVELS: Final[tuple[str, ...]] = (
    "dev",
    "integration",
    "auth",
)


# ============================================================================
# 3. Expected raw dataset schema
# ============================================================================
#
# The Carbon eukaryote_generator configuration is expected to expose these
# 14 columns.
#
# These definitions describe the RAW dataset only.
# Derived CPU features and GPU/model-derived features are defined by the
# corresponding enrichment assets rather than being added here.
# ============================================================================

EXPECTED_COLUMNS: Final[dict[str, str]] = {
    "begin_of_sequence": "string",
    "end_of_sequence": "string",
    "begin_of_gene": "string",
    "end_of_gene": "string",
    "gene_type": "string",
    "species_type": "string",
    "strand": "string",
    "sequence": "string",
    "molecule_type": "string",
    "topology": "string",
    "taxonomy": "string",
    "record_id": "string",
    "start": "int64",
    "end": "int64",
}

EXPECTED_COLUMN_NAMES: Final[tuple[str, ...]] = tuple(
    EXPECTED_COLUMNS.keys()
)

EXPECTED_COLUMN_COUNT: Final[int] = len(EXPECTED_COLUMNS)


# ============================================================================
# 4. Required fields
# ============================================================================
#
# These are the fields that downstream processing fundamentally depends on.
# A missing/null value here is more consequential than a general metadata
# anomaly.
# ============================================================================

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "record_id",
    "sequence",
    "taxonomy",
    "strand",
    "start",
    "end",
)


# ============================================================================
# 5. Valid categorical/token vocabularies
# ============================================================================
#
# These describe values observed/expected in categorical fields of the raw
# corpus. They are used by CPU validation checks.
# ============================================================================

VALID_SEQUENCE_TOKENS: Final[dict[str, frozenset[str]]] = {
    "begin_of_sequence": frozenset({"<s>"}),
    "end_of_sequence": frozenset({"</s>"}),
    "begin_of_gene": frozenset({"<bog>", "<bok>"}),
    "end_of_gene": frozenset({"<eog>", "<eok>"}),
    "strand": frozenset({"<+>", "<->"}),
    "molecule_type": frozenset({"DNA", "RNA"}),
    "topology": frozenset({"linear", "circular"}),
}


# ============================================================================
# 6. Gene-boundary token pairing
# ============================================================================
#
# The corpus contains two recognized boundary-token families:
#
#     <bog> -> <eog>
#     <bok> -> <eok>
#
# The CPU validation layer reports unexpected combinations but does not
# necessarily treat them as corruption during development.
# ============================================================================

GENE_BOUNDARY_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("<bog>", "<eog>"),
    ("<bok>", "<eok>"),
)


# ============================================================================
# 7. Nucleotide alphabet
# ============================================================================
#
# Standard bases:
#     A C G T
#
# IUPAC ambiguity codes:
#     N R Y S W K M B D H V
#
# Gap:
#     -
#
# Sequences containing only A/C/G/T are tracked separately as "clean ACGT".
# A sequence containing an accepted ambiguity code is still valid, but is
# not classified as clean ACGT.
# ============================================================================

IUPAC_NUCLEOTIDE_CHARS: Final[frozenset[str]] = frozenset(
    "ACGTNRYSWKMBDHV-"
)

CLEAN_NUCLEOTIDE_CHARS: Final[frozenset[str]] = frozenset(
    "ACGT"
)


# ============================================================================
# 8. Taxonomy expectations
# ============================================================================

KNOWN_TAXONOMY_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "Eukaryota",
        "Bacteria",
        "Archaea",
        "Viruses",
    }
)

MIN_TAXONOMY_LEVELS: Final[int] = 2


# ============================================================================
# 9. Coordinate expectations
# ============================================================================

COORDINATE_COLUMNS: Final[tuple[str, str]] = (
    "start",
    "end",
)

MIN_COORDINATE_VALUE: Final[int] = 0