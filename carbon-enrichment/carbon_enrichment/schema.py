"""
Shared data contracts for the Carbon enrichment pipeline.

This module is intentionally independent of Dagster assets and compute
resources. It defines:

1. Hugging Face dataset/configuration identifiers.
2. The validation tiers used by the development workflow.
3. The expected raw Carbon dataset schema.
4. Allowed categorical/token values where the raw contract is established.
5. Valid gene-boundary token pairs.
6. The accepted IUPAC nucleotide alphabet.
7. Structural taxonomy expectations.
8. Coordinate expectations.

Important design principle
--------------------------
The raw dataset contract is lossless. This module must not impose
arbitrary truncation or fixed-width representations on variable-length
biological information such as taxonomy.

Derived CPU features and GPU/model-derived features belong to their
respective processing modules rather than the raw-data contract.
"""

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
# These define the development workflow.
#
# dev:
#     Fast development feedback.
#
# integration:
#     Larger integration-scale execution.
#
# auth:
#     Pre-release / authentication-scale execution.
#
# Streaming does not support split-slicing syntax (e.g. "train[:100]") —
# IterableDataset has no length to slice against, so the builder rejects
# it as an invalid split name. Split selection and stream truncation are
# therefore represented as two separate concerns:
#
#     STREAMING_SPLIT   -> which named HF split to open ("train")
#     VALIDATION_LEVEL_ROWS -> how many examples to .take() from it
#
# This also makes each tier reproducible: it's tied to an explicit row
# count rather than a split-expression that depends on dataset length.
# ============================================================================

STREAMING_SPLIT: Final[str] = "train"

# NOTE: only "dev" and "auth" are backed by a measured/benchmarked row
# count. Do not add "integration" here until its row count has actually
# been measured against the dataset — guessing it via interpolation
# (e.g. assuming exact proportionality to the 25% figure) would silently
# encode an unverified number as a hard contract.
VALIDATION_LEVEL_ROWS: Final[dict[str, int]] = {
    "dev": 100,
    "integration": 9_264_679,
    "auth": 11_580_849,
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
# The Carbon eukaryote_generator configuration exposes these 14 raw columns.
#
# These definitions describe the RAW dataset only.
#
# No derived feature is added here.
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
# These fields are required for downstream CPU enrichment.
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
# These are contracts for categorical fields whose representations are
# established by the Carbon dataset.
#
# gene_type is intentionally NOT included here yet. The dataset exposes
# multiple gene_type classes, but this module should not invent a complete
# six-value vocabulary from a partial sample.
#
# Validation of gene_type therefore focuses on field presence/type and
# non-nullness where appropriate until the complete vocabulary has been
# explicitly established.
# ============================================================================

VALID_SEQUENCE_TOKENS: Final[dict[str, frozenset[str]]] = {
    "begin_of_sequence": frozenset({"<s>"}),
    "end_of_sequence": frozenset({"</s>"}),
    "begin_of_gene": frozenset({
        "<bog>",
        "<bok>",
    }),
    "end_of_gene": frozenset({
        "<eog>",
        "<eok>",
    }),
    "species_type": frozenset({
        "<fng>",
    }),
    "strand": frozenset({
        "<+>",
        "<->",
    }),
    "molecule_type": frozenset({
        "DNA",
    }),
    "topology": frozenset({
        "linear",
        "circular",
    }),
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
# Validation should reject/report unexpected combinations rather than
# assuming that every gene uses the same boundary family.
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
# Accepted ambiguity codes remain valid sequence characters. They are
# tracked separately by enrichment/QC rather than being discarded.
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
#
# Taxonomy is variable-depth biological information.
#
# IMPORTANT:
#     Do NOT define MAX_TAXONOMY_RANKS.
#
# The complete taxonomy string must remain intact. Enrichment may derive:
#
#     taxonomy_domain
#     taxonomy_depth
#
# without truncating the original taxonomy.
#
# Example:
#
#     Eukaryota;Fungi;Dikarya;...;Agaricaceae;Agaricus
#
# remains completely intact in `taxonomy`.
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