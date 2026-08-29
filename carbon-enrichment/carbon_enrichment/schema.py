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
9. GPU enrichment output contracts.
10. GPU token-mask semantics.

Important design principle
--------------------------
The raw dataset contract is lossless. This module must not impose
arbitrary truncation or fixed-width representations on variable-length
biological information such as taxonomy.

Derived CPU features and GPU/model-derived features belong to their
respective processing modules rather than the raw-data contract.

GPU output schemas are declared here only as data contracts: they define
the columns and shared invariants of materialized GPU assets without
embedding model-specific computation or feature semantics.
"""

from typing import Final

# ============================================================================
# 1. Hugging Face dataset
# ============================================================================

HF_DATASET_PATH: Final[str] = "HuggingFaceBio/carbon-pretraining-corpus"

HF_DATASET_CONFIG: Final[str] = "eukaryote_generator"

HF_DATASET_SPLIT: Final[str] = "train"


# ============================================================================
# 2. Validation / dataset configuration
# ============================================================================

STREAMING_SPLIT: Final[str] = "train"

VALIDATION_LEVEL_ROWS: Final[dict[str, int]] = {
    "dev": 1_000,
    "integration": 3_241_000,
    "auth": 32_410_000,
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
# The Carbon eukaryote_generator configuration exposes these raw columns.
#
# These definitions describe the RAW dataset only.
#
# No derived feature is added here.
#
# The deduplicated corpus retains this same biological/source schema.
# Deduplication is enforced using BIOLOGICAL_KEY_COLUMNS below.
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

EXPECTED_COLUMN_NAMES: Final[tuple[str, ...]] = tuple(EXPECTED_COLUMNS.keys())

EXPECTED_COLUMN_COUNT: Final[int] = len(EXPECTED_COLUMNS)


# ============================================================================
# 4. Required fields
# ============================================================================
#
# These fields are required for downstream CPU enrichment, tokenization,
# and GPU enrichment.
#
# record_id/start/end together form the canonical biological identity.
# sequence is the model/tokenization input.
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
# 5. Row / biological identity
# ============================================================================
#
# `record_id` is an accession/reference identifier. It is NOT unique at the
# sequence-row level. A single record_id may correspond to many sequence rows.
#
# The biological identity of a sequence interval is represented by:
#
#     (record_id, start, end)
#
# This three-column tuple is the canonical composite key for the pipeline.
#
# The raw source may contain repeated occurrences of this tuple.
#
# The deduplicated corpus MUST contain exactly one row for each tuple.
#
# Because the deduplicated corpus guarantees uniqueness of this tuple, the
# same composite key can be used directly as the GPU join key. No additional
# serialized or surrogate row_key is required.
# ============================================================================

BIOLOGICAL_KEY_COLUMNS: Final[tuple[str, str, str]] = (
    "record_id",
    "start",
    "end",
)

BIOLOGICAL_KEY_UNIQUE_AFTER_DEDUP: Final[bool] = True

# GPU-derived assets join directly on the biological composite key.
#
# This is intentionally a tuple of source columns rather than a materialized
# `row_key` column.
GPU_JOIN_KEY: Final[tuple[str, str, str]] = BIOLOGICAL_KEY_COLUMNS


# ============================================================================
# 6. Valid categorical/token vocabularies
# ============================================================================
#
# These are contracts for categorical fields whose representations are
# established by the Carbon dataset.
#
# gene_type is intentionally NOT included here yet. The dataset exposes
# multiple gene_type classes, but this module should not invent a complete
# vocabulary from a partial sample.
#
# Validation of gene_type therefore focuses on field presence/type and
# non-nullness where appropriate until the complete vocabulary has been
# explicitly established.
# ============================================================================

VALID_SEQUENCE_TOKENS: Final[dict[str, frozenset[str]]] = {
    "begin_of_sequence": frozenset({"<s>"}),
    "end_of_sequence": frozenset({"</s>"}),
    "begin_of_gene": frozenset(
        {
            "<bog>",
            "<bok>",
        }
    ),
    "end_of_gene": frozenset(
        {
            "<eog>",
            "<eok>",
        }
    ),
    "species_type": frozenset(
        {
            "<fng>",
        }
    ),
    "strand": frozenset(
        {
            "<+>",
            "<->",
        }
    ),
    "molecule_type": frozenset(
        {
            "DNA",
        }
    ),
    "topology": frozenset(
        {
            "linear",
            "circular",
        }
    ),
}


# ============================================================================
# 7. Gene-boundary token pairing
# ============================================================================

GENE_BOUNDARY_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("<bog>", "<eog>"),
    ("<bok>", "<eok>"),
)


# ============================================================================
# 8. Nucleotide alphabet
# ============================================================================

IUPAC_NUCLEOTIDE_CHARS: Final[frozenset[str]] = frozenset("ACGTNRYSWKMBDHV-")

CLEAN_NUCLEOTIDE_CHARS: Final[frozenset[str]] = frozenset("ACGT")


# ============================================================================
# 9. Taxonomy expectations
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
# 10. Coordinate expectations
# ============================================================================

COORDINATE_COLUMNS: Final[tuple[str, str]] = (
    "start",
    "end",
)

MIN_COORDINATE_VALUE: Final[int] = 0


# ============================================================================
# 11. GPU enrichment output contracts
# ============================================================================
#
# Every GPU-derived asset preserves the canonical biological composite key:
#
#     (record_id, start, end)
#
# GPU assets must never depend on row position for alignment.
#
# The tokenized corpus additionally contains the model input representation.
# ============================================================================

TOKENIZED_CORPUS_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "start",
    "end",
    "token_ids",
    "token_mask",
    "token_length",
)

EMBEDDING_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "start",
    "end",
    "embedding",
    "embedding_norm",
)

LIKELIHOOD_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "start",
    "end",
    "mean_log_prob",
    "sum_log_prob",
    "perplexity",
    "supervised_position_count",
    "min_token_logprob",
    "argmin_position",
    "per_token_logprob_std",
)


# ============================================================================
# 12. GPU asset join key
# ============================================================================
#
# Every GPU-derived asset joins using the same three biological columns:
#
#     record_id
#     start
#     end
#
# The deduplicated corpus guarantees that this composite key is unique.
#
# No row_key/surrogate identity column is materialized.
#
# Downstream GPU assets must preserve these columns directly.
# ============================================================================

# GPU_JOIN_KEY is declared above alongside BIOLOGICAL_KEY_COLUMNS so that
# all identity-related contracts have a single source of truth.


# ============================================================================
# 13. GPU token-mask semantics
# ============================================================================
#
# token_mask is NOT a conventional binary attention mask.
#
#     -2       padding
#     -1       BPE/text token
#      0       DNA special token
#      1..k    partial/full k-mer contribution
#
# These values are part of the GPU pipeline's data contract and therefore
# must not be hardcoded independently in downstream stages.
#
# The maximum k-mer length is intentionally not defined here because it
# should come from the tokenizer/model configuration rather than being
# duplicated as a schema constant.
# ============================================================================

TOKEN_MASK_PADDING: Final[int] = -2

TOKEN_MASK_BPE_TEXT: Final[int] = -1

TOKEN_MASK_DNA_SPECIAL: Final[int] = 0

TOKEN_MASK_MIN_KMER_LENGTH: Final[int] = 1
