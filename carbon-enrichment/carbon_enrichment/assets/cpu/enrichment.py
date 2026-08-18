"""
M2 — CPU framework-derived enrichment.

This asset consumes the normalized Carbon dataset and adds deterministic,
model-independent biological/data-quality features.

Features produced
-----------------
Composition:
    - gc_content
    - gc_skew
    - sequence_length
    - gene_length
    - shannon_entropy
    - kmer_frequency_vector

Structural / positional:
    - relative_gene_position
    - strand_normalized_sequence
    - taxonomy_domain
    - taxonomy_rank_1 ... taxonomy_rank_7
    - is_coding_region

Quality:
    - qc_flag

Design principles
-----------------
1. CPU-only. No Carbon model or GPU dependency.
2. Deterministic. The same normalized input produces the same output.
3. Batch-oriented. Hugging Face Dataset.map(..., batched=True) is used to
   avoid materializing the complete dataset as pandas.
4. Row count is preserved.
5. The input normalized dataset is not modified in place.
6. Model-derived features such as likelihoods and embeddings are explicitly
   excluded; those belong to the GPU tier.

Important semantic assumptions
------------------------------
`relative_gene_position`:
    The current raw schema provides start/end coordinates but does not
    provide an explicit parent sequence/contig length. Therefore the
    implementation uses:

        start / end

    as a coordinate-relative position proxy.

    This should NOT be described as a genome-wide relative position until
    a parent sequence length is available. If a true parent/contig length
    becomes available later, this feature should be replaced with:

        start / parent_sequence_length

`is_coding_region`:
    This is derived conservatively from gene_type using a small configurable
    vocabulary. Unknown gene types are represented as False rather than
    inferred as non-coding with biological certainty.

`qc_flag`:
    The priority order is:
        truncated
        ambiguous_bases
        low_complexity
        clean

    These are data-quality labels, not biological labels.

Taxonomy:
    Taxonomy is represented as a semicolon-delimited lineage. The first
    component is stored as `taxonomy_domain`; subsequent components are
    stored in fixed generic rank columns. Exact biological rank names are
    deliberately not inferred from position alone.
"""

from __future__ import annotations

import math
from typing import Any

import dagster as dg
from datasets import Dataset


# ============================================================================
# Configuration
# ============================================================================

# 3-mer representation:
#
#     4^3 = 64 possible canonical A/C/G/T k-mers
#
# This is intentionally kept small because the vector is currently intended
# as a CPU clustering baseline rather than a high-dimensional sequence model.
KMER_SIZE = 3
KMER_VECTOR_SIZE = 4**KMER_SIZE

_BASES = "ACGT"

_KMER_INDEX = {
    f"{a}{b}{c}": i
    for i, (a, b, c) in enumerate(
        (
            (a, b, c)
            for a in _BASES
            for b in _BASES
            for c in _BASES
        )
    )
}


# Low-complexity threshold.
#
# Shannon entropy for a four-symbol alphabet ranges from 0 to 2 bits.
# A threshold of 1.0 is intentionally conservative enough to identify
# strongly compositionally biased sequences without claiming that every
# low-entropy sequence is biologically problematic.
LOW_COMPLEXITY_ENTROPY_THRESHOLD = 1.0


# Gene types that are explicitly treated as coding.
#
# This vocabulary should be expanded only after inspecting the actual
# Carbon corpus values. Unknown values remain False.
CODING_GENE_TYPES = frozenset(
    {
        "CDS",
        "cds",
        "coding",
        "protein_coding",
    }
)


# Seven generic ranks after the domain/root.
#
# These are deliberately called rank_1, rank_2, ... rather than assigning
# biological names such as phylum/class/order unless the dataset itself
# provides explicit rank labels.
MAX_TAXONOMY_RANKS = 7


# ============================================================================
# Sequence utilities
# ============================================================================

_IUPAC_COMPLEMENT = str.maketrans(
    {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "V": "B",
        "D": "H",
        "H": "D",
        "-": "-",
    }
)


def _reverse_complement(sequence: str) -> str:
    """Return the IUPAC-aware reverse complement of a nucleotide sequence."""

    return sequence.translate(_IUPAC_COMPLEMENT)[::-1]


def _gc_content(sequence: str) -> float:
    """Calculate GC fraction among canonical A/C/G/T bases."""

    if not sequence:
        return 0.0

    canonical = [
        base
        for base in sequence
        if base in _BASES
    ]

    if not canonical:
        return 0.0

    gc = sum(
        base in {"G", "C"}
        for base in canonical
    )

    return gc / len(canonical)


def _gc_skew(sequence: str) -> float:
    """Calculate GC skew: (G-C)/(G+C).

    Returns 0.0 when neither G nor C is present.
    """

    g = sequence.count("G")
    c = sequence.count("C")

    denominator = g + c

    if denominator == 0:
        return 0.0

    return (g - c) / denominator


def _shannon_entropy(sequence: str) -> float:
    """Calculate Shannon entropy over canonical A/C/G/T bases."""

    if not sequence:
        return 0.0

    counts = {
        base: sequence.count(base)
        for base in _BASES
    }

    total = sum(counts.values())

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counts.values():
        if count == 0:
            continue

        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def _kmer_frequency_vector(sequence: str) -> list[float]:
    """Return a normalized 3-mer frequency vector of length 64.

    Only canonical A/C/G/T k-mers contribute to the vector. K-mers containing
    IUPAC ambiguity characters are ignored rather than assigned arbitrarily.
    """

    vector = [0.0] * KMER_VECTOR_SIZE

    if len(sequence) < KMER_SIZE:
        return vector

    valid_kmers = 0

    for index in range(len(sequence) - KMER_SIZE + 1):
        kmer = sequence[index : index + KMER_SIZE]

        kmer_index = _KMER_INDEX.get(kmer)

        if kmer_index is None:
            continue

        vector[kmer_index] += 1.0
        valid_kmers += 1

    if valid_kmers:
        vector = [
            value / valid_kmers
            for value in vector
        ]

    return vector


# ============================================================================
# Taxonomy utilities
# ============================================================================


def _split_taxonomy(value: Any) -> list[str]:
    """Split a semicolon-delimited taxonomy lineage."""

    if value is None:
        return []

    if not isinstance(value, str):
        return []

    return [
        part.strip()
        for part in value.split(";")
        if part.strip()
    ]


def _taxonomy_features(
    taxonomy: Any,
) -> dict[str, str | None]:
    """Create fixed-width taxonomy columns."""

    ranks = _split_taxonomy(taxonomy)

    output: dict[str, str | None] = {
        "taxonomy_domain": ranks[0] if ranks else None,
    }

    for index in range(1, MAX_TAXONOMY_RANKS + 1):
        output[f"taxonomy_rank_{index}"] = (
            ranks[index]
            if index < len(ranks)
            else None
        )

    return output


# ============================================================================
# Quality utilities
# ============================================================================


def _has_ambiguous_bases(sequence: str) -> bool:
    """Return True when a sequence contains IUPAC ambiguity characters."""

    return any(
        base not in _BASES
        for base in sequence
    )


def _is_truncated(
    sequence: str,
    begin_of_sequence: Any,
    end_of_sequence: Any,
) -> bool:
    """Identify records lacking expected sequence boundary markers.

    The Carbon raw schema exposes begin/end sequence tokens. We use these as
    a data-integrity signal rather than making a biological claim about
    truncation.
    """

    if not sequence:
        return True

    return (
        begin_of_sequence != "<s>"
        or end_of_sequence != "</s>"
    )


def _quality_flag(
    sequence: str,
    entropy: float,
    begin_of_sequence: Any,
    end_of_sequence: Any,
) -> str:
    """Assign the highest-priority applicable data-quality flag."""

    if _is_truncated(
        sequence,
        begin_of_sequence,
        end_of_sequence,
    ):
        return "truncated"

    if _has_ambiguous_bases(sequence):
        return "ambiguous_bases"

    if entropy < LOW_COMPLEXITY_ENTROPY_THRESHOLD:
        return "low_complexity"

    return "clean"


# ============================================================================
# Batch enrichment
# ============================================================================


def _enrich_batch(
    batch: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Generate framework-derived features for one Dataset batch."""

    sequences = batch["sequence"]
    starts = batch["start"]
    ends = batch["end"]
    strands = batch["strand"]
    taxonomies = batch["taxonomy"]
    gene_types = batch["gene_type"]

    output: dict[str, list[Any]] = {
        "gc_content": [],
        "gc_skew": [],
        "sequence_length": [],
        "gene_length": [],
        "shannon_entropy": [],
        "kmer_frequency_vector": [],
        "relative_gene_position": [],
        "strand_normalized_sequence": [],
        "taxonomy_domain": [],
    }

    for index in range(1, MAX_TAXONOMY_RANKS + 1):
        output[f"taxonomy_rank_{index}"] = []

    output["is_coding_region"] = []
    output["qc_flag"] = []

    for sequence, start, end, strand, taxonomy, gene_type in zip(
        sequences,
        starts,
        ends,
        strands,
        taxonomies,
        gene_types,
    ):
        sequence = sequence or ""

        gc = _gc_content(sequence)
        skew = _gc_skew(sequence)
        length = len(sequence)
        entropy = _shannon_entropy(sequence)

        gene_length = max(
            int(end) - int(start),
            0,
        )

        # The raw schema provides start/end but not the parent sequence
        # length. This is therefore a coordinate-relative proxy rather than
        # a true genome-relative position.
        denominator = max(int(end), 1)
        relative_position = int(start) / denominator

        normalized_sequence = (
            _reverse_complement(sequence)
            if strand == "<->"
            else sequence
        )

        taxonomy_values = _taxonomy_features(taxonomy)

        output["gc_content"].append(gc)
        output["gc_skew"].append(skew)
        output["sequence_length"].append(length)
        output["gene_length"].append(gene_length)
        output["shannon_entropy"].append(entropy)
        output["kmer_frequency_vector"].append(
            _kmer_frequency_vector(sequence)
        )
        output["relative_gene_position"].append(
            relative_position
        )
        output["strand_normalized_sequence"].append(
            normalized_sequence
        )
        output["taxonomy_domain"].append(
            taxonomy_values["taxonomy_domain"]
        )

        for rank in range(1, MAX_TAXONOMY_RANKS + 1):
            output[f"taxonomy_rank_{rank}"].append(
                taxonomy_values[f"taxonomy_rank_{rank}"]
            )

        output["is_coding_region"].append(
            gene_type in CODING_GENE_TYPES
        )

        output["qc_flag"].append(
            _quality_flag(
                sequence,
                entropy,
                batch["begin_of_sequence"][
                    len(output["qc_flag"])
                ],
                batch["end_of_sequence"][
                    len(output["qc_flag"])
                ],
            )
        )

    return output


# ============================================================================
# Dagster asset
# ============================================================================


@dg.asset(
    group_name="cpu_enrichment",
    io_manager_key="hf_parquet_io_manager",
    compute_kind="cpu",
    description=(
        "CPU-only, framework-derived enrichment of normalized Carbon "
        "sequences. Produces composition, structural, taxonomy, and "
        "data-quality features without model inference."
    ),
)
def carbon_cpu_enriched_sequences(
    context: dg.AssetExecutionContext,
    carbon_normalized_sequences: Dataset,
) -> Dataset:
    """Create the CPU-derived enrichment dataset."""

    input_rows = carbon_normalized_sequences.num_rows

    context.log.info(
        f"Computing CPU framework enrichment for "
        f"{input_rows:,} Carbon rows."
    )

    enriched = carbon_normalized_sequences.map(
        _enrich_batch,
        batched=True,
        batch_size=1_000,
        desc="CPU framework enrichment",
    )

    output_rows = enriched.num_rows

    if output_rows != input_rows:
        raise RuntimeError(
            "CPU enrichment changed the number of rows: "
            f"input={input_rows}, output={output_rows}"
        )

    expected_new_columns = {
        "gc_content",
        "gc_skew",
        "sequence_length",
        "gene_length",
        "shannon_entropy",
        "kmer_frequency_vector",
        "relative_gene_position",
        "strand_normalized_sequence",
        "taxonomy_domain",
        "is_coding_region",
        "qc_flag",
    }

    expected_new_columns.update(
        f"taxonomy_rank_{index}"
        for index in range(1, MAX_TAXONOMY_RANKS + 1)
    )

    missing_columns = expected_new_columns - set(
        enriched.column_names
    )

    if missing_columns:
        raise RuntimeError(
            "CPU enrichment did not produce all expected columns: "
            f"{sorted(missing_columns)}"
        )

    context.add_output_metadata(
        {
            "input_rows": input_rows,
            "output_rows": output_rows,
            "rows_preserved": output_rows == input_rows,
            "kmer_size": KMER_SIZE,
            "kmer_vector_dimensions": KMER_VECTOR_SIZE,
            "low_complexity_entropy_threshold": (
                LOW_COMPLEXITY_ENTROPY_THRESHOLD
            ),
            "taxonomy_rank_columns": MAX_TAXONOMY_RANKS,
            "new_columns": dg.MetadataValue.md(
                "\n".join(
                    f"- `{column}`"
                    for column in sorted(expected_new_columns)
                )
            ),
        }
    )

    return enriched