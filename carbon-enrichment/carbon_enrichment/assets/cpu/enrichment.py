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


import math
from typing import Any

import dagster as dg
import numpy as np
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

# Byte lookup for the hot sequence-processing loop.
# Values 0..3 correspond to A/C/G/T and -1 means non-canonical.
_BASE_CODE = [-1] * 256
_BASE_CODE[ord("A")] = 0
_BASE_CODE[ord("C")] = 1
_BASE_CODE[ord("G")] = 2
_BASE_CODE[ord("T")] = 3


def _reverse_complement(sequence: str) -> str:
    """Return the IUPAC-aware reverse complement of a nucleotide sequence."""
    return sequence.translate(_IUPAC_COMPLEMENT)[::-1]


def _sequence_features(
    sequence: str,
) -> tuple[float, float, float, bool, list[float]]:
    """Calculate composition, QC, and 3-mer features with NumPy.

    Semantics intentionally match the original implementation:
    - GC content uses only canonical A/C/G/T bases.
    - GC skew uses canonical G/C counts.
    - Shannon entropy uses canonical A/C/G/T counts.
    - Any non-ACGT character marks the sequence as ambiguous.
    - Only canonical 3-mers contribute to the 64-dimensional vector.
    - K-mers crossing an ambiguous character are ignored.

    NumPy is used for the high-volume nucleotide/k-mer operations. The
    sequence is converted once to uint8 ASCII codes, mapped to 0..3 for
    A/C/G/T, and the 3-mer indices are generated as vectorized integer
    operations followed by ``np.bincount``.
    """
    if not sequence:
        return 0.0, 0.0, 0.0, False, [0.0] * KMER_VECTOR_SIZE

    # ASCII conversion is substantially cheaper than constructing one Python
    # string per k-mer. ``encode`` also preserves the existing ASCII-oriented
    # semantics of the input corpus.
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)

    # Map ASCII A/C/G/T to 0..3. All other bytes become -1.
    codes = np.full(raw.shape, -1, dtype=np.int8)
    codes[raw == 65] = 0   # A
    codes[raw == 67] = 1   # C
    codes[raw == 71] = 2   # G
    codes[raw == 84] = 3   # T

    canonical = codes >= 0
    ambiguous = bool(np.any(~canonical))

    counts = np.bincount(
        codes[canonical],
        minlength=4,
    ).astype(np.int64, copy=False)
    canonical_count = int(counts.sum())

    if canonical_count == 0:
        gc_content = 0.0
        entropy = 0.0
    else:
        gc_content = float((counts[1] + counts[2]) / canonical_count)

        probabilities = counts[counts > 0] / canonical_count
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    gc_count = int(counts[1] + counts[2])
    gc_skew = (
        float((counts[2] - counts[1]) / gc_count)
        if gc_count
        else 0.0
    )

    # Vectorized 3-mer encoding. A/C/G/T are represented by two bits, so the
    # integer index is identical to the canonical ACGT lexicographic index.
    if len(codes) < KMER_SIZE:
        kmer_vector = [0.0] * KMER_VECTOR_SIZE
    else:
        valid = (
            (codes[:-2] >= 0)
            & (codes[1:-1] >= 0)
            & (codes[2:] >= 0)
        )

        if np.any(valid):
            first = codes[:-2][valid].astype(np.int16, copy=False)
            second = codes[1:-1][valid].astype(np.int16, copy=False)
            third = codes[2:][valid].astype(np.int16, copy=False)

            kmer_indices = (first << 4) | (second << 2) | third
            kmer_counts = np.bincount(
                kmer_indices,
                minlength=KMER_VECTOR_SIZE,
            )
            valid_kmers = int(kmer_indices.size)
            kmer_vector = (
                kmer_counts.astype(np.float64) / valid_kmers
            ).tolist()
        else:
            kmer_vector = [0.0] * KMER_VECTOR_SIZE

    return gc_content, gc_skew, entropy, ambiguous, kmer_vector


# ============================================================================
# Taxonomy utilities
# ============================================================================


def _append_taxonomy_features(
    output: dict[str, list[Any]],
    taxonomy: Any,
) -> None:
    """Split one lineage and append its fixed-width taxonomy representation."""
    if isinstance(taxonomy, str):
        ranks = [
            part.strip()
            for part in taxonomy.split(";")
            if part.strip()
        ]
    else:
        ranks = []

    output["taxonomy_domain"].append(
        ranks[0] if ranks else None
    )

    for index in range(1, MAX_TAXONOMY_RANKS + 1):
        output[f"taxonomy_rank_{index}"].append(
            ranks[index] if index < len(ranks) else None
        )


# ============================================================================
# Quality utilities
# ============================================================================


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
    return (
        not sequence
        or begin_of_sequence != "<s>"
        or end_of_sequence != "</s>"
    )


def _quality_flag(
    sequence: str,
    entropy: float,
    ambiguous: bool,
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

    if ambiguous:
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
    begin_tokens = batch["begin_of_sequence"]
    end_tokens = batch["end_of_sequence"]

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

    append = {key: value.append for key, value in output.items()}

    for (
        sequence,
        start,
        end,
        strand,
        taxonomy,
        gene_type,
        begin_token,
        end_token,
    ) in zip(
        sequences,
        starts,
        ends,
        strands,
        taxonomies,
        gene_types,
        begin_tokens,
        end_tokens,
    ):
        sequence = sequence or ""

        gc, skew, entropy, ambiguous, kmer_vector = _sequence_features(
            sequence
        )

        start_int = int(start)
        end_int = int(end)
        gene_length = max(end_int - start_int, 0)

        # The raw schema provides start/end but not the parent sequence
        # length. This is therefore a coordinate-relative proxy rather than
        # a true genome-relative position.
        relative_position = start_int / max(end_int, 1)

        normalized_sequence = (
            _reverse_complement(sequence)
            if strand == "<->"
            else sequence
        )

        append["gc_content"](gc)
        append["gc_skew"](skew)
        append["sequence_length"](len(sequence))
        append["gene_length"](gene_length)
        append["shannon_entropy"](entropy)
        append["kmer_frequency_vector"](kmer_vector)
        append["relative_gene_position"](relative_position)
        append["strand_normalized_sequence"](normalized_sequence)

        if isinstance(taxonomy, str):
            ranks = [
                part.strip()
                for part in taxonomy.split(";")
                if part.strip()
            ]
        else:
            ranks = []

        append["taxonomy_domain"](
            ranks[0] if ranks else None
        )

        for rank in range(1, MAX_TAXONOMY_RANKS + 1):
            append[f"taxonomy_rank_{rank}"](
                ranks[rank] if rank < len(ranks) else None
            )

        append["is_coding_region"](
            gene_type in CODING_GENE_TYPES
        )

        append["qc_flag"](
            _quality_flag(
                sequence,
                entropy,
                ambiguous,
                begin_token,
                end_token,
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