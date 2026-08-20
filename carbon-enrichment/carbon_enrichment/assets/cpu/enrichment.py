"""M2 — Carbon CPU framework-derived enrichment.

Pure, batch-oriented CPU enrichment kernel. This module does not materialize
Hugging Face Datasets and does not perform Parquet I/O.
"""

from typing import Any

import numpy as np

KMER_SIZE = 3
KMER_VECTOR_SIZE = 4**KMER_SIZE
LOW_COMPLEXITY_ENTROPY_THRESHOLD = 1.0

# Only includes tokens actually observed in the Carbon corpus. Do not
# silently map additional values (e.g. "CDS", "coding") to coding-region
# status unless they have been confirmed present in the data.
CODING_GENE_TYPES = frozenset({
    "<cds>",
})

_IUPAC_COMPLEMENT = str.maketrans({
    "A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
    "R": "Y", "Y": "R", "S": "S", "W": "W", "K": "M",
    "M": "K", "B": "V", "V": "B", "D": "H", "H": "D", "-": "-",
})


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_IUPAC_COMPLEMENT)[::-1]


def _sequence_features(
    sequence: str,
) -> tuple[float, float, float, bool, list[float]]:
    """Calculate composition, QC, and 3-mer features with NumPy."""
    if not sequence:
        return 0.0, 0.0, 0.0, False, [0.0] * KMER_VECTOR_SIZE

    raw = np.frombuffer(
        sequence.encode("ascii", "replace"),
        dtype=np.uint8,
    )

    codes = np.full(raw.shape, -1, dtype=np.int8)
    codes[raw == 65] = 0  # A
    codes[raw == 67] = 1  # C
    codes[raw == 71] = 2  # G
    codes[raw == 84] = 3  # T

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
        gc_content = float(
            (counts[1] + counts[2]) / canonical_count
        )
        probabilities = counts[counts > 0] / canonical_count
        entropy = float(
            -np.sum(probabilities * np.log2(probabilities))
        )

    gc_count = int(counts[1] + counts[2])
    gc_skew = (
        float((counts[2] - counts[1]) / gc_count)
        if gc_count
        else 0.0
    )

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


def _is_truncated(
    sequence: str,
    begin_of_sequence: Any,
    end_of_sequence: Any,
) -> bool:
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


def enrich_batch(
    batch: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Enrich one bounded batch while preserving all raw columns."""
    sequences = batch["sequence"]
    starts = batch["start"]
    ends = batch["end"]
    strands = batch["strand"]
    taxonomies = batch["taxonomy"]
    gene_types = batch["gene_type"]
    begin_tokens = batch["begin_of_sequence"]
    end_tokens = batch["end_of_sequence"]

    # Preserve every original Carbon column.
    output: dict[str, list[Any]] = {
        column: list(values)
        for column, values in batch.items()
    }

    # Derived features.
    output.update(
        {
            "gc_content": [],
            "gc_skew": [],
            "sequence_length": [],
            "gene_length": [],
            "shannon_entropy": [],
            "kmer_frequency_vector": [],
            "relative_gene_position": [],
            "strand_normalized_sequence": [],
            "taxonomy_domain": [],
            "is_coding_region": [],
            "qc_flag": [],
        }
    )

    append = {
        key: output[key].append
        for key in (
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
        )
    }

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

        # Coordinate-relative proxy; not genome-relative without parent length.
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

        append["taxonomy_domain"](ranks[0] if ranks else None)

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