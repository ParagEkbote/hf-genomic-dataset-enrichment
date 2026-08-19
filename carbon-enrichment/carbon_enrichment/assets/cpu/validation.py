"""
M1 — Data foundation: schema validation + quality checks.

These checks are attached to the `carbon_raw_sequences` Dagster asset rather
than being separate milestone gates. They run automatically whenever the
asset is materialized.

Only `check_raw_schema` is blocking. A schema break means downstream data
cannot be trusted. Other checks are treated as quality signals during
development so that minor corpus anomalies do not unnecessarily interrupt
the development loop.

Scale note:
    These checks materialize the current batch to pandas for vectorized
    validation. This is appropriate for the dev (100 rows) and integration
    (1%) validation tiers.

    Before the 25% authentication run, replace pandas-based full-batch
    validation with batched Dataset.map/filter reductions to keep memory
    usage bounded.
"""

import re

import dagster as dg

from carbon_enrichment.assets.cpu.ingest import carbon_raw_sequences
from carbon_enrichment.schema import (
    EXPECTED_COLUMNS,
    GENE_BOUNDARY_PAIRS,
    IUPAC_NUCLEOTIDE_CHARS,
    KNOWN_TAXONOMY_ROOTS,
    REQUIRED_FIELDS,
    VALID_SEQUENCE_TOKENS,
)


_CLEAN_ACGT_RE = re.compile(r"^[ACGT]+$")


# ============================================================================
# 1. Raw schema
# ============================================================================


@dg.asset_check(asset=carbon_raw_sequences, blocking=True)
def check_raw_schema(
    context: dg.AssetCheckExecutionContext,
    carbon_raw_sequences,
) -> dg.AssetCheckResult:
    """Verify that the raw dataset contains exactly the expected columns."""

    actual_columns = set(carbon_raw_sequences.column_names)
    expected_columns = set(EXPECTED_COLUMNS)

    missing = expected_columns - actual_columns
    unexpected = actual_columns - expected_columns

    passed = not missing and not unexpected

    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "missing_columns": dg.MetadataValue.json(sorted(missing)),
            "unexpected_columns": dg.MetadataValue.json(sorted(unexpected)),
            "expected_column_count": len(expected_columns),
            "actual_column_count": len(actual_columns),
        },
        description=(
            "All expected columns are present with no unexpected columns."
            if passed
            else (
                "Schema mismatch: "
                f"missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        ),
    )


# ============================================================================
# 2. Categorical/token vocabulary
# ============================================================================


@dg.asset_check(asset=carbon_raw_sequences)
def check_token_vocabularies(
    context: dg.AssetCheckExecutionContext,
    carbon_raw_sequences,
) -> dg.AssetCheckResult:
    """Verify that categorical/token columns contain recognized values."""

    df = carbon_raw_sequences.to_pandas()
    violations: dict[str, int] = {}

    for column, valid_values in VALID_SEQUENCE_TOKENS.items():
        if column not in df.columns:
            continue

        bad = ~df[column].isin(valid_values)
        n_bad = int(bad.sum())

        if n_bad:
            violations[column] = n_bad

    passed = not violations

    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "rows_checked": len(df),
            "violations_by_column": dg.MetadataValue.json(violations),
        },
        description=(
            "All categorical/token columns contain recognized values."
            if passed
            else f"Unexpected token values found: {violations}"
        ),
    )


# ============================================================================
# 3. Gene boundary pairing
# ============================================================================


@dg.asset_check(asset=carbon_raw_sequences)
def check_gene_boundary_pairing(
    context: dg.AssetCheckExecutionContext,
    carbon_raw_sequences,
) -> dg.AssetCheckResult:
    """Report gene-boundary token pairing for M2 triage.

    The corpus contains two recognized boundary-token families:

        <bog> -> <eog>
        <bok> -> <eok>

    Unexpected combinations are reported as diagnostic metadata rather than
    treated as a hard failure during development. This allows M2 to determine
    whether the observed record types should be split, filtered, or otherwise
    handled.
    """

    df = carbon_raw_sequences.to_pandas()
    valid_pairs = set(GENE_BOUNDARY_PAIRS)

    actual_pairs = list(
        zip(
            df["begin_of_gene"],
            df["end_of_gene"],
        )
    )

    mismatched = [
        pair for pair in actual_pairs if pair not in valid_pairs
    ]

    pair_counts: dict[str, int] = {}

    for pair in actual_pairs:
        key = f"{pair[0]}/{pair[1]}"
        pair_counts[key] = pair_counts.get(key, 0) + 1

    # Intentionally non-blocking. The purpose of this check is to characterize
    # the corpus and provide information for the downstream enrichment stage.
    return dg.AssetCheckResult(
        passed=True,
        metadata={
            "rows_checked": len(df),
            "boundary_pair_distribution": dg.MetadataValue.json(pair_counts),
            "mismatched_pair_count": len(mismatched),
        },
        description=(
            "Boundary token pair distribution captured for M2 triage. "
            f"{len(mismatched)} row(s) contain unexpected boundary pairs."
        ),
    )


# ============================================================================
# 4. Sequence alphabet
# ============================================================================


@dg.asset_check(asset=carbon_raw_sequences)
def check_sequence_alphabet(
    context: dg.AssetCheckExecutionContext,
    carbon_raw_sequences,
) -> dg.AssetCheckResult:
    """Verify that sequences are non-empty and use the accepted IUPAC alphabet."""

    df = carbon_raw_sequences.to_pandas()
    seqs = df["sequence"].astype(str)

    empty_mask = seqs.str.len() == 0
    n_empty = int(empty_mask.sum())

    non_empty = seqs[~empty_mask]

    clean_acgt_mask = non_empty.str.match(_CLEAN_ACGT_RE)
    n_clean = int(clean_acgt_mask.sum())

    def _has_invalid_chars(sequence: str) -> bool:
        return not set(sequence).issubset(IUPAC_NUCLEOTIDE_CHARS)

    invalid_mask = non_empty.apply(_has_invalid_chars)
    n_invalid = int(invalid_mask.sum())

    passed = n_empty == 0 and n_invalid == 0

    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "rows_checked": len(df),
            "empty_sequences": n_empty,
            "clean_acgt_only": n_clean,
            "clean_acgt_pct": round(
                100 * n_clean / max(len(non_empty), 1),
                2,
            ),
            "invalid_alphabet_rows": n_invalid,
            "min_sequence_length": (
                int(seqs.str.len().min()) if len(seqs) else None
            ),
            "max_sequence_length": (
                int(seqs.str.len().max()) if len(seqs) else None
            ),
            "mean_sequence_length": (
                round(float(seqs.str.len().mean()), 1)
                if len(seqs)
                else None
            ),
        },
        description=(
            "All sequences are non-empty and within the IUPAC nucleotide "
            "alphabet."
            if passed
            else (
                f"{n_empty} empty sequence(s), "
                f"{n_invalid} row(s) with non-IUPAC characters."
            )
        ),
    )


# ============================================================================
# 5. Coordinates
# ============================================================================


@dg.asset_check(asset=carbon_raw_sequences)
def check_coordinates(
    context: dg.AssetCheckExecutionContext,
    carbon_raw_sequences,
) -> dg.AssetCheckResult:
    """Verify non-negative genomic coordinates with start < end."""

    df = carbon_raw_sequences.to_pandas()

    negative = int(
        ((df["start"] < 0) | (df["end"] < 0)).sum()
    )

    inverted = int(
        (df["start"] >= df["end"]).sum()
    )

    passed = negative == 0 and inverted == 0

    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "rows_checked": len(df),
            "negative_coordinate_rows": negative,
            "start_gte_end_rows": inverted,
        },
        description=(
            "All coordinates are non-negative with start < end."
            if passed
            else (
                f"{negative} row(s) with negative coordinates, "
                f"{inverted} row(s) with start >= end."
            )
        ),
    )


# ============================================================================
# 6. Taxonomy format
# ============================================================================


@dg.asset_check(asset=carbon_raw_sequences)
def check_taxonomy_format(
    context: dg.AssetCheckExecutionContext,
    carbon_raw_sequences,
) -> dg.AssetCheckResult:
    """Verify non-empty taxonomy rooted at a recognized domain."""

    df = carbon_raw_sequences.to_pandas()
    taxonomy = df["taxonomy"].astype(str)

    empty = int((taxonomy.str.len() == 0).sum())

    def _root_ok(value: str) -> bool:
        root = value.split(";")[0]
        return root in KNOWN_TAXONOMY_ROOTS

    root_bad = int((~taxonomy.apply(_root_ok)).sum())

    thin_lineage = int(
        (taxonomy.str.count(";") < 1).sum()
    )

    passed = empty == 0 and root_bad == 0

    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "rows_checked": len(df),
            "empty_taxonomy": empty,
            "unknown_root_domain_rows": root_bad,
            "thin_lineage_rows": thin_lineage,
        },
        description=(
            "All taxonomy strings are non-empty and have a recognized "
            "root domain."
            if passed
            else (
                f"{empty} empty taxonomy value(s), "
                f"{root_bad} row(s) with unrecognized root domain."
            )
        ),
    )


# ============================================================================
# 7. Required-field completeness
# ============================================================================


@dg.asset_check(asset=carbon_raw_sequences)
def check_required_field_completeness(
    context: dg.AssetCheckExecutionContext,
    carbon_raw_sequences,
) -> dg.AssetCheckResult:
    """Verify that required identifier and metadata fields contain no nulls."""

    df = carbon_raw_sequences.to_pandas()

    null_counts = {
        column: int(df[column].isna().sum())
        for column in REQUIRED_FIELDS
        if column in df.columns
    }

    total_nulls = sum(null_counts.values())
    passed = total_nulls == 0

    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "rows_checked": len(df),
            "null_counts_by_column": dg.MetadataValue.json(null_counts),
            "total_null_values": total_nulls,
        },
        description=(
            "No nulls found in required fields."
            if passed
            else f"Null values found: {null_counts}"
        ),
    )