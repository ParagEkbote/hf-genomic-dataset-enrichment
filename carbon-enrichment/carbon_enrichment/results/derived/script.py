#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


HF_DATASET = "AINovice2005/carbon-pilot-corpus-dedup"

JOIN_COLUMNS = [
    "record_id",
    "start",
    "end",
]

SUPPORTED_RANKS = {
    "domain": 1,
    "kingdom": 2,
    "subkingdom": 3,
    "phylum": 4,
    "subphylum": 5,
    "class": 6,
    "subclass": 7,
    "order": 8,
    "suborder": 9,
    "family": 10,
    "genus": 11,
}


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join Layer 3 likelihood metrics to taxonomy and "
            "calculate taxonomic likelihood summaries."
        )
    )

    parser.add_argument(
        "--likelihood",
        default="/teamspace/studios/this_studio/hf-genomic-dataset-enrichment/carbon-enrichment/carbon_enrichment/results/derived/results/phase45/distributions/layer3_record_metrics.csv",
        help="Path to Layer 3 metrics CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/derived/taxonomic_likelihood",
        help="Directory for CSV results.",
    )

    parser.add_argument(
        "--rank",
        default="class",
        choices=list(SUPPORTED_RANKS),
        help="Taxonomic rank to analyze. Default: class.",
    )

    parser.add_argument(
        "--min-n",
        type=int,
        default=10,
        help=(
            "Minimum number of windows required for a taxon "
            "to appear in the result. Default: 10."
        ),
    )

    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.95,
        help="Upper likelihood tail. Default: 0.95.",
    )

    parser.add_argument(
        "--bottom-quantile",
        type=float,
        default=0.05,
        help="Lower likelihood tail. Default: 0.05.",
    )

    return parser.parse_args()


# ============================================================
# Load Layer 3 likelihood results
# ============================================================

def load_likelihood(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading likelihood metrics:")
    print(f"      {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Likelihood file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "record_id",
        "start",
        "end",
        "length_adjusted_likelihood",
        "length_adjusted_z",
        "length_adjusted_robust_z",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required likelihood columns: "
            + ", ".join(sorted(missing))
        )

    columns = [
        "record_id",
        "start",
        "end",
        "sequence_length",
        "mean_log_prob",
        "length_adjusted_likelihood",
        "length_adjusted_z",
        "length_adjusted_robust_z",
        "perplexity",
    ]

    df = df[
        [
            column
            for column in columns
            if column in df.columns
        ]
    ].copy()

    # Normalize join-key types.
    df["record_id"] = (
        df["record_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["start"] = pd.to_numeric(
        df["start"],
        errors="coerce",
    )

    df["end"] = pd.to_numeric(
        df["end"],
        errors="coerce",
    )

    # Normalize likelihood metrics.
    numeric_columns = [
        "sequence_length",
        "mean_log_prob",
        "length_adjusted_likelihood",
        "length_adjusted_z",
        "length_adjusted_robust_z",
        "perplexity",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.dropna(
        subset=[
            "record_id",
            "start",
            "end",
            "length_adjusted_likelihood",
        ]
    )

    df["start"] = df["start"].astype("int64")
    df["end"] = df["end"].astype("int64")

    # The analysis unit is the genomic window.
    duplicate_count = df.duplicated(
        subset=JOIN_COLUMNS
    ).sum()

    if duplicate_count:
        print(
            f"      WARNING: {duplicate_count:,} duplicate "
            "window rows detected."
        )

        # Keep the first observation. This prevents a many-to-many
        # taxonomy join from artificially multiplying observations.
        df = df.drop_duplicates(
            subset=JOIN_COLUMNS,
            keep="first",
        )

    print(f"      Rows: {len(df):,}")
    print(
        f"      Unique windows: "
        f"{df[JOIN_COLUMNS].drop_duplicates().shape[0]:,}"
    )
    print()

    return df


# ============================================================
# Load taxonomy
# ============================================================

def load_taxonomy() -> pd.DataFrame:
    print("[2/5] Loading taxonomy dataset:")
    print(f"      {HF_DATASET}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required.\n"
            "Install it with:\n"
            "    pip install datasets"
        ) from exc

    dataset = load_dataset(
        HF_DATASET,
        split="train",
    )

    required = {
        "record_id",
        "start",
        "end",
        "taxonomy",
    }

    missing = required - set(dataset.column_names)

    if missing:
        raise ValueError(
            "Taxonomy dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    columns = [
        "record_id",
        "start",
        "end",
        "taxonomy",
        "taxonomy_depth",
    ]

    columns = [
        column
        for column in columns
        if column in dataset.column_names
    ]

    df = dataset.select_columns(columns).to_pandas()

    df["record_id"] = (
        df["record_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["start"] = pd.to_numeric(
        df["start"],
        errors="coerce",
    )

    df["end"] = pd.to_numeric(
        df["end"],
        errors="coerce",
    )

    df["taxonomy"] = (
        df["taxonomy"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "record_id",
            "start",
            "end",
        ]
    )

    df = df[
        df["taxonomy"] != ""
    ].copy()

    df["start"] = df["start"].astype("int64")
    df["end"] = df["end"].astype("int64")

    # If multiple annotations exist for a window, retain the deepest
    # available taxonomy.
    if "taxonomy_depth" in df.columns:
        df["taxonomy_depth"] = pd.to_numeric(
            df["taxonomy_depth"],
            errors="coerce",
        ).fillna(0)

        df = (
            df.sort_values(
                JOIN_COLUMNS + ["taxonomy_depth"]
            )
            .drop_duplicates(
                subset=JOIN_COLUMNS,
                keep="last",
            )
        )
    else:
        df = df.drop_duplicates(
            subset=JOIN_COLUMNS,
            keep="first",
        )

    print(f"      Taxonomy rows: {len(df):,}")
    print()

    return df


# ============================================================
# Extract taxonomic rank
# ============================================================

def extract_taxon(
    taxonomy: str,
    rank: str,
) -> str:
    """
    Extract a taxonomic rank from a semicolon-delimited lineage.

    Expected lineage ordering:

        domain;
        kingdom;
        subkingdom;
        phylum;
        subphylum;
        class;
        subclass;
        order;
        suborder;
        family;
        genus
    """

    rank_number = SUPPORTED_RANKS[rank]

    parts = [
        part.strip()
        for part in str(taxonomy).split(";")
    ]

    index = rank_number - 1

    if index >= len(parts):
        return ""

    value = parts[index]

    if not value:
        return ""

    return value


# ============================================================
# Join likelihood to taxonomy
# ============================================================

def join_taxonomy(
    likelihood: pd.DataFrame,
    taxonomy: pd.DataFrame,
    rank: str,
) -> pd.DataFrame:
    print("[3/5] Joining likelihood windows to taxonomy...")
    print(
        "      Join key: "
        "(record_id, start, end)"
    )

    likelihood_n = len(likelihood)

    joined = likelihood.merge(
        taxonomy,
        on=JOIN_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    taxonomy_matched = (
        joined["taxonomy"]
        .notna()
        .sum()
    )

    joined["selected_taxon"] = (
        joined["taxonomy"]
        .map(
            lambda value: extract_taxon(
                value,
                rank,
            )
            if pd.notna(value)
            else ""
        )
    )

    joined["selected_taxon"] = (
        joined["selected_taxon"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    rank_matched = (
        joined["selected_taxon"] != ""
    ).sum()

    print(
        f"      Layer 3 windows: "
        f"{likelihood_n:,}"
    )

    print(
        f"      Taxonomy matched: "
        f"{taxonomy_matched:,}"
    )

    print(
        f"      {rank} matched: "
        f"{rank_matched:,}"
    )

    print(
        f"      {rank} unavailable: "
        f"{likelihood_n - rank_matched:,}"
    )

    if taxonomy_matched == 0:
        raise RuntimeError(
            "No rows matched taxonomy on "
            "(record_id, start, end)."
        )

    joined = joined[
        joined["selected_taxon"] != ""
    ].copy()

    print(
        f"      Final taxonomic observations: "
        f"{len(joined):,}"
    )

    print()

    return joined


# ============================================================
# Calculate taxonomic results
# ============================================================

def calculate_results(
    df: pd.DataFrame,
    rank: str,
    min_n: int,
    top_quantile: float,
    bottom_quantile: float,
) -> pd.DataFrame:
    print("[4/5] Calculating taxonomic likelihood statistics...")

    likelihood = df[
        "length_adjusted_likelihood"
    ]

    # Empirical cohort thresholds.
    upper_threshold = likelihood.quantile(
        top_quantile
    )

    lower_threshold = likelihood.quantile(
        bottom_quantile
    )

    df = df.copy()

    df["upper_tail"] = (
        df["length_adjusted_likelihood"]
        >= upper_threshold
    )

    df["lower_tail"] = (
        df["length_adjusted_likelihood"]
        <= lower_threshold
    )

    cohort_n = len(df)

    cohort_upper_rate = (
        df["upper_tail"].mean()
    )

    cohort_lower_rate = (
        df["lower_tail"].mean()
    )

    grouped = (
        df.groupby(
            "selected_taxon",
            dropna=False,
        )
        .agg(
            n=(
                "length_adjusted_likelihood",
                "size",
            ),

            mean_length_adjusted_likelihood=(
                "length_adjusted_likelihood",
                "mean",
            ),

            median_length_adjusted_likelihood=(
                "length_adjusted_likelihood",
                "median",
            ),

            sd_length_adjusted_likelihood=(
                "length_adjusted_likelihood",
                "std",
            ),

            mean_length_adjusted_z=(
                "length_adjusted_z",
                "mean",
            ),

            median_length_adjusted_z=(
                "length_adjusted_z",
                "median",
            ),

            mean_length_adjusted_robust_z=(
                "length_adjusted_robust_z",
                "mean",
            ),

            median_length_adjusted_robust_z=(
                "length_adjusted_robust_z",
                "median",
            ),

            upper_tail_n=(
                "upper_tail",
                "sum",
            ),

            lower_tail_n=(
                "lower_tail",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "selected_taxon": "taxon"
            }
        )
    )

    # Tail rates.
    grouped["upper_tail_rate"] = (
        grouped["upper_tail_n"]
        / grouped["n"]
    )

    grouped["lower_tail_rate"] = (
        grouped["lower_tail_n"]
        / grouped["n"]
    )

    # Enrichment relative to empirical cohort rate.
    if cohort_upper_rate > 0:
        grouped["upper_tail_enrichment"] = (
            grouped["upper_tail_rate"]
            / cohort_upper_rate
        )
    else:
        grouped["upper_tail_enrichment"] = float("nan")

    if cohort_lower_rate > 0:
        grouped["lower_tail_enrichment"] = (
            grouped["lower_tail_rate"]
            / cohort_lower_rate
        )
    else:
        grouped["lower_tail_enrichment"] = float("nan")

    grouped["taxonomic_rank"] = rank

    grouped["cohort_n"] = cohort_n

    grouped["cohort_upper_tail_rate"] = (
        cohort_upper_rate
    )

    grouped["cohort_lower_tail_rate"] = (
        cohort_lower_rate
    )

    grouped["upper_threshold"] = (
        upper_threshold
    )

    grouped["lower_threshold"] = (
        lower_threshold
    )

    # Minimum sample-size filter.
    grouped = grouped[
        grouped["n"] >= min_n
    ].copy()

    grouped = grouped[
        [
            "taxonomic_rank",
            "taxon",
            "n",

            "mean_length_adjusted_likelihood",
            "median_length_adjusted_likelihood",
            "sd_length_adjusted_likelihood",

            "mean_length_adjusted_z",
            "median_length_adjusted_z",

            "mean_length_adjusted_robust_z",
            "median_length_adjusted_robust_z",

            "upper_tail_n",
            "upper_tail_rate",
            "upper_tail_enrichment",

            "lower_tail_n",
            "lower_tail_rate",
            "lower_tail_enrichment",

            "cohort_n",
            "cohort_upper_tail_rate",
            "cohort_lower_tail_rate",

            "upper_threshold",
            "lower_threshold",
        ]
    ]

    # Primary ordering: taxonomic groups with the highest
    # median model-derived likelihood.
    grouped = grouped.sort_values(
        "median_length_adjusted_likelihood",
        ascending=False,
    )

    print(
        f"      Taxa retained (n >= {min_n}): "
        f"{len(grouped):,}"
    )

    print(
        f"      Upper threshold: "
        f"{upper_threshold:.6f}"
    )

    print(
        f"      Lower threshold: "
        f"{lower_threshold:.6f}"
    )

    print(
        f"      Upper-tail rate: "
        f"{cohort_upper_rate:.4%}"
    )

    print(
        f"      Lower-tail rate: "
        f"{cohort_lower_rate:.4%}"
    )

    print()

    return grouped


# ============================================================
# Save CSV only
# ============================================================

def save_results(
    results: pd.DataFrame,
    output_dir: str,
    rank: str,
) -> Path:
    print("[5/5] Writing CSV result...")

    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output
        / f"taxon_{rank}_likelihood_summary.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("Complete")
    print("=" * 60)
    print(f"Output:")
    print(f"  {output_path}")
    print("=" * 60)
    print()

    print("Top taxa by median length-adjusted likelihood:")
    print()

    display_columns = [
        "taxon",
        "n",
        "median_length_adjusted_likelihood",
        "upper_tail_enrichment",
        "lower_tail_enrichment",
    ]

    if not results.empty:
        print(
            results[
                display_columns
            ]
            .head(20)
            .to_string(index=False)
        )

    print()

    print("Top taxa by upper-tail enrichment:")
    print()

    if not results.empty:
        enrichment = (
            results
            .sort_values(
                "upper_tail_enrichment",
                ascending=False,
            )
        )

        print(
            enrichment[
                display_columns
            ]
            .head(20)
            .to_string(index=False)
        )

    return output_path


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()

    if not 0 < args.bottom_quantile < 0.5:
        raise ValueError(
            "--bottom-quantile must be between 0 and 0.5."
        )

    if not 0.5 < args.top_quantile < 1:
        raise ValueError(
            "--top-quantile must be between 0.5 and 1."
        )

    if args.bottom_quantile >= args.top_quantile:
        raise ValueError(
            "--bottom-quantile must be less than "
            "--top-quantile."
        )

    if args.min_n < 1:
        raise ValueError(
            "--min-n must be at least 1."
        )

    likelihood = load_likelihood(
        args.likelihood
    )

    taxonomy = load_taxonomy()

    joined = join_taxonomy(
        likelihood,
        taxonomy,
        args.rank,
    )

    results = calculate_results(
        joined,
        args.rank,
        args.min_n,
        args.top_quantile,
        args.bottom_quantile,
    )

    save_results(
        results,
        args.output_dir,
        args.rank,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())