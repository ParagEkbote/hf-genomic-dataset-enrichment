from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment/results/derived/case_study/phase45/record_metrics.csv"
DEFAULT_OUTPUT_DIR = "results/case_study/phase45"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_rate(numerator, denominator):
    """Return numerator / denominator, with NaN for zero denominators."""
    return np.where(
        denominator > 0,
        numerator / denominator,
        np.nan,
    )


def q95(series):
    return series.quantile(0.95)


def q99(series):
    return series.quantile(0.99)


def correlation(a, b):
    valid = a.notna() & b.notna()

    if valid.sum() < 2:
        return np.nan

    return a[valid].corr(b[valid])


# ---------------------------------------------------------------------
# Load and normalize
# ---------------------------------------------------------------------

def load_record_metrics(path: Path) -> pd.DataFrame:
    print("=" * 72)
    print("PHASE 4.5 TAXON ANALYSIS")
    print("=" * 72)
    print(f"Input: {path}")

    df = pd.read_csv(path)

    print(f"Records: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    required = [
        "record_id",
        "taxon_group",
        "multi_layer_anomaly_score",
        "is_gc_outlier",
        "is_length_outlier",
        "is_likelihood_outlier",
        "is_embedding_outlier",
        "outlier_layer_count",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "sequence_length",
        "gc_content",
        "shannon_entropy",
        "gene_length",
        "mean_log_prob",
        "sum_log_prob",
        "perplexity",
        "supervised_position_count",
        "min_token_logprob",
        "argmin_position",
        "per_token_logprob_std",
        "embedding_norm",
        "gc_z",
        "length_z",
        "perplexity_z",
        "embedding_norm_z",
        "entropy_z",
        "min_token_logprob_z",
        "token_std_z",
        "multi_layer_anomaly_score",
        "outlier_layer_count",
        "normalized_argmin_position",
        "bounded_argmin_position",
        "distance_to_sequence_boundary",
        "non_embedding_anomaly",
        "likelihood_signal",
        "cpu_signal",
        "embedding_signal",
        "intra_taxon_gc_percentile",
        "intra_taxon_length_percentile",
        "intra_taxon_perplexity_percentile",
        "intra_taxon_embedding_norm_percentile",
        "distance_to_nearest_gene_boundary",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    boolean_columns = [
        "is_gc_outlier",
        "is_length_outlier",
        "is_likelihood_outlier",
        "is_embedding_outlier",
        "argmin_inside_gene_boundary",
    ]

    for column in boolean_columns:
        if column not in df.columns:
            continue

        if df[column].dtype == object:
            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                    }
                )
            )

        df[column] = (
            df[column]
            .fillna(False)
            .astype(bool)
        )

    # Treat missing taxon as an explicit category.
    df["taxon_group"] = (
        df["taxon_group"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    return df


# ---------------------------------------------------------------------
# Tier 1 + Tier 2
# ---------------------------------------------------------------------

def build_taxon_summary(df: pd.DataFrame) -> pd.DataFrame:
    print()
    print("Building taxon_summary.csv...")

    grouped = df.groupby(
        "taxon_group",
        sort=True,
        dropna=False,
    )

    summary = grouped.agg(
        # -------------------------------------------------------------
        # Population
        # -------------------------------------------------------------

        record_count=(
            "record_id",
            "size",
        ),

        # -------------------------------------------------------------
        # Anomaly distribution
        # -------------------------------------------------------------

        mean_anomaly_score=(
            "multi_layer_anomaly_score",
            "mean",
        ),

        median_anomaly_score=(
            "multi_layer_anomaly_score",
            "median",
        ),

        p95_anomaly_score=(
            "multi_layer_anomaly_score",
            q95,
        ),

        p99_anomaly_score=(
            "multi_layer_anomaly_score",
            q99,
        ),

        max_anomaly_score=(
            "multi_layer_anomaly_score",
            "max",
        ),

        # -------------------------------------------------------------
        # Tier 1: individual outlier layers
        # -------------------------------------------------------------

        gc_outliers=(
            "is_gc_outlier",
            "sum",
        ),

        length_outliers=(
            "is_length_outlier",
            "sum",
        ),

        likelihood_outliers=(
            "is_likelihood_outlier",
            "sum",
        ),

        embedding_outliers=(
            "is_embedding_outlier",
            "sum",
        ),

        # -------------------------------------------------------------
        # Tier 1: simultaneous anomalies
        # -------------------------------------------------------------

        any_outliers=(
            "outlier_layer_count",
            lambda x: (x >= 1).sum(),
        ),

        multi_layer_2plus=(
            "outlier_layer_count",
            lambda x: (x >= 2).sum(),
        ),

        multi_layer_3plus=(
            "outlier_layer_count",
            lambda x: (x >= 3).sum(),
        ),

        # -------------------------------------------------------------
        # Tier 2: global z-score behavior
        # -------------------------------------------------------------

        mean_gc_z=(
            "gc_z",
            "mean",
        ),

        sd_gc_z=(
            "gc_z",
            "std",
        ),

        mean_length_z=(
            "length_z",
            "mean",
        ),

        sd_length_z=(
            "length_z",
            "std",
        ),

        mean_perplexity_z=(
            "perplexity_z",
            "mean",
        ),

        sd_perplexity_z=(
            "perplexity_z",
            "std",
        ),

        mean_embedding_norm_z=(
            "embedding_norm_z",
            "mean",
        ),

        sd_embedding_norm_z=(
            "embedding_norm_z",
            "std",
        ),

        # -------------------------------------------------------------
        # Tier 2: within-taxon relative position
        # -------------------------------------------------------------

        mean_taxon_gc_percentile=(
            "intra_taxon_gc_percentile",
            "mean",
        ),

        mean_taxon_length_percentile=(
            "intra_taxon_length_percentile",
            "mean",
        ),

        mean_taxon_perplexity_percentile=(
            "intra_taxon_perplexity_percentile",
            "mean",
        ),

        mean_taxon_embedding_percentile=(
            "intra_taxon_embedding_norm_percentile",
            "mean",
        ),

        # -------------------------------------------------------------
        # Tier 2: within-taxon upper tails
        # -------------------------------------------------------------

        gc_taxon_upper_tail=(
            "intra_taxon_gc_percentile",
            lambda x: (x >= 0.95).sum(),
        ),

        length_taxon_upper_tail=(
            "intra_taxon_length_percentile",
            lambda x: (x >= 0.95).sum(),
        ),

        perplexity_taxon_upper_tail=(
            "intra_taxon_perplexity_percentile",
            lambda x: (x >= 0.95).sum(),
        ),

        embedding_taxon_upper_tail=(
            "intra_taxon_embedding_norm_percentile",
            lambda x: (x >= 0.95).sum(),
        ),

        # -------------------------------------------------------------
        # Tier 2: signal levels
        # -------------------------------------------------------------

        mean_cpu_signal=(
            "cpu_signal",
            "mean",
        ),

        mean_likelihood_signal=(
            "likelihood_signal",
            "mean",
        ),

        mean_embedding_signal=(
            "embedding_signal",
            "mean",
        ),

        mean_non_embedding_anomaly=(
            "non_embedding_anomaly",
            "mean",
        ),
    ).reset_index()

    n = summary["record_count"]

    # -----------------------------------------------------------------
    # Rates
    # -----------------------------------------------------------------

    summary["any_outlier_rate"] = safe_rate(
        summary["any_outliers"],
        n,
    )

    summary["gc_outlier_rate"] = safe_rate(
        summary["gc_outliers"],
        n,
    )

    summary["length_outlier_rate"] = safe_rate(
        summary["length_outliers"],
        n,
    )

    summary["likelihood_outlier_rate"] = safe_rate(
        summary["likelihood_outliers"],
        n,
    )

    summary["embedding_outlier_rate"] = safe_rate(
        summary["embedding_outliers"],
        n,
    )

    summary["multi_layer_2plus_rate"] = safe_rate(
        summary["multi_layer_2plus"],
        n,
    )

    summary["multi_layer_3plus_rate"] = safe_rate(
        summary["multi_layer_3plus"],
        n,
    )

    summary["gc_taxon_upper_tail_rate"] = safe_rate(
        summary["gc_taxon_upper_tail"],
        n,
    )

    summary["length_taxon_upper_tail_rate"] = safe_rate(
        summary["length_taxon_upper_tail"],
        n,
    )

    summary["perplexity_taxon_upper_tail_rate"] = safe_rate(
        summary["perplexity_taxon_upper_tail"],
        n,
    )

    summary["embedding_taxon_upper_tail_rate"] = safe_rate(
        summary["embedding_taxon_upper_tail"],
        n,
    )

    # -----------------------------------------------------------------
    # Dominant outlier layer
    # -----------------------------------------------------------------

    layer_columns = [
        "gc_outliers",
        "length_outliers",
        "likelihood_outliers",
        "embedding_outliers",
    ]

    layer_names = {
        "gc_outliers": "gc",
        "length_outliers": "length",
        "likelihood_outliers": "likelihood",
        "embedding_outliers": "embedding",
    }

    summary["dominant_outlier_layer"] = (
        summary[layer_columns]
        .idxmax(axis=1)
        .map(layer_names)
    )

    summary["dominant_outlier_count"] = (
        summary[layer_columns].max(axis=1)
    )

    # -----------------------------------------------------------------
    # Sort by strongest tail anomaly first.
    # -----------------------------------------------------------------

    summary = summary.sort_values(
        [
            "p99_anomaly_score",
            "record_count",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return summary


# ---------------------------------------------------------------------
# Tier 2: signal relationships
# ---------------------------------------------------------------------

def build_signal_profile(df: pd.DataFrame) -> pd.DataFrame:
    print("Building taxon_signal_profile.csv...")

    grouped = df.groupby(
        "taxon_group",
        sort=True,
        dropna=False,
    )

    rows = []

    for taxon, group in grouped:

        row = {
            "taxon_group": taxon,
            "record_count": len(group),

            # ---------------------------------------------------------
            # Mean signals
            # ---------------------------------------------------------

            "mean_cpu_signal": group["cpu_signal"].mean(),
            "mean_likelihood_signal": (
                group["likelihood_signal"].mean()
            ),
            "mean_embedding_signal": (
                group["embedding_signal"].mean()
            ),
            "mean_non_embedding_anomaly": (
                group["non_embedding_anomaly"].mean()
            ),

            # ---------------------------------------------------------
            # Signal relationships
            # ---------------------------------------------------------

            "cpu_likelihood_corr": correlation(
                group["cpu_signal"],
                group["likelihood_signal"],
            ),

            "cpu_embedding_corr": correlation(
                group["cpu_signal"],
                group["embedding_signal"],
            ),

            "likelihood_embedding_corr": correlation(
                group["likelihood_signal"],
                group["embedding_signal"],
            ),
        }

        # -------------------------------------------------------------
        # Dominant signal
        # -------------------------------------------------------------

        signals = {
            "cpu": row["mean_cpu_signal"],
            "likelihood": row["mean_likelihood_signal"],
            "embedding": row["mean_embedding_signal"],
        }

        valid = {
            key: value
            for key, value in signals.items()
            if pd.notna(value)
        }

        if valid:
            row["dominant_signal"] = max(
                valid,
                key=valid.get,
            )
        else:
            row["dominant_signal"] = np.nan

        # -------------------------------------------------------------
        # Signal agreement
        # -------------------------------------------------------------

        signal_df = group[
            [
                "cpu_signal",
                "likelihood_signal",
                "embedding_signal",
            ]
        ]

        positive = (signal_df > 0).sum(axis=1)
        negative = (signal_df < 0).sum(axis=1)

        row["all_three_positive_rate"] = (
            (positive == 3).mean()
        )

        row["all_three_negative_rate"] = (
            (negative == 3).mean()
        )

        row["mixed_signal_rate"] = (
            ((positive > 0) & (negative > 0)).mean()
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Tier 3: candidate discovery
# ---------------------------------------------------------------------

def build_candidates(
    summary: pd.DataFrame,
    signal_profile: pd.DataFrame,
    min_records: int,
) -> pd.DataFrame:

    print("Building taxon_candidates.csv...")

    result = summary.merge(
        signal_profile,
        on=[
            "taxon_group",
            "record_count",
        ],
        how="left",
        suffixes=("", "_signal"),
    )

    # Candidate analysis should not rank tiny taxa.
    result = result[
        result["record_count"] >= min_records
    ].copy()

    # -----------------------------------------------------------------
    # Reference distributions
    #
    # These are discovery heuristics, not statistical significance tests.
    # -----------------------------------------------------------------

    anomaly_rate_median = (
        result["any_outlier_rate"].median()
    )

    multilayer_rate_median = (
        result["multi_layer_2plus_rate"].median()
    )

    result["high_anomaly_rate_candidate"] = (
        result["any_outlier_rate"]
        > anomaly_rate_median
    )

    result["high_multilayer_candidate"] = (
        result["multi_layer_2plus_rate"]
        > multilayer_rate_median
    )

    # -----------------------------------------------------------------
    # Model-only candidate
    #
    # Positive likelihood + embedding signals while CPU signal is not
    # positive.
    # -----------------------------------------------------------------

    result["likelihood_embedding_without_cpu"] = (
        (result["mean_likelihood_signal"] > 0)
        & (result["mean_embedding_signal"] > 0)
        & (result["mean_cpu_signal"] <= 0)
    )

    # -----------------------------------------------------------------
    # Conventional-feature-driven candidate
    # -----------------------------------------------------------------

    result["cpu_without_model_support"] = (
        (result["mean_cpu_signal"] > 0)
        & (result["mean_likelihood_signal"] <= 0)
        & (result["mean_embedding_signal"] <= 0)
    )

    # -----------------------------------------------------------------
    # Broad vs sparse anomaly behavior
    #
    # mean / p99:
    #   higher -> anomaly is distributed more broadly
    #
    # max / mean:
    #   higher -> a few extreme records dominate
    # -----------------------------------------------------------------

    result["mean_to_p99_anomaly_ratio"] = (
        result["mean_anomaly_score"]
        / result["p99_anomaly_score"].replace(
            0,
            np.nan,
        )
    )

    result["max_to_mean_anomaly_ratio"] = (
        result["max_anomaly_score"]
        / result["mean_anomaly_score"].replace(
            0,
            np.nan,
        )
    )

    # -----------------------------------------------------------------
    # Candidate classification
    # -----------------------------------------------------------------

    def classify(row):

        if row["multi_layer_3plus_rate"] > 0:
            return "multi_layer"

        if row["likelihood_embedding_without_cpu"]:
            return "likelihood_embedding_without_cpu"

        if row["high_multilayer_candidate"]:
            return "high_multilayer_rate"

        if row["high_anomaly_rate_candidate"]:
            return "high_anomaly_rate"

        if (
            pd.notna(row["max_to_mean_anomaly_ratio"])
            and row["max_to_mean_anomaly_ratio"] >= 3
        ):
            return "sparse_extreme_records"

        return "ordinary"

    result["candidate_category"] = result.apply(
        classify,
        axis=1,
    )

    # Strong candidates first.
    category_order = {
        "multi_layer": 0,
        "likelihood_embedding_without_cpu": 1,
        "high_multilayer_rate": 2,
        "high_anomaly_rate": 3,
        "sparse_extreme_records": 4,
        "ordinary": 5,
    }

    result["_category_order"] = (
        result["candidate_category"]
        .map(category_order)
        .fillna(99)
    )

    result = result.sort_values(
        [
            "_category_order",
            "multi_layer_2plus_rate",
            "p99_anomaly_score",
            "record_count",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    )

    return result.drop(
        columns=["_category_order"]
    )


# ---------------------------------------------------------------------
# Tier 3: extreme individual records
# ---------------------------------------------------------------------

def build_extreme_records(
    df: pd.DataFrame,
    top_n_per_taxon: int,
) -> pd.DataFrame:

    print("Building taxon_extreme_records.csv...")

    columns = [
        "taxon_group",
        "record_id",
        "start",
        "end",
        "sequence_length",
        "gc_content",
        "shannon_entropy",
        "perplexity",
        "embedding_norm",
        "multi_layer_anomaly_score",
        "is_gc_outlier",
        "is_length_outlier",
        "is_likelihood_outlier",
        "is_embedding_outlier",
        "outlier_layer_count",
        "cpu_signal",
        "likelihood_signal",
        "embedding_signal",
        "non_embedding_anomaly",
        "intra_taxon_gc_percentile",
        "intra_taxon_length_percentile",
        "intra_taxon_perplexity_percentile",
        "intra_taxon_embedding_norm_percentile",
        "normalized_argmin_position",
        "distance_to_sequence_boundary",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    result = (
        df.sort_values(
            "multi_layer_anomaly_score",
            ascending=False,
        )
        .groupby(
            "taxon_group",
            sort=False,
            dropna=False,
        )
        .head(top_n_per_taxon)
        [columns]
        .copy()
    )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate taxon-level Phase 4.5 summaries "
            "from record_metrics.csv."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help="Path to record_metrics.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )

    parser.add_argument(
        "--min-records",
        type=int,
        default=30,
        help=(
            "Minimum taxon size for candidate discovery. "
            "taxon_summary.csv still includes all taxa."
        ),
    )

    parser.add_argument(
        "--top-n-per-taxon",
        type=int,
        default=10,
        help=(
            "Number of extreme records retained per taxon."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    df = load_record_metrics(args.input)
    df = normalize_types(df)

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    taxon_summary = build_taxon_summary(df)

    signal_profile = build_signal_profile(df)

    candidates = build_candidates(
        taxon_summary,
        signal_profile,
        min_records=args.min_records,
    )

    extreme_records = build_extreme_records(
        df,
        top_n_per_taxon=args.top_n_per_taxon,
    )

    # -----------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------

    summary_path = (
        args.output_dir / "taxon_summary.csv"
    )

    signal_path = (
        args.output_dir / "taxon_signal_profile.csv"
    )

    candidate_path = (
        args.output_dir / "taxon_candidates.csv"
    )

    extreme_path = (
        args.output_dir / "taxon_extreme_records.csv"
    )

    taxon_summary.to_csv(
        summary_path,
        index=False,
    )

    signal_profile.to_csv(
        signal_path,
        index=False,
    )

    candidates.to_csv(
        candidate_path,
        index=False,
    )

    extreme_records.to_csv(
        extreme_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("TAXON ANALYSIS COMPLETE")
    print("=" * 72)

    print(f"Records:         {len(df):,}")
    print(f"Taxa:            {len(taxon_summary):,}")
    print(
        f"Candidate taxa:  {len(candidates):,}"
    )

    print()
    print("Outputs:")
    print(f"  {summary_path}")
    print(f"  {signal_path}")
    print(f"  {candidate_path}")
    print(f"  {extreme_path}")

    print()
    print("Top 20 taxa by P99 anomaly:")
    print()

    display_columns = [
        "taxon_group",
        "record_count",
        "any_outlier_rate",
        "multi_layer_2plus_rate",
        "multi_layer_3plus_rate",
        "p95_anomaly_score",
        "p99_anomaly_score",
        "max_anomaly_score",
        "dominant_outlier_layer",
    ]

    print(
        taxon_summary[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Candidate categories:")
    print()

    print(
        candidates["candidate_category"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()