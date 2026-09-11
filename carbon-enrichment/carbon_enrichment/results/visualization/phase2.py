#!/usr/bin/env python3
"""
Phase 2 visualization pipeline.

Reads derived Phase 2 CSV metrics and produces publication-quality
visualizations for the enriched genomic corpus.

Input:
    metrics/derived/phase2_csv/

Output:
    metrics/derived/phase2_viz/

Corpus-level distributions:
    sequence_length_distribution.csv
        sequence_length_bucket, n

    gc_content_distribution.csv
        gc_bucket, n

    length_gc_distribution.csv
        length_log_bucket, gc_bucket, n

    taxonomy_depth_distribution.csv
        taxonomy_depth, n

    taxonomy_cardinality.csv
        domain_distinct_count,
        kingdom_distinct_count,
        subkingdom_distinct_count,
        phylum_distinct_count,
        subphylum_distinct_count,
        class_distinct_count,
        subclass_distinct_count,
        order_distinct_count,
        suborder_distinct_count,
        family_distinct_count,
        genus_distinct_count

    taxonomy_composition.csv
        taxonomy_class, n

Biological / validation metrics:
    taxonomy_composition.csv
    gc_skew_vs_coding.csv
    gc_skew_vs_taxonomy_class.csv
    stop_codon_validation.csv
    fickett_proxy_validation.csv
    range_sanity_report.csv

Figure sequence:
    01 Sequence-length distribution
    02 GC-content distribution
    03 Length × GC distribution
    04 Taxonomic depth distribution
    05 Taxonomic cardinality
    06 Taxonomic composition
    07 GC-skew by coding status
    08 GC-skew by taxonomy class
    09 Stop-codon validation
    10 Fickett-proxy validation

QC report:
    11_range_sanity_report.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INPUT_DIR = Path(
    "/teamspace/studios/this_studio/"
    "hf-genomic-dataset-enrichment/"
    "metrics/derived/phase2_csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "/teamspace/studios/this_studio/"
    "hf-genomic-dataset-enrichment/"
    "metrics/derived/phase2_viz"
)

FIG_DPI = 330


plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": FIG_DPI,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# Consistent multi-colour palette for categorical figures.
# The palette is intentionally restrained so the figures remain suitable
# for scientific reports and presentations.
PLOT_COLORS = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
    "#BAB0AC",  # gray
    "#5F9EAD",  # blue-teal
]

# =============================================================================
# Utilities
# =============================================================================

def load_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    """Load a Phase 2 CSV and report its dimensions."""

    path = input_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required metric file does not exist:\n{path}"
        )

    df = pd.read_csv(path)

    # Defensive header cleanup.
    df.columns = [str(c).strip() for c in df.columns]

    print(
        f"Loaded {filename}: "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    return df


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    filename: str,
) -> None:
    """Validate that the expected schema exists."""

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nUnexpected schema in {filename}.\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def _parse_numeric_value(raw) -> float:
    """
    Parse a single scalar to float.

    Handles:
        1185
        "1,185"
        "0,1234"

    The per-value approach prevents large values containing thousands
    separators from silently becoming NaN.
    """

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan

    s = str(raw).strip()

    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    # Plain numeric.
    try:
        return float(s)
    except ValueError:
        pass

    # Thousands separator.
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass

    # European decimal comma.
    try:
        return float(
            s.replace(".", "").replace(",", ".")
        )
    except ValueError:
        pass

    return np.nan


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Robust per-value numeric coercion."""

    return series.apply(_parse_numeric_value)


def natural_sort_key(label: str):
    """
    Sort labels containing numbers numerically rather than lexicographically.
    """

    match = re.search(r"-?\d+(\.\d+)?", str(label))

    if match:
        return (0, float(match.group()))

    return (1, str(label))


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
) -> None:
    """Save a figure at publication resolution."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / filename

    fig.savefig(
        path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {path}")


def annotate_bar_values(
    ax,
    bars,
    values,
    fmt="{:,.0f}",
    fontsize=9,
):
    """Add values above bars."""

    values = list(values)

    max_value = max(
        [abs(float(v)) for v in values if np.isfinite(v)],
        default=0,
    )

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue

        height = bar.get_height()

        offset = max(
            max_value * 0.01,
            0.01,
        )

        ax.annotate(
            fmt.format(value),
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 4 if height >= 0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=fontsize,
        )


# =============================================================================
# Statistical utilities
# =============================================================================

def welch_ttest_from_summary(
    mean_a: float,
    sd_a: float,
    n_a: float,
    mean_b: float,
    sd_b: float,
    n_b: float,
) -> tuple[float, float]:
    """
    Welch's t-test computed from summary statistics.
    """

    se_a = (sd_a ** 2) / n_a
    se_b = (sd_b ** 2) / n_b

    se_diff = np.sqrt(se_a + se_b)

    if se_diff == 0:
        return np.nan, np.nan

    t_stat = (mean_a - mean_b) / se_diff

    df_num = (se_a + se_b) ** 2
    df_den = (
        (se_a ** 2) / (n_a - 1)
        + (se_b ** 2) / (n_b - 1)
    )

    dof = (
        df_num / df_den
        if df_den > 0
        else (n_a + n_b - 2)
    )

    p_value = 2 * stats.t.sf(
        np.abs(t_stat),
        dof,
    )

    return t_stat, p_value


def pooled_cohens_d(
    mean_a: float,
    sd_a: float,
    n_a: float,
    mean_b: float,
    sd_b: float,
    n_b: float,
) -> float:
    """Cohen's d using pooled standard deviation."""

    pooled_var = (
        ((n_a - 1) * sd_a ** 2
         + (n_b - 1) * sd_b ** 2)
        / (n_a + n_b - 2)
    )

    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return np.nan

    return (mean_a - mean_b) / pooled_sd


def plot_mean_ci_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    value_col: str,
    stddev_col: str,
    filename: str,
    title: str,
    ylabel: str,
    source_name: str,
    zero_reference_line: bool = False,
) -> None:
    """
    Plot mean ± 95% CI.

    For exactly two groups, annotate Welch's t-test and Cohen's d.
    """

    require_columns(
        df,
        [
            "group",
            value_col,
            stddev_col,
            "n",
        ],
        source_name,
    )

    plot_df = df.copy()

    for column in [
        value_col,
        stddev_col,
        "n",
    ]:
        plot_df[column] = coerce_numeric(
            plot_df[column]
        )

    plot_df = plot_df.dropna(
        subset=[
            value_col,
            stddev_col,
            "n",
        ]
    )

    plot_df = plot_df[
        plot_df["n"] > 0
    ]

    if plot_df.empty:
        raise ValueError(
            f"No usable rows in {source_name}."
        )

    plot_df = (
        plot_df
        .sort_values(value_col)
        .reset_index(drop=True)
    )

    sem = (
        plot_df[stddev_col]
        / np.sqrt(plot_df["n"])
    )

    ci95 = 1.96 * sem

    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(
        figsize=(9, 6.5)
    )

    bars = ax.bar(
        x,
        plot_df[value_col],
        yerr=ci95,
        capsize=5,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.6,
        color=PLOT_COLORS[:len(plot_df)],
    )

    if zero_reference_line:
        ax.axhline(
            0,
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

    for bar, value, n in zip(
        bars,
        plot_df[value_col],
        plot_df["n"],
    ):
        ax.annotate(
            f"{value:.4g}\n(n={int(n):,})",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df["group"].astype(str)
    )

    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    subtitle = ""

    if len(plot_df) == 2:
        a, b = (
            plot_df.iloc[0],
            plot_df.iloc[1],
        )

        t_stat, p_value = (
            welch_ttest_from_summary(
                a[value_col],
                a[stddev_col],
                a["n"],
                b[value_col],
                b[stddev_col],
                b["n"],
            )
        )

        d = pooled_cohens_d(
            a[value_col],
            a[stddev_col],
            a["n"],
            b[value_col],
            b[stddev_col],
            b["n"],
        )

        if np.isfinite(p_value):
            sig = (
                "significant"
                if p_value < 0.05
                else "not significant"
            )

            subtitle = (
                f"Welch's t-test: "
                f"t={t_stat:.2f}, "
                f"p={p_value:.3g} "
                f"({sig}) | "
                f"Cohen's d={d:.2f}"
            )

    ax.set_title(
        title
        + (
            f"\n{subtitle}"
            if subtitle
            else ""
        ),
        fontsize=12,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        filename,
    )


# =============================================================================
# 1. Sequence-length distribution
# =============================================================================

def plot_sequence_length_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot sequence-length distribution.

    Source:
        sequence_length_distribution.csv

    Schema:
        length_bucket, n
    """

    source_name = "sequence_length_distribution.csv"

    require_columns(
        df,
        [
            "length_bucket",
            "n",
        ],
        source_name,
    )

    plot_df = df.copy()

    plot_df["n"] = coerce_numeric(plot_df["n"])

    plot_df = plot_df.dropna(subset=["n"])

    plot_df["length_bucket"] = (
        plot_df["length_bucket"].astype(str)
    )

    plot_df = plot_df.sort_values(
        "length_bucket",
        key=lambda s: s.map(natural_sort_key),
    )

    if plot_df.empty:
        raise ValueError(
            f"{source_name} contains no usable rows."
        )

    total = plot_df["n"].sum()

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    x = np.arange(len(plot_df))

    bars = ax.bar(
        x,
        plot_df["n"],
        color=PLOT_COLORS[:len(plot_df)],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.5,
    )

    annotate_bar_values(
        ax,
        bars,
        plot_df["n"],
        fmt="{:,.0f}",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df["length_bucket"],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Sequence-length bucket"
    )

    ax.set_ylabel(
        "Number of sequences"
    )

    ax.set_title(
        "Sequence-length distribution of the enriched corpus"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    if total > 0:
        ax.text(
            0.99,
            0.98,
            f"Total sequences: {total:,.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "01_sequence_length_distribution.png",
    )

# =============================================================================
# 2. GC-content distribution
# =============================================================================

def plot_gc_content_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot GC-content distribution.

    Source:
        gc_content_distribution.csv

    Schema:
        gc_bucket, n
    """

    source_name = (
        "gc_content_distribution.csv"
    )

    require_columns(
        df,
        [
            "gc_bucket",
            "n",
        ],
        source_name,
    )

    plot_df = df.copy()

    plot_df["n"] = coerce_numeric(
        plot_df["n"]
    )

    plot_df = plot_df.dropna(
        subset=["n"]
    )

    plot_df["gc_bucket"] = (
        plot_df["gc_bucket"].astype(str)
    )

    plot_df = plot_df.sort_values(
        "gc_bucket",
        key=lambda s: s.map(
            natural_sort_key
        ),
    )

    if plot_df.empty:
        raise ValueError(
            f"{source_name} contains no usable rows."
        )

    total = plot_df["n"].sum()

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    x = np.arange(len(plot_df))

    bars = ax.bar(
        x,
        plot_df["n"],
        color=PLOT_COLORS[:len(plot_df)],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.5,
    )

    annotate_bar_values(
        ax,
        bars,
        plot_df["n"],
        fmt="{:,.0f}",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df["gc_bucket"],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "GC-content bucket"
    )

    ax.set_ylabel(
        "Number of sequences"
    )

    ax.set_title(
        "GC-content distribution of the enriched corpus"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    if total > 0:
        ax.text(
            0.99,
            0.98,
            f"Total sequences: {total:,.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "02_gc_content_distribution.png",
    )


# =============================================================================
# 3. Length × GC distribution
# =============================================================================

def plot_length_gc_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot the aggregated sequence-length × GC-content distribution.

    Source:
        length_gc_distribution.csv

    Schema:
        length_log_bucket, gc_bucket, n
    """

    source_name = (
        "length_gc_distribution.csv"
    )

    require_columns(
        df,
        [
            "length_log_bucket",
            "gc_bucket",
            "n",
        ],
        source_name,
    )

    plot_df = df.copy()

    plot_df["n"] = coerce_numeric(
        plot_df["n"]
    )

    plot_df = plot_df.dropna(
        subset=["n"]
    )

    plot_df[
        "length_log_bucket"
    ] = plot_df[
        "length_log_bucket"
    ].astype(str)

    plot_df["gc_bucket"] = (
        plot_df["gc_bucket"].astype(str)
    )

    matrix = plot_df.pivot_table(
        index="gc_bucket",
        columns="length_log_bucket",
        values="n",
        aggfunc="sum",
        fill_value=0,
    )

    matrix = matrix.reindex(
        index=sorted(
            matrix.index,
            key=natural_sort_key,
        ),
        columns=sorted(
            matrix.columns,
            key=natural_sort_key,
        ),
    )

    if matrix.empty:
        raise ValueError(
            f"{source_name} contains no usable data."
        )

    fig, ax = plt.subplots(
        figsize=(12, 7.5)
    )

    values = matrix.to_numpy(
        dtype=float
    )

    log_values = np.log10(
        values + 1
    )

    image = ax.imshow(
        log_values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="viridis",
    )

    ax.set_xlabel(
        "Sequence-length bucket"
    )

    ax.set_ylabel(
        "GC-content bucket"
    )

    ax.set_title(
        "Broad sequence-length and GC-content coverage"
    )

    ax.set_xticks(
        np.arange(
            len(matrix.columns)
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        matrix.index
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "log10(sequence count + 1)"
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "03_length_gc_distribution.png",
    )


# =============================================================================
# 4. Taxonomic depth distribution
# =============================================================================

def plot_taxonomy_depth_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot the distribution of taxonomic annotation depth.

    Source:
        taxonomy_depth_distribution.csv

    Schema:
        taxonomy_depth, n

    Important:
        Taxonomy depth is treated as an annotation-resolution measure.
        It is not relabeled as a biological taxonomic rank.
    """

    source_name = (
        "taxonomy_depth_distribution.csv"
    )

    require_columns(
        df,
        [
            "taxonomy_depth",
            "n",
        ],
        source_name,
    )

    plot_df = df.copy()

    plot_df["taxonomy_depth"] = (
        coerce_numeric(
            plot_df["taxonomy_depth"]
        )
    )

    plot_df["n"] = coerce_numeric(
        plot_df["n"]
    )

    plot_df = plot_df.dropna(
        subset=[
            "taxonomy_depth",
            "n",
        ]
    )

    plot_df = plot_df.sort_values(
        "taxonomy_depth"
    )

    if plot_df.empty:
        raise ValueError(
            f"{source_name} contains no usable rows."
        )

    total = plot_df["n"].sum()

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    bars = ax.bar(
        plot_df["taxonomy_depth"],
        plot_df["n"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    # Label only sufficiently large bars to prevent
    # the figure from becoming unreadable.
    max_n = plot_df["n"].max()

    for bar, value in zip(
        bars,
        plot_df["n"],
    ):
        if value >= max_n * 0.02:
            ax.annotate(
                f"{int(value):,}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    value,
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel(
        "Taxonomic annotation depth"
    )

    ax.set_ylabel(
        "Number of sequences"
    )

    ax.set_title(
        "Distribution of taxonomic annotation depth"
    )

    ax.set_xticks(
        plot_df["taxonomy_depth"]
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    if total > 0:
        ax.text(
            0.99,
            0.98,
            f"Total sequences: {total:,.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "04_taxonomy_depth_distribution.png",
    )


# =============================================================================
# 5. Taxonomic cardinality
# =============================================================================

def plot_taxonomy_cardinality(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot the number of distinct taxa represented at each hierarchical rank.

    Source:
        taxonomy_cardinality.csv
    """

    expected_columns = [
        "domain_distinct_count",
        "kingdom_distinct_count",
        "subkingdom_distinct_count",
        "phylum_distinct_count",
        "subphylum_distinct_count",
        "class_distinct_count",
        "subclass_distinct_count",
        "order_distinct_count",
        "suborder_distinct_count",
        "family_distinct_count",
        "genus_distinct_count",
    ]

    require_columns(
        df,
        expected_columns,
        "taxonomy_cardinality.csv",
    )

    if df.empty:
        raise ValueError(
            "taxonomy_cardinality.csv contains no rows."
        )

    row = df.iloc[0]

    ranks = [
        "Domain",
        "Kingdom",
        "Subkingdom",
        "Phylum",
        "Subphylum",
        "Class",
        "Subclass",
        "Order",
        "Suborder",
        "Family",
        "Genus",
    ]

    values = [
        row[column]
        for column in expected_columns
    ]

    values = coerce_numeric(
        pd.Series(values)
    )

    taxonomy_df = pd.DataFrame(
        {
            "rank": ranks,
            "count": values,
        }
    )

    dropped = taxonomy_df[
        taxonomy_df["count"].isna()
    ]["rank"].tolist()

    if dropped:
        print(
            "Warning: taxonomy_cardinality.csv -- "
            f"could not parse rank(s): {dropped}"
        )

    taxonomy_df = taxonomy_df.dropna()

    if taxonomy_df.empty:
        raise ValueError(
            "No usable taxonomy cardinality values."
        )

    fig, ax = plt.subplots(
        figsize=(12, 7.5)
    )

    x = np.arange(
        len(taxonomy_df)
    )

    bars = ax.bar(
        x,
        taxonomy_df["count"],
        color=PLOT_COLORS[:len(taxonomy_df)],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.5,
    )

    annotate_bar_values(
        ax,
        bars,
        taxonomy_df["count"],
        fmt="{:,.0f}",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        taxonomy_df["rank"],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Taxonomic rank"
    )

    ax.set_ylabel(
        "Number of distinct taxa"
    )

    ax.set_title(
        "Distinct taxa represented across hierarchical ranks"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "05_taxonomy_cardinality.png",
    )


# =============================================================================
# 6. Taxonomic composition
# =============================================================================

def plot_taxonomy_composition(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Plot the dominant taxonomy classes by record count.

    Source:
        taxonomy_composition.csv

    Expected schema:
        taxonomy_class, n

    If the source instead uses `group`, that column is accepted
    as a defensive fallback.
    """

    source_name = (
        "taxonomy_composition.csv"
    )

    if "taxonomy_class" in df.columns:
        group_col = "taxonomy_class"
    elif "group" in df.columns:
        group_col = "group"
    else:
        raise ValueError(
            f"\nUnexpected schema in {source_name}.\n"
            "Expected 'taxonomy_class' (or fallback 'group').\n"
            f"Available columns: {list(df.columns)}"
        )

    require_columns(
        df,
        [group_col, "n"],
        source_name,
    )

    plot_df = df.copy()

    plot_df["n"] = coerce_numeric(
        plot_df["n"]
    )

    plot_df = plot_df.dropna(
        subset=["n"]
    )

    plot_df[group_col] = (
        plot_df[group_col].astype(str)
    )

    # Combine duplicate labels defensively.
    plot_df = (
        plot_df
        .groupby(group_col, as_index=False)["n"]
        .sum()
    )

    total = plot_df["n"].sum()

    plot_df = (
        plot_df
        .sort_values("n", ascending=False)
        .head(top_n)
        .sort_values("n", ascending=True)
    )

    if plot_df.empty:
        raise ValueError(
            f"{source_name} contains no usable rows."
        )

    fig_height = max(
        6,
        0.35 * len(plot_df) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )

    y = np.arange(
        len(plot_df)
    )

    bars = ax.barh(
        y,
        plot_df["n"],
        color=PLOT_COLORS[:len(plot_df)],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        plot_df[group_col]
    )

    ax.set_xlabel(
        "Number of sequences"
    )

    ax.set_ylabel(
        "Taxonomy class"
    )

    ax.set_title(
        f"Taxonomic composition of the enriched corpus\n"
        f"Top {len(plot_df)} taxonomic groups by sequence count"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # Add count + percentage.
    for bar, value in zip(
        bars,
        plot_df["n"],
    ):
        percentage = (
            100 * value / total
            if total > 0
            else 0
        )

        ax.annotate(
            f"{int(value):,} ({percentage:.1f}%)",
            xy=(
                bar.get_width(),
                bar.get_y()
                + bar.get_height() / 2,
            ),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "06_taxonomy_composition.png",
    )


# =============================================================================
# 7. GC-skew by coding status
# =============================================================================

def plot_gc_skew_coding(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Compare mean GC skew between coding-status groups."""

    plot_mean_ci_comparison(
        df,
        output_dir,
        value_col="mean_gc_skew",
        stddev_col="mean_gc_skew_stddev",
        filename="07_gc_skew_coding.png",
        title="GC-skew difference by coding status",
        ylabel="Mean GC skew (95% CI)",
        source_name="gc_skew_vs_coding.csv",
        zero_reference_line=True,
    )


# =============================================================================
# 8. GC-skew by taxonomy class
# =============================================================================

def plot_gc_skew_taxonomy(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Plot mean GC skew by taxonomy class.

    The largest classes by sample size are retained.
    """

    source_name = (
        "gc_skew_vs_taxonomy_class.csv"
    )

    require_columns(
        df,
        [
            "group",
            "mean_gc_skew",
            "mean_gc_skew_stddev",
            "n",
        ],
        source_name,
    )

    plot_df = df.copy()

    for column in [
        "mean_gc_skew",
        "mean_gc_skew_stddev",
        "n",
    ]:
        plot_df[column] = coerce_numeric(
            plot_df[column]
        )

    plot_df = plot_df.dropna(
        subset=[
            "mean_gc_skew",
            "mean_gc_skew_stddev",
            "n",
        ]
    )

    plot_df = (
        plot_df
        .sort_values(
            "n",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "mean_gc_skew",
            ascending=True,
        )
    )

    if plot_df.empty:
        raise ValueError(
            f"{source_name} contains no usable rows."
        )

    y = np.arange(
        len(plot_df)
    )

    fig_height = max(
        6,
        0.35 * len(plot_df) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )

    ax.errorbar(
        plot_df["mean_gc_skew"],
        y,
        xerr=plot_df[
            "mean_gc_skew_stddev"
        ],
        fmt="o",
        capsize=3,
        linewidth=1.5,
        color=PLOT_COLORS[0],
        markerfacecolor=PLOT_COLORS[1],
        markeredgecolor="black",
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        plot_df["group"].astype(str)
    )

    ax.set_xlabel(
        "Mean GC skew ± SD"
    )

    ax.set_ylabel(
        "Taxonomy class"
    )

    ax.set_title(
        "GC skew across taxonomic classes\n"
        f"Top {len(plot_df)} classes by sample size"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "08_gc_skew_taxonomy_class.png",
    )


# =============================================================================
# 9. Stop-codon validation
# =============================================================================

def plot_stop_codon_validation(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot mean stop-codon counts across reading frames."""

    source_name = (
        "stop_codon_validation.csv"
    )

    require_columns(
        df,
        [
            "group",
            "mean_stops_frame0",
            "mean_stops_frame1",
            "mean_stops_frame2",
            "n",
        ],
        source_name,
    )

    plot_df = df.copy()

    frame_cols = [
        "mean_stops_frame0",
        "mean_stops_frame1",
        "mean_stops_frame2",
    ]

    for column in frame_cols + ["n"]:
        plot_df[column] = coerce_numeric(
            plot_df[column]
        )

    n_before = len(plot_df)

    plot_df = plot_df.dropna(
        subset=frame_cols
    )

    n_after = len(plot_df)

    if n_after == 0:
        raise ValueError(
            f"{source_name}: all rows were dropped after "
            "numeric coercion."
        )

    if n_after < n_before:
        print(
            f"Warning: {source_name} dropped "
            f"{n_before - n_after} row(s)."
        )

    x = np.arange(
        len(plot_df)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    bars0 = ax.bar(
        x - width,
        plot_df["mean_stops_frame0"],
        width,
        label="Frame 0",
        color=PLOT_COLORS[0],
    )

    bars1 = ax.bar(
        x,
        plot_df["mean_stops_frame1"],
        width,
        label="Frame 1",
        color=PLOT_COLORS[1],
    )

    bars2 = ax.bar(
        x + width,
        plot_df["mean_stops_frame2"],
        width,
        label="Frame 2",
        color=PLOT_COLORS[2],
    )

    max_val = plot_df[
        frame_cols
    ].to_numpy().max()

    if np.isfinite(max_val) and max_val > 0:
        ax.set_ylim(
            0,
            max_val * 1.2,
        )

    for bar_group in (
        bars0,
        bars1,
        bars2,
    ):
        for bar in bar_group:
            height = bar.get_height()

            ax.annotate(
                f"{height:.3g}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    height,
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df["group"].astype(str)
    )

    ax.set_xlabel("Group")

    ax.set_ylabel(
        "Mean stop-codon count"
    )

    ax.set_title(
        "Stop-codon distribution across reading frames"
    )

    ax.legend(frameon=False)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "09_stop_codon_validation.png",
    )


# =============================================================================
# 10. Fickett proxy validation
# =============================================================================

def plot_fickett_validation(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Compare mean Fickett-proxy variance between groups."""

    plot_mean_ci_comparison(
        df,
        output_dir,
        value_col="mean_variance",
        stddev_col="mean_variance_stddev",
        filename="10_fickett_proxy_validation.png",
        title="Fickett-proxy variance by group",
        ylabel="Mean variance (95% CI)",
        source_name="fickett_proxy_validation.csv",
        zero_reference_line=False,
    )


# =============================================================================
# 11. Range sanity report
# =============================================================================

def write_range_sanity_summary(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Write the range-sanity report as a compact text summary.

    This is deliberately a QC report rather than a conventional figure.
    """

    source_name = (
        "range_sanity_report.csv"
    )

    require_columns(
        df,
        [
            "homopolymer_exceeds_length",
            "gc_content_out_of_range",
            "tm_estimate_implausible",
            "cpg_odds_negative",
            "total_rows",
        ],
        source_name,
    )

    if df.empty:
        raise ValueError(
            f"{source_name} contains no rows."
        )

    row = df.iloc[0]

    checks = {
        "Homopolymer exceeds sequence length":
            row["homopolymer_exceeds_length"],

        "GC content outside [0, 1]":
            row["gc_content_out_of_range"],

        "Implausible Tm estimate":
            row["tm_estimate_implausible"],

        "Negative CpG odds":
            row["cpg_odds_negative"],
    }

    total_rows = coerce_numeric(
        pd.Series([row["total_rows"]])
    ).iloc[0]

    lines = [
        "PHASE 2 RANGE SANITY REPORT",
        "=" * 60,
        "",
        f"Total rows checked: {int(total_rows):,}",
        "",
    ]

    for name, value in checks.items():
        count = _parse_numeric_value(value)

        if np.isfinite(count):
            count = int(count)

            if total_rows > 0:
                rate = (
                    100.0
                    * count
                    / total_rows
                )

                lines.append(
                    f"{name}: "
                    f"{count:,} "
                    f"({rate:.6f}%)"
                )
            else:
                lines.append(
                    f"{name}: {count:,}"
                )
        else:
            lines.append(
                f"{name}: {value}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / "11_range_sanity_report.txt"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 2 visualizations "
            "from derived genomic metrics."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Phase 2 CSV directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Visualization output directory.",
    )

    parser.add_argument(
        "--taxonomy-top-n",
        type=int,
        default=20,
        help=(
            "Number of taxonomy classes to show "
            "in taxonomy composition and GC-skew plots."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("PHASE 2 VISUALIZATION PIPELINE")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Load corpus-level metrics
    # -------------------------------------------------------------------------

    sequence_length = load_csv(
        args.input_dir,
        "sequence_length_distribution.csv",
    )

    gc_content = load_csv(
        args.input_dir,
        "gc_content_distribution.csv",
    )

    length_gc = load_csv(
        args.input_dir,
        "length_gc_distribution.csv",
    )

    taxonomy_depth = load_csv(
        args.input_dir,
        "taxonomy_depth_distribution.csv",
    )

    taxonomy_cardinality = load_csv(
        args.input_dir,
        "taxonomy_cardinality.csv",
    )

    taxonomy_composition = load_csv(
        args.input_dir,
        "taxonomy_composition.csv",
    )

    # -------------------------------------------------------------------------
    # Load biological / validation metrics
    # -------------------------------------------------------------------------

    gc_skew_coding = load_csv(
        args.input_dir,
        "gc_skew_vs_coding.csv",
    )

    gc_skew_taxonomy = load_csv(
        args.input_dir,
        "gc_skew_vs_taxonomy_class.csv",
    )

    stop_codons = load_csv(
        args.input_dir,
        "stop_codon_validation.csv",
    )

    fickett = load_csv(
        args.input_dir,
        "fickett_proxy_validation.csv",
    )

    range_sanity = load_csv(
        args.input_dir,
        "range_sanity_report.csv",
    )

    print()
    print("Generating visualizations...")
    print()

    # -------------------------------------------------------------------------
    # Corpus structure
    # -------------------------------------------------------------------------

    plot_sequence_length_distribution(
        sequence_length,
        args.output_dir,
    )

    plot_gc_content_distribution(
        gc_content,
        args.output_dir,
    )

    plot_length_gc_distribution(
        length_gc,
        args.output_dir,
    )

    plot_taxonomy_depth_distribution(
        taxonomy_depth,
        args.output_dir,
    )

    plot_taxonomy_cardinality(
        taxonomy_cardinality,
        args.output_dir,
    )

    plot_taxonomy_composition(
        taxonomy_composition,
        args.output_dir,
        top_n=args.taxonomy_top_n,
    )

    # -------------------------------------------------------------------------
    # Biological relationships
    # -------------------------------------------------------------------------

    plot_gc_skew_coding(
        gc_skew_coding,
        args.output_dir,
    )

    plot_gc_skew_taxonomy(
        gc_skew_taxonomy,
        args.output_dir,
        top_n=args.taxonomy_top_n,
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    plot_stop_codon_validation(
        stop_codons,
        args.output_dir,
    )

    plot_fickett_validation(
        fickett,
        args.output_dir,
    )

    write_range_sanity_summary(
        range_sanity,
        args.output_dir,
    )

    print()
    print("=" * 80)
    print("PHASE 2 VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(
        f"Output directory: {args.output_dir}"
    )


if __name__ == "__main__":
    main()