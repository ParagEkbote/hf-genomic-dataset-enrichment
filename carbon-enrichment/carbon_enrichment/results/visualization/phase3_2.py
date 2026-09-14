#!/usr/bin/env python3
"""
Level 3 — KNN biological analysis for Carbon-3B embeddings.

This is a standalone visualization/analysis script.
It does NOT require Dagster.

Inputs
------
1. KNN edge table:
   metrics/derived/phase4/knn_neighbors.csv

2. CPU-enriched sequence metadata:
   HuggingFaceBio/AINovice2005 carbon-cpu-enriched-sequences

Processing
----------
- Converts the KNN CSV to temporary Parquet because ClickHouseResource's
  register_dataset() expects Parquet.
- Registers the KNN Parquet locally with ClickHouse.
- Registers the CPU-enriched HF dataset directly as remote Parquet shards.
- Retrieves metadata only for sequence spans occurring in the KNN graph.
- Performs the metadata joins and biological-property calculations in
  ClickHouse.
- Writes the original four derived CSV files plus four biological-analysis
  summary CSVs.
- Generates six publication-oriented PNG figures.

Important
---------
The KNN file contains exact query/neighbor identity:
(record_id, start, end), so the embeddings dataset is NOT required here.

The taxonomy column is treated as an ordered semicolon-delimited lineage.
Because the public CPU-enriched dataset does not provide explicit rank labels
for every taxonomy component, this script reports shared taxonomy depth and
the deepest shared taxon rather than falsely assigning fixed ranks such as
"genus" or "family" by position.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from carbon_enrichment.resources.clickhouse import (
    ClickHouseConfig,
    ClickHouseResource,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CPU_HF_GLOB = (
    "https://huggingface.co/datasets/"
    "AINovice2005/carbon-cpu-enriched-sequences/resolve/main/*.parquet"
)

DEFAULT_KNN = (
    "/teamspace/studios/this_studio/"
    "hf-genomic-dataset-enrichment/"
    "metrics/derived/phase4/knn_neighbors.csv"
)

DEFAULT_OUTPUT = (
    "/teamspace/studios/this_studio/"
    "hf-genomic-dataset-enrichment/"
    "metrics/derived/phase4/level3"
)

REQUIRED_KNN_COLUMNS = [
    "query_row_index",
    "query_record_id",
    "query_start",
    "query_end",
    "neighbor_rank",
    "neighbor_row_index",
    "neighbor_record_id",
    "neighbor_start",
    "neighbor_end",
    "cosine_similarity",
    "cosine_distance",
]

REQUIRED_CPU_COLUMNS = [
    "record_id",
    "start",
    "end",
    "taxonomy",
    "gc_content",
    "gc_skew",
    "sequence_length",
]

LOGGER = logging.getLogger("level3_knn_analysis")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# KNN conversion
# ---------------------------------------------------------------------------


def convert_knn_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
) -> None:
    """
    Convert the local KNN CSV to Parquet.

    The entire CPU dataset is never loaded here. Only the KNN edge table is
    converted.
    """
    LOGGER.info("Reading KNN CSV: %s", csv_path)

    table = pacsv.read_csv(
        str(csv_path),
        read_options=pacsv.ReadOptions(
            use_threads=True,
            block_size=64 * 1024 * 1024,
        ),
    )

    columns = set(table.column_names)
    missing = [c for c in REQUIRED_KNN_COLUMNS if c not in columns]

    if missing:
        raise ValueError("KNN file is missing required columns: " + ", ".join(missing))

    LOGGER.info(
        "KNN rows: %s | columns: %s",
        f"{table.num_rows:,}",
        table.num_columns,
    )

    pq.write_table(
        table,
        str(parquet_path),
        compression="zstd",
        use_dictionary=True,
    )

    LOGGER.info("Temporary KNN Parquet written: %s", parquet_path)


# ---------------------------------------------------------------------------
# ClickHouse query
# ---------------------------------------------------------------------------


def build_analysis_query(
    knn_source: str,
    cpu_source: str,
) -> str:
    """
    Build a ClickHouse query that:

    1. obtains all sequence identities occurring in the KNN graph;
    2. retrieves only their CPU-enrichment metadata;
    3. joins metadata onto query and neighbor sides;
    4. calculates biological-property deltas;
    5. calculates taxonomy agreement metrics.
    """

    return f"""
WITH
requested AS
(
    SELECT
        query_record_id AS record_id,
        query_start AS start,
        query_end AS end
    FROM {knn_source}

    UNION DISTINCT

    SELECT
        neighbor_record_id AS record_id,
        neighbor_start AS start,
        neighbor_end AS end
    FROM {knn_source}
),

cpu_meta AS
(
    SELECT
        c.record_id,
        c.start,
        c.end,

        any(c.taxonomy) AS taxonomy,
        any(c.gc_content) AS gc_content,
        any(c.gc_skew) AS gc_skew,
        any(c.sequence_length) AS sequence_length,

        count() AS metadata_rows

    FROM {cpu_source} AS c

    INNER JOIN requested AS r
        ON c.record_id = r.record_id
       AND c.start = r.start
       AND c.end = r.end

    GROUP BY
        c.record_id,
        c.start,
        c.end
),

base AS
(
    SELECT
        k.query_row_index,
        k.query_record_id,
        k.query_start,
        k.query_end,

        k.neighbor_rank,
        k.neighbor_row_index,
        k.neighbor_record_id,
        k.neighbor_start,
        k.neighbor_end,

        k.cosine_similarity,
        k.cosine_distance,

        q.taxonomy AS query_taxonomy,
        n.taxonomy AS neighbor_taxonomy,

        q.gc_content AS query_gc,
        n.gc_content AS neighbor_gc,

        q.gc_skew AS query_gc_skew,
        n.gc_skew AS neighbor_gc_skew,

        q.sequence_length AS query_sequence_length,
        n.sequence_length AS neighbor_sequence_length,

        q.metadata_rows AS query_metadata_rows,
        n.metadata_rows AS neighbor_metadata_rows

    FROM {knn_source} AS k

    LEFT JOIN cpu_meta AS q
        ON k.query_record_id = q.record_id
       AND k.query_start = q.start
       AND k.query_end = q.end

    LEFT JOIN cpu_meta AS n
        ON k.neighbor_record_id = n.record_id
       AND k.neighbor_start = n.start
       AND k.neighbor_end = n.end
),

with_taxonomy AS
(
    SELECT
        *,

        if(
            query_taxonomy IS NULL
            OR neighbor_taxonomy IS NULL,
            0,
            arrayCount(
                x -> x,
                arrayMap(
                    i ->
                        i <= least(
                            length(splitByChar(';', query_taxonomy)),
                            length(splitByChar(';', neighbor_taxonomy))
                        )
                        AND
                        splitByChar(';', query_taxonomy)[i]
                        =
                        splitByChar(';', neighbor_taxonomy)[i],
                    range(
                        1,
                        least(
                            length(splitByChar(';', query_taxonomy)),
                            length(splitByChar(';', neighbor_taxonomy))
                        ) + 1
                    )
                )
            )
        ) AS shared_taxonomic_depth,

        arrayStringConcat(
            arraySlice(
                splitByChar(';', query_taxonomy),
                1,
                if(
                    query_taxonomy IS NULL
                    OR neighbor_taxonomy IS NULL,
                    0,
                    arrayCount(
                        x -> x,
                        arrayMap(
                            i ->
                                i <= least(
                                    length(splitByChar(';', query_taxonomy)),
                                    length(splitByChar(';', neighbor_taxonomy))
                                )
                                AND
                                splitByChar(';', query_taxonomy)[i]
                                =
                                splitByChar(';', neighbor_taxonomy)[i],
                            range(
                                1,
                                least(
                                    length(splitByChar(';', query_taxonomy)),
                                    length(splitByChar(';', neighbor_taxonomy))
                                ) + 1
                            )
                        )
                    )
                )
            ),
            ';'
        ) AS shared_taxonomy_prefix

    FROM base
)

SELECT
    *,

    if(
        query_gc IS NULL OR neighbor_gc IS NULL,
        NULL,
        abs(query_gc - neighbor_gc)
    ) AS delta_gc,

    if(
        query_sequence_length IS NULL
        OR neighbor_sequence_length IS NULL,
        NULL,
        abs(
            log1p(toFloat64(query_sequence_length))
            -
            log1p(toFloat64(neighbor_sequence_length))
        )
    ) AS delta_log_length,

    if(
        query_gc_skew IS NULL OR neighbor_gc_skew IS NULL,
        NULL,
        abs(query_gc_skew - neighbor_gc_skew)
    ) AS delta_gc_skew,

    if(
        query_taxonomy IS NULL
        OR neighbor_taxonomy IS NULL,
        'taxonomy_missing',

        if(
            query_taxonomy = neighbor_taxonomy,
            'same_terminal_taxonomy',

            if(
                shared_taxonomic_depth = 0,
                'no_shared_taxonomic_prefix',

                concat(
                    'shared_prefix_depth_',
                    toString(shared_taxonomic_depth)
                )
            )
        )
    ) AS taxonomic_relationship,

    if(
        query_taxonomy IS NULL
        OR neighbor_taxonomy IS NULL,
        NULL,

        query_taxonomy = neighbor_taxonomy
    ) AS same_terminal_taxonomy

FROM with_taxonomy
ORDER BY
    query_row_index,
    neighbor_rank
"""


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def load_knn_metadata(
    clickhouse: ClickHouseResource,
) -> pd.DataFrame:
    knn_source = clickhouse.source_expr("knn")
    cpu_source = clickhouse.source_expr("cpu")

    sql = build_analysis_query(
        knn_source=knn_source,
        cpu_source=cpu_source,
    )

    LOGGER.info("Executing Level 3 ClickHouse analysis")

    table = clickhouse.query_arrow(sql)

    LOGGER.info(
        "ClickHouse returned %s rows and %s columns",
        f"{table.num_rows:,}",
        table.num_columns,
    )

    return table.to_pandas()


# ---------------------------------------------------------------------------
# Derived CSV 1
# ---------------------------------------------------------------------------


def make_similarity_by_rank(
    df: pd.DataFrame,
    output: Path,
) -> None:

    result = (
        df.dropna(subset=["cosine_similarity"])
        .groupby("neighbor_rank")["cosine_similarity"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )

    result.to_csv(
        output / "knn_similarity_by_rank.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Derived CSV 2
# ---------------------------------------------------------------------------


def make_taxonomic_consistency(
    df: pd.DataFrame,
    output: Path,
) -> None:

    valid = df.dropna(subset=["query_taxonomy", "neighbor_taxonomy"]).copy()

    rows = []

    for k in (1, 5, 10):
        subset = valid[valid["neighbor_rank"] <= k]

        if subset.empty:
            continue

        grouped = subset.groupby(
            [
                "query_record_id",
                "query_start",
                "query_end",
            ]
        )

        query_count = grouped.ngroups

        same_terminal = grouped["same_terminal_taxonomy"].max().mean()

        depth1 = (grouped["shared_taxonomic_depth"].max() >= 1).mean()

        depth3 = (grouped["shared_taxonomic_depth"].max() >= 3).mean()

        depth5 = (grouped["shared_taxonomic_depth"].max() >= 5).mean()

        depth8 = (grouped["shared_taxonomic_depth"].max() >= 8).mean()

        depth10 = (grouped["shared_taxonomic_depth"].max() >= 10).mean()

        rows.append(
            {
                "k": k,
                "query_count": query_count,
                "same_terminal_taxonomy_fraction": same_terminal,
                "shared_depth_ge_1_fraction": depth1,
                "shared_depth_ge_3_fraction": depth3,
                "shared_depth_ge_5_fraction": depth5,
                "shared_depth_ge_8_fraction": depth8,
                "shared_depth_ge_10_fraction": depth10,
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        output / "knn_taxonomic_consistency.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Derived CSV 3
# ---------------------------------------------------------------------------


def make_taxonomic_relationships(
    df: pd.DataFrame,
    output: Path,
) -> None:

    columns = [
        "query_record_id",
        "query_start",
        "query_end",
        "neighbor_rank",
        "neighbor_record_id",
        "neighbor_start",
        "neighbor_end",
        "cosine_similarity",
        "cosine_distance",
        "query_taxonomy",
        "neighbor_taxonomy",
        "shared_taxonomic_depth",
        "shared_taxonomy_prefix",
        "taxonomic_relationship",
        "same_terminal_taxonomy",
    ]

    result = df[columns].copy()

    result.to_csv(
        output / "knn_taxonomic_relationships.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Derived CSV 4
# ---------------------------------------------------------------------------


def make_sequence_properties(
    df: pd.DataFrame,
    output: Path,
) -> None:

    columns = [
        "query_record_id",
        "query_start",
        "query_end",
        "neighbor_rank",
        "neighbor_record_id",
        "neighbor_start",
        "neighbor_end",
        "cosine_similarity",
        "query_gc",
        "neighbor_gc",
        "delta_gc",
        "query_gc_skew",
        "neighbor_gc_skew",
        "delta_gc_skew",
        "query_sequence_length",
        "neighbor_sequence_length",
        "delta_log_length",
        "taxonomic_relationship",
        "shared_taxonomic_depth",
    ]

    result = df[columns].copy()

    result.to_csv(
        output / "knn_sequence_properties.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Additional biological-analysis CSVs
# ---------------------------------------------------------------------------


def make_taxonomy_depth_similarity_summary(
    df: pd.DataFrame,
    output: Path,
) -> None:
    """
    Summarize embedding similarity as a function of shared taxonomy depth.

    This is the main quantitative table behind the hierarchical-taxonomy
    interpretation of the KNN graph.
    """
    valid = df.dropna(subset=["cosine_similarity", "shared_taxonomic_depth"]).copy()

    valid["shared_taxonomic_depth"] = pd.to_numeric(
        valid["shared_taxonomic_depth"],
        errors="coerce",
    )
    valid = valid.dropna(subset=["shared_taxonomic_depth"])
    valid["shared_taxonomic_depth"] = valid["shared_taxonomic_depth"].astype(int)

    result = (
        valid.groupby("shared_taxonomic_depth")["cosine_similarity"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values("shared_taxonomic_depth")
    )

    result["iqr"] = result["q75"] - result["q25"]

    result.to_csv(
        output / "knn_similarity_by_shared_taxonomy_depth.csv",
        index=False,
    )


def make_biological_correlations(
    df: pd.DataFrame,
    output: Path,
) -> None:
    """
    Compute rank-based associations between embedding similarity and
    biological/sequence-property variables.

    Spearman correlation is used because these relationships need not be
    linear and cosine similarity is highly concentrated near one.
    """
    valid = df.copy()

    valid["abs_delta_gc_skew"] = pd.to_numeric(
        valid.get("delta_gc_skew"),
        errors="coerce",
    )

    candidates = {
        "shared_taxonomic_depth": "Shared taxonomy depth",
        "delta_gc": "Absolute GC-content difference",
        "delta_gc_skew": "Absolute GC-skew difference",
        "delta_log_length": "Absolute log1p sequence-length difference",
    }

    rows = []

    for column, interpretation in candidates.items():
        if column not in valid.columns:
            continue

        subset = valid[["cosine_similarity", column]].dropna()

        if len(subset) < 3:
            continue

        rho = subset["cosine_similarity"].corr(
            subset[column],
            method="spearman",
        )

        rows.append(
            {
                "variable": column,
                "interpretation": interpretation,
                "n_pairs": len(subset),
                "spearman_rho": rho,
                "absolute_spearman_rho": abs(rho),
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        output / "knn_biological_correlations.csv",
        index=False,
    )


def make_taxonomic_relationship_summary(
    df: pd.DataFrame,
    output: Path,
) -> None:
    """
    Pair-level biological summary by taxonomic relationship.
    """
    valid = df.dropna(subset=["cosine_similarity", "taxonomic_relationship"]).copy()

    result = (
        valid.groupby("taxonomic_relationship")
        .agg(
            n_pairs=("cosine_similarity", "count"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            median_cosine_similarity=("cosine_similarity", "median"),
            std_cosine_similarity=("cosine_similarity", "std"),
            mean_shared_taxonomic_depth=(
                "shared_taxonomic_depth",
                "mean",
            ),
            median_shared_taxonomic_depth=(
                "shared_taxonomic_depth",
                "median",
            ),
            mean_delta_gc=("delta_gc", "mean"),
            median_delta_gc=("delta_gc", "median"),
            mean_delta_log_length=(
                "delta_log_length",
                "mean",
            ),
            median_delta_log_length=(
                "delta_log_length",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            "median_cosine_similarity",
            ascending=False,
        )
    )

    result.to_csv(
        output / "knn_taxonomic_relationship_summary.csv",
        index=False,
    )


def make_biological_property_bins(
    df: pd.DataFrame,
    output: Path,
) -> None:
    """
    Bin sequence-property differences and summarize similarity within bins.

    This makes the GC/length effects quantitatively inspectable without
    storing another sequence-level dataset.
    """
    rows = []

    specifications = [
        (
            "delta_gc",
            "gc_difference_bin",
            [0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.25, np.inf],
            [
                "0–0.01",
                "0.01–0.025",
                "0.025–0.05",
                "0.05–0.10",
                "0.10–0.15",
                "0.15–0.25",
                "≥0.25",
            ],
        ),
        (
            "delta_log_length",
            "log_length_difference_bin",
            [0, 0.25, 0.5, 1, 2, 3, 4, np.inf],
            [
                "0–0.25",
                "0.25–0.5",
                "0.5–1",
                "1–2",
                "2–3",
                "3–4",
                "≥4",
            ],
        ),
        (
            "delta_gc_skew",
            "gc_skew_difference_bin",
            [0, 0.01, 0.025, 0.05, 0.10, 0.20, np.inf],
            [
                "0–0.01",
                "0.01–0.025",
                "0.025–0.05",
                "0.05–0.10",
                "0.10–0.20",
                "≥0.20",
            ],
        ),
    ]

    for value_column, bin_column, bins, labels in specifications:
        if value_column not in df.columns:
            continue

        subset = df[["cosine_similarity", value_column]].dropna().copy()

        if subset.empty:
            continue

        subset[bin_column] = pd.cut(
            subset[value_column],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        summary = (
            subset.groupby(
                bin_column,
                observed=False,
            )["cosine_similarity"]
            .agg(
                n="count",
                mean="mean",
                median="median",
                std="std",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )

        summary.insert(
            0,
            "property",
            value_column,
        )

        rows.append(summary)

    if rows:
        result = pd.concat(rows, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "property",
                "bin",
                "n",
                "mean",
                "median",
                "std",
                "q25",
                "q75",
            ]
        )

    result.to_csv(
        output / "knn_similarity_by_biological_property_bins.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Plot 1
# ---------------------------------------------------------------------------


def plot_similarity_by_rank(
    df: pd.DataFrame,
    output: Path,
) -> None:
    stats = (
        df.dropna(subset=["cosine_similarity"])
        .groupby("neighbor_rank")["cosine_similarity"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    # Colour-blind-friendly semantic palette.
    mean_color = "#0072B2"
    median_color = "#D55E00"

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(
        stats["neighbor_rank"],
        stats["mean"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        color=mean_color,
        label="Mean cosine similarity",
    )

    ax.plot(
        stats["neighbor_rank"],
        stats["median"],
        marker="s",
        linewidth=2.5,
        markersize=7,
        color=median_color,
        label="Median cosine similarity",
    )

    ax.set_xlabel("KNN neighbour rank", fontsize=15)
    ax.set_ylabel("Cosine similarity", fontsize=15)
    ax.set_title(
        "Carbon-3B KNN similarity by neighbour rank",
        fontsize=18,
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=True, fontsize=12)

    fig.tight_layout()
    fig.savefig(
        output / "similarity_by_neighbor_rank.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2
# ---------------------------------------------------------------------------


def plot_taxonomic_agreement(
    df: pd.DataFrame,
    output: Path,
) -> None:
    valid = df.dropna(subset=["query_taxonomy", "neighbor_taxonomy"])

    rows = []

    for k in (1, 5, 10):
        subset = valid[valid["neighbor_rank"] <= k]

        if subset.empty:
            continue

        grouped = subset.groupby(
            [
                "query_record_id",
                "query_start",
                "query_end",
            ]
        )

        rows.append(
            {
                "k": k,
                "same_terminal_taxonomy": (
                    grouped["same_terminal_taxonomy"].max().mean() * 100
                ),
                "shared_depth_ge_5": (
                    (grouped["shared_taxonomic_depth"].max() >= 5).mean() * 100
                ),
                "shared_depth_ge_10": (
                    (grouped["shared_taxonomic_depth"].max() >= 10).mean() * 100
                ),
            }
        )

    stats = pd.DataFrame(rows)

    # Ordered semantic palette: strongest taxonomic identity is darkest.
    colors = {
        "Same terminal taxonomy": "#0072B2",
        "Shared taxonomy depth ≥ 10": "#009E73",
        "Shared taxonomy depth ≥ 5": "#56B4E9",
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))

    x = np.arange(len(stats))
    width = 0.25

    ax.bar(
        x - width,
        stats["same_terminal_taxonomy"],
        width,
        color=colors["Same terminal taxonomy"],
        label="Same terminal taxonomy",
    )

    ax.bar(
        x,
        stats["shared_depth_ge_5"],
        width,
        color=colors["Shared taxonomy depth ≥ 5"],
        label="Shared taxonomy depth ≥ 5",
    )

    ax.bar(
        x + width,
        stats["shared_depth_ge_10"],
        width,
        color=colors["Shared taxonomy depth ≥ 10"],
        label="Shared taxonomy depth ≥ 10",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in stats["k"]])
    ax.set_ylim(0, 100)

    ax.set_ylabel(
        "Queries with at least one matching neighbour (%)",
        fontsize=15,
    )
    ax.set_xlabel("KNN neighbourhood", fontsize=15)
    ax.set_title(
        "Taxonomic agreement within Carbon-3B KNN neighbourhoods",
        fontsize=18,
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=True, fontsize=11)

    fig.tight_layout()
    fig.savefig(
        output / "taxonomic_agreement.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3
# ---------------------------------------------------------------------------


def plot_knn_taxonomic_composition(
    df: pd.DataFrame,
    output: Path,
) -> None:
    valid = df.dropna(subset=["shared_taxonomic_depth"]).copy()

    categories = pd.cut(
        valid["shared_taxonomic_depth"],
        bins=[-1, 0, 2, 4, 7, 9, np.inf],
        labels=[
            "No shared prefix",
            "Depth 1–2",
            "Depth 3–4",
            "Depth 5–7",
            "Depth 8–9",
            "Depth ≥10",
        ],
    )

    counts = categories.value_counts().sort_index()

    # Sequential progression from weak to deep taxonomic sharing.
    colors = [
        "#D55E00",
        "#E69F00",
        "#F0E442",
        "#56B4E9",
        "#009E73",
        "#0072B2",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=colors,
    )

    ax.set_xlabel("Shared taxonomy depth", fontsize=15)
    ax.set_ylabel("KNN edges", fontsize=15)
    ax.set_title(
        "Taxonomic composition of Carbon-3B KNN edges",
        fontsize=18,
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.22)

    # Compact count labels make the biological distribution directly readable.
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(
        output / "knn_taxonomic_composition.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4
# ---------------------------------------------------------------------------


def plot_similarity_by_relationship(
    df: pd.DataFrame,
    output: Path,
) -> None:
    valid = df.dropna(subset=["cosine_similarity", "taxonomic_relationship"]).copy()

    # Parse only relationships whose labels explicitly encode shared depth.
    depth_rows = valid[
        valid["taxonomic_relationship"]
        .astype(str)
        .str.startswith("shared_prefix_depth_")
    ].copy()

    depth_rows["shared_depth"] = (
        depth_rows["taxonomic_relationship"]
        .astype(str)
        .str.extract(r"shared_prefix_depth_(\d+)", expand=False)
        .astype(float)
    )

    depth_rows = depth_rows.dropna(subset=["shared_depth"])
    depth_rows["shared_depth"] = depth_rows["shared_depth"].astype(int)

    depths = sorted(depth_rows["shared_depth"].unique())

    grouped = [
        depth_rows.loc[
            depth_rows["shared_depth"] == depth,
            "cosine_similarity",
        ].values
        for depth in depths
    ]

    terminal = valid.loc[
        valid["taxonomic_relationship"] == "same_terminal_taxonomy",
        "cosine_similarity",
    ].values

    labels = [f"Depth {d}" for d in depths]
    datasets = grouped.copy()

    # Add terminal-taxonomy category only if present.
    if len(terminal) > 0:
        labels.append("Same terminal")
        datasets.append(terminal)

    if not datasets:
        LOGGER.warning(
            "No taxonomic relationship groups available for similarity boxplot."
        )
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot(
        datasets,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#000000", "linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
    )

    # Sequential blue palette: deeper shared taxonomy = darker blue.
    cmap = plt.get_cmap("Blues")
    n_depths = max(len(depths), 1)

    for i, patch in enumerate(bp["boxes"]):
        if i < len(depths):
            fraction = 0.35 + 0.55 * (i / max(n_depths - 1, 1))
            patch.set_facecolor(cmap(fraction))
        else:
            patch.set_facecolor("#D55E00")

    ax.set_xlabel("Shared taxonomy depth", fontsize=15)
    ax.set_ylabel("Cosine similarity", fontsize=15)
    ax.set_title(
        "Carbon-3B similarity across taxonomic relationships",
        fontsize=18,
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.22)

    fig.tight_layout()
    fig.savefig(
        output / "similarity_by_taxonomic_relationship.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5
# ---------------------------------------------------------------------------


def plot_similarity_vs_delta_gc(
    df: pd.DataFrame,
    output: Path,
) -> None:
    valid = df.dropna(subset=["cosine_similarity", "delta_gc"])

    if len(valid) > 250_000:
        valid = valid.sample(
            n=250_000,
            random_state=42,
        )

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.scatter(
        valid["delta_gc"],
        valid["cosine_similarity"],
        s=6,
        alpha=0.13,
        color="#0072B2",
        rasterized=True,
    )

    ax.set_xlabel("Absolute GC-content difference", fontsize=15)
    ax.set_ylabel("Cosine similarity", fontsize=15)
    ax.set_title(
        "Carbon-3B similarity versus GC-content difference",
        fontsize=18,
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.22)

    fig.tight_layout()
    fig.savefig(
        output / "similarity_vs_delta_gc.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6
# ---------------------------------------------------------------------------


def plot_similarity_vs_delta_log_length(
    df: pd.DataFrame,
    output: Path,
) -> None:
    valid = df.dropna(
        subset=[
            "cosine_similarity",
            "delta_log_length",
        ]
    )

    if len(valid) > 250_000:
        valid = valid.sample(
            n=250_000,
            random_state=42,
        )

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.scatter(
        valid["delta_log_length"],
        valid["cosine_similarity"],
        s=6,
        alpha=0.13,
        color="#009E73",
        rasterized=True,
    )

    ax.set_xlabel(
        "Absolute log1p(sequence-length difference)",
        fontsize=15,
    )
    ax.set_ylabel("Cosine similarity", fontsize=15)
    ax.set_title(
        "Carbon-3B similarity versus sequence-length difference",
        fontsize=18,
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.22)

    fig.tight_layout()
    fig.savefig(
        output / "similarity_vs_delta_log_length.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Level 3 Carbon-3B KNN biological analysis "
            "using ClickHouseResource."
        )
    )

    parser.add_argument(
        "--knn",
        type=Path,
        default=Path(DEFAULT_KNN),
        help="Path to knn_neighbors.csv",
    )

    parser.add_argument(
        "--cpu-hf-glob",
        default=DEFAULT_CPU_HF_GLOB,
        help="HF Parquet glob for CPU-enriched metadata.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output directory.",
    )

    parser.add_argument(
        "--clickhouse-binary",
        default=None,
        help="Optional ClickHouse binary path.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional ClickHouse thread count.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()

    args = parse_args()

    knn_path = args.knn.resolve()
    output_dir = args.output.resolve()

    if not knn_path.exists():
        raise FileNotFoundError(f"KNN file does not exist: {knn_path}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info("Output directory: %s", output_dir)

    # ---------------------------------------------------------------
    # ClickHouse configuration
    # ---------------------------------------------------------------

    config_kwargs = {}

    if args.clickhouse_binary:
        config_kwargs["binary_path"] = args.clickhouse_binary

    if args.threads:
        config_kwargs["threads"] = args.threads

    clickhouse_config = ClickHouseConfig(**config_kwargs)

    clickhouse = ClickHouseResource(clickhouse_config)

    # ---------------------------------------------------------------
    # Temporary KNN Parquet
    # ---------------------------------------------------------------

    temp_dir = Path(tempfile.mkdtemp(prefix="level3_knn_"))

    temp_knn = temp_dir / "knn_neighbors.parquet"

    try:
        convert_knn_csv_to_parquet(
            knn_path,
            temp_knn,
        )

        # -----------------------------------------------------------
        # Register datasets
        # -----------------------------------------------------------

        LOGGER.info("Registering KNN dataset with ClickHouse")

        clickhouse.register_dataset(
            "knn",
            str(temp_knn),
        )

        LOGGER.info("Registering CPU-enriched HF dataset with ClickHouse")

        clickhouse.register_hf_dataset(
            "cpu",
            args.cpu_hf_glob,
            revision="main",
            pattern="*.parquet",
        )

        LOGGER.info("Registered ClickHouse sources: knn, cpu")

        # -----------------------------------------------------------
        # Analysis
        # -----------------------------------------------------------

        df = load_knn_metadata(clickhouse)

        if df.empty:
            raise RuntimeError("ClickHouse returned zero KNN/metadata rows.")

        # -----------------------------------------------------------
        # Normalize dtypes
        # -----------------------------------------------------------

        numeric_columns = [
            "neighbor_rank",
            "cosine_similarity",
            "cosine_distance",
            "query_gc",
            "neighbor_gc",
            "query_gc_skew",
            "neighbor_gc_skew",
            "query_sequence_length",
            "neighbor_sequence_length",
            "delta_gc",
            "delta_gc_skew",
            "delta_log_length",
            "shared_taxonomic_depth",
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # -----------------------------------------------------------
        # Derived CSVs
        # -----------------------------------------------------------

        LOGGER.info("Writing derived CSVs")

        make_similarity_by_rank(
            df,
            output_dir,
        )

        make_taxonomic_consistency(
            df,
            output_dir,
        )

        make_taxonomic_relationships(
            df,
            output_dir,
        )

        make_sequence_properties(
            df,
            output_dir,
        )

        make_taxonomy_depth_similarity_summary(
            df,
            output_dir,
        )

        make_biological_correlations(
            df,
            output_dir,
        )

        make_taxonomic_relationship_summary(
            df,
            output_dir,
        )

        make_biological_property_bins(
            df,
            output_dir,
        )

        # -----------------------------------------------------------
        # Plots
        # -----------------------------------------------------------

        LOGGER.info("Generating plots")

        plot_similarity_by_rank(
            df,
            plots_dir,
        )

        plot_taxonomic_agreement(
            df,
            plots_dir,
        )

        plot_knn_taxonomic_composition(
            df,
            plots_dir,
        )

        plot_similarity_by_relationship(
            df,
            plots_dir,
        )

        plot_similarity_vs_delta_gc(
            df,
            plots_dir,
        )

        plot_similarity_vs_delta_log_length(
            df,
            plots_dir,
        )

        LOGGER.info("Level 3 analysis complete.")

        LOGGER.info(
            "Rows analysed: %s",
            f"{len(df):,}",
        )

        LOGGER.info(
            "Results: %s",
            output_dir,
        )

    finally:
        LOGGER.info("Removing temporary KNN Parquet directory")

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
