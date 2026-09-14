#!/usr/bin/env python3
"""
Standalone Level-3 KNN biological visualization regeneration.

No Dagster and no embedding regeneration are required.

Primary input:
    knn_taxonomic_relationships.csv

Optional:
    knn_similarity_by_shared_taxonomy_depth.csv
    knn_taxonomic_consistency.csv
    knn_sequence_properties.csv
    knn_biological_correlations.csv

Outputs:
    knn_taxonomic_composition.png
    knn_similarity_by_shared_taxonomy_depth.png
    knn_taxonomic_relationship_boxplot.png
    knn_similarity_vs_delta_gc.png
    knn_similarity_vs_delta_log_length.png
    knn_similarity_vs_delta_gc_skew.png
    knn_taxonomic_consistency_vs_k.png
    knn_biological_correlations.png

Self-neighbours are removed when exact query/neighbor record_id + start + end
columns are available.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 360
FS = 1.12

BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
LIGHT_BLUE = "#56B4E9"
GOLD = "#E69F00"
PURPLE = "#7B61A8"
GREY = "#666666"

plt.rcParams.update(
    {
        "font.size": 10.5 * FS,
        "axes.titlesize": 14 * FS,
        "axes.labelsize": 11 * FS,
        "xtick.labelsize": 9.5 * FS,
        "ytick.labelsize": 9.5 * FS,
        "legend.fontsize": 9.5 * FS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
    }
)


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((x for x in names if x in df.columns), None)


def load_csv(path: Path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required CSV not found: {path}")
        print(f"[SKIP] {path.name} not found")
        return None
    df = pd.read_csv(path)
    print(f"[LOAD] {path.name}: {len(df):,} rows")
    return df


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def remove_self_neighbours(df):
    cols = {
        "query_record_id",
        "neighbor_record_id",
        "query_start",
        "neighbor_start",
        "query_end",
        "neighbor_end",
    }
    if not cols.issubset(df.columns):
        print("[WARN] Could not apply self-neighbour filter: identity columns missing.")
        return df.copy(), 0

    mask = (
        df["query_record_id"].astype(str).eq(df["neighbor_record_id"].astype(str))
        & num(df, "query_start").eq(num(df, "neighbor_start"))
        & num(df, "query_end").eq(num(df, "neighbor_end"))
    )
    n = int(mask.sum())
    print(f"[FILTER] Removed {n:,} self-neighbour edges")
    return df.loc[~mask].copy(), n


def save(fig, output, stem):
    path = output / f"{stem}.png"
    # constrained_layout handles the figure geometry; bbox_inches=tight
    # provides a final safe crop around labels.
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {path}")


def depth_col(df):
    return first_existing(
        df,
        [
            "shared_taxonomic_depth",
            "shared_taxonomy_depth",
            "shared_depth",
            "taxonomy_shared_depth",
            "common_taxonomy_depth",
        ],
    )


def relationship_col(df):
    return first_existing(
        df, ["taxonomic_relationship", "taxonomy_relationship", "relationship"]
    )


def similarity_col(df):
    return first_existing(df, ["cosine_similarity", "similarity"])


def composition_bin(x):
    if pd.isna(x):
        return "Unknown"
    x = int(x)
    if x <= 0:
        return "No shared taxonomy"
    if x <= 4:
        return "Shared depth 1–4"
    if x <= 7:
        return "Shared depth 5–7"
    if x <= 9:
        return "Shared depth 8–9"
    return "Shared depth 10+"


def plot_composition(df, output):
    dc = depth_col(df)
    rc = relationship_col(df)

    if dc:
        cats = num(df, dc).map(composition_bin)
        order = [
            "No shared taxonomy",
            "Shared depth 1–4",
            "Shared depth 5–7",
            "Shared depth 8–9",
            "Shared depth 10+",
            "Unknown",
        ]
    elif rc:
        cats = df[rc].astype(str)
        order = cats.value_counts().index.tolist()
    else:
        print("[SKIP] composition: no taxonomy depth/relationship column")
        return

    counts = cats.value_counts().reindex(order).fillna(0)
    counts = counts[counts > 0]
    pct = counts / counts.sum() * 100

    palette = [BLUE, LIGHT_BLUE, TEAL, GOLD, ORANGE, GREY]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    bars = ax.bar(
        np.arange(len(pct)),
        pct.values,
        color=palette[: len(pct)],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(np.arange(len(pct)))
    ax.set_xticklabels(pct.index, rotation=18, ha="right")
    ax.set_xlabel("Taxonomic relationship")
    ax.set_ylabel("KNN edges (%)")
    ax.set_title("KNN taxonomic composition")

    ymax = max(float(pct.max()) * 1.18, 5)
    ax.set_ylim(0, ymax)

    for b, p in zip(bars, pct):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + ymax * 0.015,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.7 * FS,
        )

    ax.text(
        0.99,
        0.98,
        f"Edges analysed: {len(df):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5 * FS,
    )
    save(fig, output, "knn_taxonomic_composition")


def plot_similarity_depth(df, output):
    dc = depth_col(df)
    sc = similarity_col(df)

    if not dc or not sc:
        print("[SKIP] similarity-by-depth: required columns missing")
        return

    tmp = pd.DataFrame(
        {
            "depth": pd.to_numeric(df[dc], errors="coerce"),
            "similarity": pd.to_numeric(df[sc], errors="coerce"),
        }
    )

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if tmp.empty:
        print("[SKIP] similarity-by-depth: no valid observations")
        return

    # Explicit aggregation avoids the pandas named-aggregation ambiguity.
    rows = []

    for depth, group in tmp.groupby("depth", sort=True):
        values = group["similarity"].to_numpy(dtype=float)

        rows.append(
            {
                "depth": float(depth),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "n": int(len(values)),
            }
        )

    s = pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)

    if s.empty:
        print("[SKIP] similarity-by-depth: no grouped observations")
        return

    print(
        f"[PLOT] Shared taxonomy depths: "
        f"{int(s['depth'].min())}–{int(s['depth'].max())} "
        f"({len(s)} depths)"
    )

    print(f"[PLOT] Similarity range: {s['median'].min():.6f}–{s['median'].max():.6f}")

    fig, ax = plt.subplots(figsize=(10.5, 6.6))

    # IQR envelope.
    ax.fill_between(
        s["depth"].to_numpy(dtype=float),
        s["q25"].to_numpy(dtype=float),
        s["q75"].to_numpy(dtype=float),
        alpha=0.22,
        color=BLUE,
        label="IQR",
    )

    # Median similarity.
    ax.plot(
        s["depth"].to_numpy(dtype=float),
        s["median"].to_numpy(dtype=float),
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        color=BLUE,
        label="Median cosine similarity",
    )

    ax.set_xlabel("Shared taxonomy depth")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity increases with shared taxonomy depth")

    ax.legend(frameon=False)

    # Annotate sample size for each depth.
    if len(s) <= 25:
        for _, row in s.iterrows():
            ax.annotate(
                f"n={int(row['n']):,}",
                (row["depth"], row["median"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7.5 * FS,
            )

    ax.set_xticks(s["depth"].astype(int).tolist())

    save(
        fig,
        output,
        "knn_similarity_by_shared_taxonomy_depth",
    )


def plot_relationship_boxplot(df, output):
    """
    Plot taxonomic-relationship similarity horizontally.

    Horizontal orientation is intentional: the dataset contains many
    relationship categories (Depth 1 ... Depth 18 + terminal taxonomy).
    This prevents long x-axis labels from colliding.
    """
    rc, sc = relationship_col(df), similarity_col(df)

    if not rc or not sc:
        print("[SKIP] relationship boxplot: required columns missing")
        return

    tmp = pd.DataFrame(
        {
            "relationship": df[rc].astype(str),
            "similarity": num(df, sc),
        }
    ).dropna()

    if tmp.empty:
        print("[SKIP] relationship boxplot: no valid observations")
        return

    dc = depth_col(df)

    if dc:
        tmp["depth"] = num(df.loc[tmp.index], dc)

        # Use the numeric shared depth for ordering.
        # Terminal-taxonomy relationships are placed after the deepest
        # shared-prefix depth.
        relationship_depth = tmp.groupby("relationship")["depth"].median().sort_values()

        order = relationship_depth.index.tolist()

        terminal_names = [
            r for r in order if "terminal" in r.lower() or "same_tax" in r.lower()
        ]

        non_terminal = [r for r in order if r not in terminal_names]

        order = non_terminal + terminal_names
    else:
        order = (
            tmp.groupby("relationship")["similarity"]
            .median()
            .sort_values()
            .index.tolist()
        )

    data = [tmp.loc[tmp["relationship"] == r, "similarity"].to_numpy() for r in order]

    def pretty_relationship(r):
        r_lower = r.lower()

        if "terminal" in r_lower or "same_tax" in r_lower:
            return "Same terminal"

        if "shared_prefix_depth_" in r_lower:
            depth = r_lower.split("shared_prefix_depth_")[-1]
            return f"Depth {depth}"

        if "depth_" in r_lower:
            depth = r_lower.split("depth_")[-1]
            return f"Depth {depth}"

        return r.replace("_", " ")

    labels = [pretty_relationship(r) for r in order]

    # One row per relationship gives enough vertical separation for all
    # 19 categories and keeps labels completely readable.
    fig_height = max(8.5, 0.48 * len(order) + 2.5)

    fig, ax = plt.subplots(
        figsize=(11.5, fig_height),
        constrained_layout=True,
    )

    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.68,
        medianprops={"linewidth": 2, "color": "black"},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        boxprops={"linewidth": 0.8},
    )

    for i, patch in enumerate(bp["boxes"]):
        # Highlight the terminal category while keeping depth categories
        # visually consistent.
        if i == len(bp["boxes"]) - 1 and "terminal" in labels[i].lower():
            patch.set_facecolor(ORANGE)
        else:
            patch.set_facecolor(LIGHT_BLUE)

        patch.set_alpha(0.70)

    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Taxonomic relationship")
    ax.set_title("Cosine similarity across taxonomic relationships")

    # Cosine values are tightly concentrated near 1.0. Use a small amount
    # of horizontal padding so the terminal box does not touch the border.
    xmin = np.nanmin(tmp["similarity"])
    xmax = np.nanmax(tmp["similarity"])
    span = max(xmax - xmin, 0.001)

    ax.set_xlim(
        xmin - 0.04 * span,
        min(1.001, xmax + 0.04 * span),
    )

    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    save(fig, output, "knn_taxonomic_relationship_boxplot")


def plot_property(df, xnames, xlabel, title, stem, output, color):
    xc, sc = first_existing(df, xnames), similarity_col(df)
    if not xc or not sc:
        print(f"[SKIP] {stem}: required columns missing")
        return

    tmp = (
        pd.DataFrame({"x": num(df, xc), "y": num(df, sc)})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if tmp.empty:
        return

    sample = tmp.sample(min(len(tmp), 120_000), random_state=42)

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.scatter(
        sample.x, sample.y, s=5, alpha=0.16, color=color, linewidths=0, rasterized=True
    )

    try:
        bins = pd.qcut(
            tmp.x, q=min(30, max(8, int(math.sqrt(len(tmp))))), duplicates="drop"
        )
        trend = (
            tmp.assign(bin=bins)
            .groupby("bin", observed=True)
            .agg(x=("x", "median"), y=("y", "median"))
            .dropna()
        )
        ax.plot(
            trend.x,
            trend.y,
            marker="o",
            markersize=4,
            linewidth=2,
            color=ORANGE,
            label="Binned median",
        )
        ax.legend(frameon=False)
    except ValueError:
        pass

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.text(
        0.015,
        0.975,
        f"Edges analysed: {len(tmp):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5 * FS,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )
    save(fig, output, stem)


def plot_consistency(df, output):
    kc = "k"

    if kc not in df.columns:
        print("[SKIP] consistency: 'k' column missing")
        return

    mappings = [
        (
            "Same terminal taxonomy",
            "same_terminal_taxonomy_fraction",
            ORANGE,
            "o",
        ),
        (
            "Shared depth ≥1",
            "shared_depth_ge_1_fraction",
            LIGHT_BLUE,
            "s",
        ),
        (
            "Shared depth ≥3",
            "shared_depth_ge_3_fraction",
            TEAL,
            "^",
        ),
        (
            "Shared depth ≥5",
            "shared_depth_ge_5_fraction",
            BLUE,
            "D",
        ),
        (
            "Shared depth ≥8",
            "shared_depth_ge_8_fraction",
            GOLD,
            "P",
        ),
        (
            "Shared depth ≥10",
            "shared_depth_ge_10_fraction",
            PURPLE,
            "X",
        ),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)

    plotted = False

    for label, col, color, marker in mappings:
        if col not in df.columns:
            print(f"[WARN] Missing consistency column: {col}")
            continue

        x = pd.to_numeric(df[kc], errors="coerce")
        y = pd.to_numeric(df[col], errors="coerce")

        # Fractions are expected to be 0–1.
        # Convert to percentages for presentation.
        if y.dropna().max() <= 1:
            y = y * 100

        tmp = (
            pd.DataFrame(
                {
                    "k": x,
                    "agreement": y,
                }
            )
            .dropna()
            .sort_values("k")
        )

        if tmp.empty:
            continue

        ax.plot(
            tmp["k"],
            tmp["agreement"],
            marker=marker,
            markersize=6,
            linewidth=2.0,
            color=color,
            label=label,
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        print("[SKIP] consistency: no valid agreement columns")
        return

    ax.set_xlabel("KNN neighbourhood size (K)")
    ax.set_ylabel("Queries satisfying criterion (%)")
    ax.set_title("Taxonomic agreement across KNN neighbourhood size")

    # Give the 100% curves visual headroom.
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.legend(
        frameon=False,
        loc="lower right",
        ncol=2,
        columnspacing=1.2,
        handlelength=2.5,
    )

    # query_count is common to the three K values, so report it once.
    if "query_count" in df.columns:
        counts = pd.to_numeric(df["query_count"], errors="coerce").dropna()

        if not counts.empty:
            ax.text(
                0.015,
                0.975,
                f"Queries per K: n = {int(counts.iloc[0]):,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5 * FS,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

    save(
        fig,
        output,
        "knn_taxonomic_consistency_vs_k",
    )


def plot_correlations(df, output):
    vc = first_existing(df, ["variable", "property", "feature"])
    rc = first_existing(df, ["spearman_rho", "rho", "spearman"])
    if not vc or not rc:
        print("[SKIP] correlations: variable/rho columns missing")
        return

    tmp = (
        pd.DataFrame({"variable": df[vc].astype(str), "rho": num(df, rc)})
        .dropna()
        .sort_values("rho")
    )

    label_map = {
        "delta_gc": "GC-content difference (ΔGC)",
        "delta_gc_skew": "GC-skew difference (ΔGC-skew)",
        "delta_log_length": "Log-length difference (Δlog-length)",
        "shared_taxonomic_depth": "Shared taxonomy depth",
    }
    tmp["display_variable"] = tmp["variable"].map(label_map).fillna(tmp["variable"])

    fig, ax = plt.subplots(figsize=(11, 6.8))
    colors = [ORANGE if x < 0 else BLUE for x in tmp.rho]
    bars = ax.barh(
        np.arange(len(tmp)),
        tmp.rho,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(np.arange(len(tmp)))
    ax.set_yticklabels(tmp["display_variable"])
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Embedding similarity vs biological properties")

    max_abs = max(float(np.abs(tmp["rho"]).max()), 0.01)
    label_offset = max_abs * 0.025

    for b, v in zip(bars, tmp.rho):
        # Place values inside the bar to avoid collision with category labels.
        if v >= 0:
            x = v - label_offset
            ha = "right"
        else:
            x = v + label_offset
            ha = "left"

        ax.text(
            x,
            b.get_y() + b.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha=ha,
            fontsize=8.5 * FS,
        )
    save(fig, output, "knn_biological_correlations")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    inp = args.input_dir
    out = args.output_dir or inp / "visualizations"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LEVEL-3 KNN BIOLOGICAL VISUALIZATION REGENERATION")
    print("=" * 78)

    rel = load_csv(inp / "knn_taxonomic_relationships.csv", required=True)
    rel, removed = remove_self_neighbours(rel)

    plot_composition(rel, out)
    plot_similarity_depth(rel, out)
    plot_relationship_boxplot(rel, out)

    seq = load_csv(inp / "knn_sequence_properties.csv")
    if seq is not None:
        seq, _ = remove_self_neighbours(seq)
    else:
        seq = rel

    plot_property(
        seq,
        ["delta_gc", "abs_delta_gc", "gc_difference", "delta_gc_content"],
        "Absolute GC-content difference (ΔGC)",
        "Embedding similarity vs GC-content difference",
        "knn_similarity_vs_delta_gc",
        out,
        BLUE,
    )
    plot_property(
        seq,
        ["delta_log_length", "abs_delta_log_length", "log_length_difference"],
        "Absolute log-length difference (Δlog-length)",
        "Embedding similarity vs sequence-length difference",
        "knn_similarity_vs_delta_log_length",
        out,
        TEAL,
    )
    plot_property(
        seq,
        ["delta_gc_skew", "abs_delta_gc_skew", "gc_skew_difference"],
        "Absolute GC-skew difference (ΔGC-skew)",
        "Embedding similarity vs GC-skew difference",
        "knn_similarity_vs_delta_gc_skew",
        out,
        PURPLE,
    )

    consistency = load_csv(inp / "knn_taxonomic_consistency.csv")
    if consistency is not None:
        plot_consistency(consistency, out)

    corr = load_csv(inp / "knn_biological_correlations.csv")
    if corr is not None:
        plot_correlations(corr, out)

    print("=" * 78)
    print(f"Self-neighbour edges excluded: {removed:,}")
    print(f"Figures written to: {out}")


if __name__ == "__main__":
    main()
