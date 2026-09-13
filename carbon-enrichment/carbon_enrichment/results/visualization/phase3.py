#!/usr/bin/env python3

"""
Level 1 — Single-sequence Carbon embedding analysis
====================================================

Target
------
record_id = NW_006267348.1
start     = 1,004,323
end       = 1,004,602

Dataset
-------
AINovice2005/carbon-embeddings

Visualizations
--------------
1. Raw embedding fingerprint
2. Embedding dimension profile
3. 3D spherical embedding fingerprint
4. 4D-style interactive embedding fingerprint
5. Embedding-value distribution
6. Unified Level-1 dashboard

Important methodological note
-----------------------------
The 3D/4D fingerprint is a visualization encoding of ONE
high-dimensional vector. It is NOT PCA, UMAP, or another
dimensionality-reduction embedding-space projection.

For multiple sequences, PCA/UMAP should be used instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "AINovice2005/carbon-embeddings"
SPLIT = "train"

TARGET_RECORD_ID = "NW_006267348.1"
TARGET_START = 1_004_323
TARGET_END = 1_004_602

OUTPUT_DIR = Path("carbon_embedding_level1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_PROGRESS = True

EXPECTED_DIMENSIONS = 3072

# Number of dimensions highlighted in the 3D plot.
TOP_N_3D = 250

# Marker size limits for 3D visualization.
MIN_MARKER_SIZE = 2.5
MAX_MARKER_SIZE = 13.0

# Distribution/profile settings.
# The central view exposes structure that is visually compressed
# when the full range is approximately -30 to +30.
CENTRAL_RANGE = 0.5
FULL_RANGE_NBINS = 120
CENTRAL_RANGE_NBINS = 100
LOG_MAGNITUDE_NBINS = 80


# ============================================================
# 1. STREAM DATASET AND FIND TARGET
# ============================================================

def find_target_embedding():

    print("=" * 80)
    print("LOADING CARBON EMBEDDING DATASET")
    print("=" * 80)

    print(f"Dataset : {DATASET_NAME}")
    print(f"Split   : {SPLIT}")

    print()
    print("Target:")
    print(f"  record_id = {TARGET_RECORD_ID}")
    print(f"  start     = {TARGET_START:,}")
    print(f"  end       = {TARGET_END:,}")
    print()

    ds = load_dataset(
        DATASET_NAME,
        split=SPLIT,
        streaming=True,
    )

    target = None
    rows_scanned = 0

    for row in ds:

        rows_scanned += 1

        if SHOW_PROGRESS and rows_scanned % 10_000 == 0:
            print(
                f"Scanned {rows_scanned:,} rows...",
                end="\r",
            )

        if (
            row["record_id"] == TARGET_RECORD_ID
            and int(row["start"]) == TARGET_START
            and int(row["end"]) == TARGET_END
        ):
            target = row
            break

    print()

    if target is None:

        raise RuntimeError(
            "\nTarget embedding was not found.\n"
            f"record_id = {TARGET_RECORD_ID}\n"
            f"start     = {TARGET_START}\n"
            f"end       = {TARGET_END}\n"
        )

    print(
        f"Target found after scanning "
        f"{rows_scanned:,} rows."
    )

    return target


# ============================================================
# 2. PREPARE EMBEDDING
# ============================================================

def prepare_embedding(target):

    embedding = np.asarray(
        target["embedding"],
        dtype=np.float32,
    )

    if embedding.ndim != 1:

        raise ValueError(
            f"Expected 1-D embedding, "
            f"got shape {embedding.shape}"
        )

    dimensions = embedding.size

    print()
    print("=" * 80)
    print("TARGET VALIDATION")
    print("=" * 80)

    print(f"record_id       : {target['record_id']}")
    print(f"start           : {int(target['start']):,}")
    print(f"end             : {int(target['end']):,}")

    print(
        f"corpus span     : "
        f"{int(target['end']) - int(target['start']):,}"
    )

    print(f"embedding dims  : {dimensions:,}")

    print(
        f"embedding_norm  : "
        f"{target['embedding_norm']}"
    )

    if dimensions != EXPECTED_DIMENSIONS:

        print()
        print(
            f"WARNING: expected "
            f"{EXPECTED_DIMENSIONS:,} dimensions "
            f"but found {dimensions:,}."
        )

    else:

        print(
            f"Dimension check : "
            f"OK ({EXPECTED_DIMENSIONS:,})"
        )

    return embedding


# ============================================================
# 3. STATISTICS
# ============================================================

def calculate_statistics(embedding):

    abs_values = np.abs(embedding)

    largest_indices = np.argsort(
        abs_values
    )[::-1][:20]

    stats = {

        "dimensions":
            int(embedding.size),

        "min":
            float(np.min(embedding)),

        "max":
            float(np.max(embedding)),

        "mean":
            float(np.mean(embedding)),

        "std":
            float(np.std(embedding)),

        "median":
            float(np.median(embedding)),

        "l2_norm_calculated":
            float(np.linalg.norm(embedding)),

        "l1_norm":
            float(np.sum(abs_values)),

        "q01":
            float(np.percentile(embedding, 1)),

        "q05":
            float(np.percentile(embedding, 5)),

        "q25":
            float(np.percentile(embedding, 25)),

        "q75":
            float(np.percentile(embedding, 75)),

        "q95":
            float(np.percentile(embedding, 95)),

        "q99":
            float(np.percentile(embedding, 99)),

        "zero_fraction":
            float(np.mean(embedding == 0)),

        "abs_gt_1":
            int(np.sum(abs_values > 1)),

        "abs_gt_5":
            int(np.sum(abs_values > 5)),

        "abs_gt_10":
            int(np.sum(abs_values > 10)),

        # Exact-zero and near-zero diagnostics.
        "exact_zero_count":
            int(np.sum(embedding == 0)),
        "exact_zero_fraction":
            float(np.mean(embedding == 0)),
        "abs_lt_1e-3_count":
            int(np.sum(abs_values < 1e-3)),
        "abs_lt_1e-2_count":
            int(np.sum(abs_values < 1e-2)),
        "abs_lt_5e-2_count":
            int(np.sum(abs_values < 5e-2)),
        "abs_lt_1e-1_count":
            int(np.sum(abs_values < 1e-1)),

        # Squared-magnitude (L2-energy) concentration.
        "top_10_l2_energy_fraction":
            float(np.sum(np.sort(embedding ** 2)[-10:]) /
                  max(np.sum(embedding ** 2), np.finfo(float).eps)),
        "top_50_l2_energy_fraction":
            float(np.sum(np.sort(embedding ** 2)[-50:]) /
                  max(np.sum(embedding ** 2), np.finfo(float).eps)),
        "top_100_l2_energy_fraction":
            float(np.sum(np.sort(embedding ** 2)[-100:]) /
                  max(np.sum(embedding ** 2), np.finfo(float).eps)),
        "top_250_l2_energy_fraction":
            float(np.sum(np.sort(embedding ** 2)[-250:]) /
                  max(np.sum(embedding ** 2), np.finfo(float).eps)),
        "top_500_l2_energy_fraction":
            float(np.sum(np.sort(embedding ** 2)[-500:]) /
                  max(np.sum(embedding ** 2), np.finfo(float).eps)),
    }

    print()
    print("=" * 80)
    print("EMBEDDING STATISTICS")
    print("=" * 80)

    for key, value in stats.items():
        print(f"{key:25s}: {value}")

    print()
    print("=" * 80)
    print("TOP 20 LARGEST-MAGNITUDE DIMENSIONS")
    print("=" * 80)

    top_rows = []

    for rank, idx in enumerate(
        largest_indices,
        start=1,
    ):

        dimension = int(idx + 1)
        value = float(embedding[idx])

        top_rows.append(
            {
                "rank": rank,
                "dimension": dimension,
                "value": value,
                "absolute_value": abs(value),
            }
        )

        print(
            f"{rank:2d}. "
            f"dimension {dimension:4d} : "
            f"{value: .6f} "
            f"|abs|={abs(value):.6f}"
        )

    return stats, top_rows


# ============================================================
# 4. PER-DIMENSION DATA
# ============================================================

def create_dimension_dataframe(embedding):

    dimensions = np.arange(
        1,
        len(embedding) + 1,
    )

    values = embedding.astype(float)

    abs_values = np.abs(values)

    # Rank by absolute magnitude.
    magnitude_order = np.argsort(
        -abs_values
    )

    magnitude_rank = np.empty(
        len(values),
        dtype=int,
    )

    magnitude_rank[
        magnitude_order
    ] = np.arange(
        1,
        len(values) + 1,
    )

    # Percentile rank based on absolute magnitude.
    percentile = (
        np.argsort(
            np.argsort(abs_values)
        )
        / max(len(values) - 1, 1)
        * 100
    )

    df = pd.DataFrame(
        {
            "dimension": dimensions,
            "value": values,
            "absolute_value": abs_values,
            "magnitude_rank": magnitude_rank,
            "magnitude_percentile": percentile,
        }
    )

    return df


# ============================================================
# 5. SAVE CSV OUTPUTS
# ============================================================

def save_outputs(
    stats,
    top_rows,
    dimension_df,
):

    stats_path = (
        OUTPUT_DIR /
        "embedding_statistics.csv"
    )

    stats_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in stats.items()
        ]
    )

    stats_df.to_csv(
        stats_path,
        index=False,
    )

    top_path = (
        OUTPUT_DIR /
        "top_embedding_dimensions.csv"
    )

    pd.DataFrame(top_rows).to_csv(
        top_path,
        index=False,
    )

    dimensions_path = (
        OUTPUT_DIR /
        "embedding_dimensions.csv"
    )

    dimension_df.to_csv(
        dimensions_path,
        index=False,
    )

    print()
    print(
        f"Saved statistics     : "
        f"{stats_path}"
    )

    print(
        f"Saved top dimensions : "
        f"{top_path}"
    )

    print(
        f"Saved dimension data : "
        f"{dimensions_path}"
    )


# ============================================================
# 6. RAW EMBEDDING FINGERPRINT
# ============================================================

def create_embedding_heatmap(
    embedding,
    target,
):

    dimensions = np.arange(
        1,
        len(embedding) + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(

            z=[embedding],

            x=dimensions,

            y=["Embedding"],

            colorscale="RdBu_r",

            zmid=0,

            colorbar=dict(
                title="Value",
            ),

            hovertemplate=(
                "Dimension: %{x}<br>"
                "Value: %{z:.6f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(

        title=(
            f"Embedding Fingerprint — "
            f"{target['record_id']} "
            f"| {len(embedding):,} dimensions"
        ),

        xaxis=dict(
            title="Embedding dimension",

            rangeslider=dict(
                visible=True,
            ),
        ),

        yaxis=dict(
            title="",
            showticklabels=False,
        ),

        template="plotly_white",

        height=360,

        margin=dict(
            l=70,
            r=70,
            t=85,
            b=85,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_fingerprint.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved fingerprint    : "
        f"{output}"
    )

    return fig


# ============================================================
# 7. IMPROVED DIMENSION PROFILE
# ============================================================

def create_dimension_profile(
    embedding,
    target,
):

    dimensions = np.arange(
        1,
        len(embedding) + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(

            x=dimensions,

            y=embedding,

            mode="lines",

            name="Embedding",

            line=dict(
                width=1.2,
            ),

            hovertemplate=(
                "<b>Dimension %{x}</b><br>"
                "Value: %{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
    )

    fig.update_layout(

        title=(
            f"Embedding Dimension Profile — "
            f"{target['record_id']}"
        ),

        xaxis=dict(
            title="Embedding dimension",

            rangeslider=dict(
                visible=True,
            ),
        ),

        yaxis=dict(
            title="Embedding value",
        ),

        template="plotly_white",

        hovermode="x unified",

        height=560,

        margin=dict(
            l=80,
            r=60,
            t=85,
            b=100,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_dimension_profile.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved dimension plot: "
        f"{output}"
    )

    return fig


# ============================================================
# 8. BUILD 3D SPHERICAL FINGERPRINT
# ============================================================

def spherical_coordinates(
    embedding,
):

    n = len(embedding)

    indices = np.arange(n)

    # Golden-angle distribution.
    #
    # This produces an approximately uniform
    # angular distribution on a sphere.
    golden_angle = np.pi * (
        3.0 - np.sqrt(5.0)
    )

    y = (
        1
        - 2 * indices / max(n - 1, 1)
    )

    radius_xy = np.sqrt(
        np.maximum(
            0,
            1 - y**2,
        )
    )

    theta = (
        golden_angle * indices
    )

    sphere_x = (
        radius_xy *
        np.cos(theta)
    )

    sphere_z = (
        radius_xy *
        np.sin(theta)
    )

    sphere_y = y

    return (
        sphere_x,
        sphere_y,
        sphere_z,
    )


def create_3d_fingerprint(
    embedding,
    target,
    dimension_df,
):

    x_sphere, y_sphere, z_sphere = (
        spherical_coordinates(
            embedding
        )
    )

    abs_values = np.abs(embedding)

    # --------------------------------------------------------
    # Select strongest dimensions for the main rendering.
    #
    # Showing all 3072 points is possible, but the strongest
    # dimensions make the structure much easier to inspect.
    # --------------------------------------------------------

    n_points = min(
        TOP_N_3D,
        len(embedding),
    )

    selected = np.argsort(
        abs_values
    )[::-1][:n_points]

    selected = np.sort(selected)

    values = embedding[selected]
    magnitudes = abs_values[selected]

    x = (
        x_sphere[selected] *
        magnitudes
    )

    y = (
        y_sphere[selected] *
        magnitudes
    )

    z = (
        z_sphere[selected] *
        magnitudes
    )

    dimensions = (
        selected + 1
    )

    ranks = dimension_df.loc[
        selected,
        "magnitude_rank",
    ].to_numpy()

    percentiles = dimension_df.loc[
        selected,
        "magnitude_percentile",
    ].to_numpy()

    # --------------------------------------------------------
    # Marker size
    # --------------------------------------------------------

    if np.max(magnitudes) == np.min(
        magnitudes
    ):

        marker_sizes = np.full(
            len(magnitudes),
            MIN_MARKER_SIZE,
        )

    else:

        scaled = (
            (
                magnitudes
                - np.min(magnitudes)
            )
            /
            (
                np.max(magnitudes)
                - np.min(magnitudes)
            )
        )

        marker_sizes = (
            MIN_MARKER_SIZE
            +
            scaled
            * (
                MAX_MARKER_SIZE
                - MIN_MARKER_SIZE
            )
        )

    hover_text = [

        (
            f"<b>Dimension {dim}</b><br>"
            f"Embedding value: {value:.6f}<br>"
            f"|Value|: {magnitude:.6f}<br>"
            f"Magnitude rank: {rank:,}<br>"
            f"Absolute magnitude percentile: "
            f"{pct:.2f}%"
            f"<extra></extra>"
        )

        for dim, value, magnitude, rank, pct
        in zip(
            dimensions,
            values,
            magnitudes,
            ranks,
            percentiles,
        )
    ]

    # --------------------------------------------------------
    # 3D scatter
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(

            x=x,
            y=y,
            z=z,

            mode="markers",

            marker=dict(

                size=marker_sizes,

                color=values,

                colorscale="RdBu_r",

                cmin=float(
                    np.min(embedding)
                ),

                cmax=float(
                    np.max(embedding)
                ),

                cmid=0,

                opacity=0.85,

                colorbar=dict(
                    title="Signed<br>value",
                ),
            ),

            text=hover_text,

            hovertemplate="%{text}",

            name="Embedding dimensions",
        )
    )

    # --------------------------------------------------------
    # Origin
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(

            x=[0],
            y=[0],
            z=[0],

            mode="markers",

            marker=dict(
                size=5,
                color="black",
            ),

            name="Origin",

            hovertemplate=(
                "<b>Origin</b>"
                "<extra></extra>"
            ),
        )
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        title=dict(

            text=(
                f"3D Embedding Fingerprint — "
                f"{target['record_id']}<br>"
                f"<sup>"
                f"Top {n_points} dimensions by "
                f"absolute magnitude"
                f"</sup>"
            ),

            x=0.5,
            xanchor="center",
        ),

        scene=dict(

            xaxis=dict(
                title="Fingerprint X",
                showbackground=True,
            ),

            yaxis=dict(
                title="Fingerprint Y",
                showbackground=True,
            ),

            zaxis=dict(
                title="Fingerprint Z",
                showbackground=True,
            ),

            aspectmode="cube",

            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.25,
                )
            ),
        ),

        template="plotly_white",

        height=780,

        margin=dict(
            l=0,
            r=0,
            t=100,
            b=0,
        ),

        legend=dict(
            x=0.01,
            y=0.99,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_3d_fingerprint.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved 3D fingerprint: "
        f"{output}"
    )

    return fig


# ============================================================
# 9. 4D-STYLE FINGERPRINT
# ============================================================

def create_4d_fingerprint(
    embedding,
    target,
    dimension_df,
):

    x_sphere, y_sphere, z_sphere = (
        spherical_coordinates(
            embedding
        )
    )

    abs_values = np.abs(embedding)

    n_points = min(
        TOP_N_3D,
        len(embedding),
    )

    selected = np.argsort(
        abs_values
    )[::-1][:n_points]

    values = embedding[selected]
    magnitudes = abs_values[selected]

    # --------------------------------------------------------
    # Radial coordinates
    # --------------------------------------------------------

    x = (
        x_sphere[selected]
        * magnitudes
    )

    y = (
        y_sphere[selected]
        * magnitudes
    )

    z = (
        z_sphere[selected]
        * magnitudes
    )

    dimensions = selected + 1

    ranks = dimension_df.loc[
        selected,
        "magnitude_rank",
    ].to_numpy()

    percentiles = dimension_df.loc[
        selected,
        "magnitude_percentile",
    ].to_numpy()

    # --------------------------------------------------------
    # Size = magnitude
    # --------------------------------------------------------

    if np.ptp(magnitudes) == 0:

        marker_sizes = np.full(
            len(magnitudes),
            MIN_MARKER_SIZE,
        )

    else:

        normalized = (
            magnitudes
            - np.min(magnitudes)
        ) / np.ptp(magnitudes)

        marker_sizes = (
            MIN_MARKER_SIZE
            +
            normalized
            * (
                MAX_MARKER_SIZE
                - MIN_MARKER_SIZE
            )
        )

    customdata = np.column_stack(
        [
            dimensions,
            values,
            magnitudes,
            ranks,
            percentiles,
        ]
    )

    # --------------------------------------------------------
    # 4D-style plot
    #
    # X/Y/Z = spatial encoding
    # Color = signed embedding value
    # Size = magnitude
    #
    # This is effectively a 3D + color + size
    # visualization.
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(

            x=x,
            y=y,
            z=z,

            mode="markers",

            marker=dict(

                size=marker_sizes,

                color=values,

                colorscale="RdBu_r",

                cmin=float(
                    np.min(embedding)
                ),

                cmax=float(
                    np.max(embedding)
                ),

                cmid=0,

                opacity=0.9,

                colorbar=dict(
                    title=(
                        "Embedding<br>"
                        "value"
                    ),
                ),
            ),

            customdata=customdata,

            hovertemplate=(
                "<b>Dimension %{customdata[0]}</b>"
                "<br>"
                "Value: %{customdata[1]:.6f}"
                "<br>"
                "|Value|: %{customdata[2]:.6f}"
                "<br>"
                "Magnitude rank: %{customdata[3]}"
                "<br>"
                "Magnitude percentile: "
                "%{customdata[4]:.2f}%"
                "<extra></extra>"
            ),

            name="Embedding",
        )
    )

    # Origin
    fig.add_trace(
        go.Scatter3d(

            x=[0],
            y=[0],
            z=[0],

            mode="markers",

            marker=dict(
                size=5,
                color="black",
            ),

            name="Origin",

            hovertemplate=(
                "<b>Origin</b>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(

        title=dict(

            text=(
                f"Interactive 4D Embedding "
                f"Fingerprint — "
                f"{target['record_id']}<br>"
                f"<sup>"
                f"3D position + signed color "
                f"+ magnitude size"
                f"</sup>"
            ),

            x=0.5,
            xanchor="center",
        ),

        scene=dict(

            xaxis=dict(
                title="X",
            ),

            yaxis=dict(
                title="Y",
            ),

            zaxis=dict(
                title="Z",
            ),

            aspectmode="cube",

            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.25,
                )
            ),
        ),

        template="plotly_white",

        height=820,

        margin=dict(
            l=0,
            r=0,
            t=105,
            b=0,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_4d_fingerprint.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved 4D fingerprint: "
        f"{output}"
    )

    return fig


# ============================================================
# 10. DISTRIBUTION AND MAGNITUDE-CONCENTRATION VIEWS
# ============================================================

def create_distribution(
    embedding,
    target,
):
    """Full-range histogram of all embedding values."""

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=embedding,
            nbinsx=FULL_RANGE_NBINS,
            hovertemplate=(
                "Embedding value: %{x:.6f}"
                "<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
            name="Embedding values",
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1,
    )

    fig.update_layout(
        title=(
            f"Embedding Value Distribution — "
            f"{target['record_id']}<br>"
            f"<sup>Full range; {len(embedding):,} dimensions</sup>"
        ),
        xaxis_title="Embedding value",
        yaxis_title="Dimension count",
        template="plotly_white",
        height=500,
        bargap=0.02,
        margin=dict(
            l=80,
            r=60,
            t=95,
            b=80,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_distribution.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved distribution   : "
        f"{output}"
    )

    return fig


def create_central_distribution(
    embedding,
    target,
):
    """
    Zoomed histogram around zero.

    This view is deliberately separated from the full-range
    distribution so small but non-zero dimensions are not visually
    compressed by a few large-magnitude values.
    """

    central = embedding[
        np.abs(embedding) <= CENTRAL_RANGE
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=central,
            nbinsx=CENTRAL_RANGE_NBINS,
            hovertemplate=(
                "Embedding value: %{x:.6f}"
                "<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
            name="Central values",
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1,
    )

    fraction = (
        len(central) / max(len(embedding), 1) * 100
    )

    fig.update_layout(
        title=(
            f"Central Embedding Distribution — "
            f"{target['record_id']}<br>"
            f"<sup>"
            f"|value| ≤ {CENTRAL_RANGE:g}; "
            f"{len(central):,}/{len(embedding):,} dimensions "
            f"({fraction:.2f}%)"
            f"</sup>"
        ),
        xaxis=dict(
            title="Embedding value",
            range=[
                -CENTRAL_RANGE,
                CENTRAL_RANGE,
            ],
        ),
        yaxis_title="Dimension count",
        template="plotly_white",
        height=500,
        bargap=0.02,
        margin=dict(
            l=80,
            r=60,
            t=95,
            b=80,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_distribution_central.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved central distribution: "
        f"{output}"
    )

    return fig


def create_log_magnitude_distribution(
    embedding,
    target,
):
    """
    Histogram of log10(abs(value)).

    Exact zeros are handled explicitly. For a dense vector they
    should normally be absent; if present, they are excluded from
    the log transform and reported in the title.
    """

    abs_values = np.abs(embedding)
    nonzero = abs_values[abs_values > 0]

    if len(nonzero) == 0:
        raise RuntimeError(
            "All embedding dimensions are exactly zero; "
            "log-magnitude distribution is undefined."
        )

    log_magnitude = np.log10(nonzero)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=log_magnitude,
            nbinsx=LOG_MAGNITUDE_NBINS,
            hovertemplate=(
                "log10(|value|): %{x:.4f}"
                "<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
            name="Log magnitude",
        )
    )

    fig.update_layout(
        title=(
            f"Embedding Magnitude Distribution — "
            f"{target['record_id']}<br>"
            f"<sup>"
            f"log10(|embedding value|); "
            f"zeros excluded: "
            f"{np.sum(abs_values == 0):,}"
            f"</sup>"
        ),
        xaxis=dict(
            title="log10(|embedding value|)",
        ),
        yaxis_title="Dimension count",
        template="plotly_white",
        height=500,
        bargap=0.02,
        margin=dict(
            l=80,
            r=60,
            t=95,
            b=80,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_distribution_log_magnitude.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved log-magnitude plot: "
        f"{output}"
    )

    return fig


def create_ranked_magnitude_profile(
    embedding,
    target,
):
    """
    Rank dimensions by absolute magnitude.

    The second trace shows cumulative L2-energy contribution,
    allowing direct inspection of whether a small subset of
    dimensions dominates the vector's squared magnitude.
    """

    abs_values = np.abs(embedding)
    order = np.argsort(abs_values)[::-1]

    ranked = abs_values[order]
    ranks = np.arange(1, len(ranked) + 1)

    squared = ranked ** 2
    total_squared = max(
        float(np.sum(squared)),
        np.finfo(float).eps,
    )
    cumulative_energy = (
        np.cumsum(squared) / total_squared * 100
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=ranks,
            y=ranked,
            mode="lines",
            name="|embedding value|",
            hovertemplate=(
                "Magnitude rank: %{x:,}"
                "<br>"
                "|Value|: %{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    # Add cumulative L2-energy on a secondary y-axis.
    fig.add_trace(
        go.Scattergl(
            x=ranks,
            y=cumulative_energy,
            mode="lines",
            name="Cumulative L2 energy (%)",
            yaxis="y2",
            hovertemplate=(
                "Magnitude rank: %{x:,}"
                "<br>"
                "Cumulative L2 energy: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"Ranked Absolute-Magnitude Profile — "
            f"{target['record_id']}<br>"
            f"<sup>"
            f"Dimensions sorted from largest to smallest "
            f"|value|"
            f"</sup>"
        ),
        xaxis=dict(
            title="Dimension rank by absolute magnitude",
        ),
        yaxis=dict(
            title="Absolute embedding magnitude",
        ),
        yaxis2=dict(
            title="Cumulative L2 energy (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        template="plotly_white",
        height=600,
        hovermode="x unified",
        margin=dict(
            l=85,
            r=90,
            t=100,
            b=85,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_ranked_magnitude.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved ranked magnitude: "
        f"{output}"
    )

    return fig


# ============================================================
# 11. UNIFIED DASHBOARD
# ============================================================

def create_dashboard(
    embedding,
    target,
    stats,
):
    dimensions = np.arange(
        1,
        len(embedding) + 1,
    )

    abs_values = np.abs(embedding)
    central = embedding[
        abs_values <= CENTRAL_RANGE
    ]

    # Ranked magnitude and cumulative L2 energy.
    order = np.argsort(abs_values)[::-1]
    ranked = abs_values[order]
    ranks = np.arange(1, len(ranked) + 1)

    squared = ranked ** 2
    total_squared = max(
        float(np.sum(squared)),
        np.finfo(float).eps,
    )
    cumulative_energy = (
        np.cumsum(squared) / total_squared * 100
    )

    # --------------------------------------------------------
    # Five-row Level-1 dashboard.
    # --------------------------------------------------------

    fig = make_subplots(
        rows=5,
        cols=1,
        vertical_spacing=0.07,
        specs=[
            [{}],
            [{}],
            [{}],
            [{}],
            [{"secondary_y": True}],
        ],
        row_heights=[
            0.17,
            0.27,
            0.19,
            0.19,
            0.18,
        ],
        subplot_titles=(
            "Embedding Fingerprint",
            "Embedding Dimension Profile",
            "Full-Range Value Distribution",
            f"Central Distribution (|value| ≤ {CENTRAL_RANGE:g})",
            "Ranked Absolute Magnitude + Cumulative L2 Energy",
        ),
    )

    # --------------------------------------------------------
    # Row 1 — Heatmap
    # --------------------------------------------------------

    fig.add_trace(
        go.Heatmap(
            z=[embedding],
            x=dimensions,
            y=["Embedding"],
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(
                title="Value",
                x=1.02,
            ),
            hovertemplate=(
                "Dimension: %{x}<br>"
                "Value: %{z:.6f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Row 2 — Dimension profile
    # --------------------------------------------------------

    fig.add_trace(
        go.Scattergl(
            x=dimensions,
            y=embedding,
            mode="lines",
            name="Embedding",
            line=dict(
                width=1.1,
            ),
            hovertemplate=(
                "<b>Dimension %{x}</b><br>"
                "Value: %{y:.6f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # Row 3 — Full distribution
    # --------------------------------------------------------

    fig.add_trace(
        go.Histogram(
            x=embedding,
            nbinsx=FULL_RANGE_NBINS,
            name="Full distribution",
            hovertemplate=(
                "Embedding value: %{x:.4f}"
                "<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    # --------------------------------------------------------
    # Row 4 — Central distribution
    # --------------------------------------------------------

    fig.add_trace(
        go.Histogram(
            x=central,
            nbinsx=CENTRAL_RANGE_NBINS,
            name="Central distribution",
            hovertemplate=(
                "Embedding value: %{x:.6f}"
                "<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
        ),
        row=4,
        col=1,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1,
        row=4,
        col=1,
    )

    # --------------------------------------------------------
    # Row 5 — Ranked magnitude + energy
    # --------------------------------------------------------

    fig.add_trace(
        go.Scattergl(
            x=ranks,
            y=ranked,
            mode="lines",
            name="|value|",
            hovertemplate=(
                "Magnitude rank: %{x:,}"
                "<br>"
                "|Value|: %{y:.6f}"
                "<extra></extra>"
            ),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=ranks,
            y=cumulative_energy,
            mode="lines",
            name="Cumulative L2 energy (%)",
            hovertemplate=(
                "Magnitude rank: %{x:,}"
                "<br>"
                "Cumulative L2 energy: %{y:.2f}%"
                "<extra></extra>"
            ),
        ),
        row=5,
        col=1,
        secondary_y=True,
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    title = (
        f"Carbon Embedding — Level 1 Analysis"
        f"<br>"
        f"<sup>"
        f"{target['record_id']} | "
        f"corpus positions "
        f"{int(target['start']):,}–"
        f"{int(target['end']):,} | "
        f"{len(embedding):,} dimensions"
        f"</sup>"
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=1750,
        showlegend=False,
        margin=dict(
            l=85,
            r=110,
            t=125,
            b=90,
        ),
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    fig.update_xaxes(
        title_text="Embedding dimension",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        showticklabels=False,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Embedding dimension",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Embedding value",
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text="Embedding value",
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text="Dimension count",
        row=3,
        col=1,
    )

    fig.update_xaxes(
        title_text="Embedding value",
        range=[
            -CENTRAL_RANGE,
            CENTRAL_RANGE,
        ],
        row=4,
        col=1,
    )

    fig.update_yaxes(
        title_text="Dimension count",
        row=4,
        col=1,
    )

    fig.update_xaxes(
        title_text="Dimension rank by absolute magnitude",
        row=5,
        col=1,
    )

    fig.update_yaxes(
        title_text="Absolute magnitude",
        row=5,
        col=1,
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Cumulative L2 energy (%)",
        range=[0, 100],
        row=5,
        col=1,
        secondary_y=True,
    )

    # --------------------------------------------------------
    # Statistics annotation
    # --------------------------------------------------------

    annotation_text = (
        f"<b>Embedding diagnostics</b><br>"
        f"Dimensions: "
        f"{stats['dimensions']:,}<br>"
        f"Exact zeros: "
        f"{stats['exact_zero_count']:,} "
        f"({stats['exact_zero_fraction'] * 100:.2f}%)<br>"
        f"Min: "
        f"{stats['min']:.4f}<br>"
        f"Max: "
        f"{stats['max']:.4f}<br>"
        f"Mean: "
        f"{stats['mean']:.4f}<br>"
        f"Std: "
        f"{stats['std']:.4f}<br>"
        f"Median: "
        f"{stats['median']:.4f}<br>"
        f"|x| &gt; 1: "
        f"{stats['abs_gt_1']:,}<br>"
        f"|x| &gt; 5: "
        f"{stats['abs_gt_5']:,}<br>"
        f"|x| &gt; 10: "
        f"{stats['abs_gt_10']:,}<br>"
        f"Top 100 L2 energy: "
        f"{stats['top_100_l2_energy_fraction'] * 100:.2f}%"
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=1.0,
        y=0.52,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor=(
            "rgba(255,255,255,0.92)"
        ),
        bordercolor="gray",
        borderwidth=1,
        font=dict(
            size=12,
        ),
    )

    output = (
        OUTPUT_DIR /
        "embedding_dashboard.html"
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        auto_open=False,
    )

    print(
        f"Saved dashboard     : "
        f"{output}"
    )

    return fig


# ============================================================
# 12. MAIN
# ============================================================

def main():

    target = (
        find_target_embedding()
    )

    embedding = (
        prepare_embedding(target)
    )

    stats, top_rows = (
        calculate_statistics(
            embedding
        )
    )

    dimension_df = (
        create_dimension_dataframe(
            embedding
        )
    )

    save_outputs(
        stats,
        top_rows,
        dimension_df,
    )

    # --------------------------------------------------------
    # 2D visualizations
    # --------------------------------------------------------

    create_embedding_heatmap(
        embedding,
        target,
    )

    create_dimension_profile(
        embedding,
        target,
    )

    create_distribution(
        embedding,
        target,
    )

    create_central_distribution(
        embedding,
        target,
    )

    create_log_magnitude_distribution(
        embedding,
        target,
    )

    create_ranked_magnitude_profile(
        embedding,
        target,
    )

    # --------------------------------------------------------
    # 3D / 4D visualizations
    # --------------------------------------------------------

    create_3d_fingerprint(
        embedding,
        target,
        dimension_df,
    )

    create_4d_fingerprint(
        embedding,
        target,
        dimension_df,
    )

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    create_dashboard(
        embedding,
        target,
        stats,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LEVEL 1 ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print("Output directory:")
    print(
        f"  {OUTPUT_DIR.resolve()}"
    )

    print()
    print("Generated files:")

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            f"  {path.name}"
        )

    print()
    print("Target:")
    print(
        f"  {TARGET_RECORD_ID} "
        f"[{TARGET_START:,} – "
        f"{TARGET_END:,}]"
    )

    print()
    print(
        "Important:"
    )

    print(
        "  The 3D/4D plots are "
        "single-embedding fingerprints."
    )

    print(
        "  They are not PCA/UMAP projections."
    )

    print(
        "  PCA/UMAP becomes appropriate "
        "when multiple sequence embeddings "
        "are available."
    )

    print()
    print(
        "  Distribution analysis includes full-range, "
        "central, log-magnitude, and ranked-magnitude views."
    )

    print()
    print(
        "Note: start/end are Carbon "
        "corpus/tokenization positions, "
        "not genomic coordinates."
    )


if __name__ == "__main__":
    main()