#!/usr/bin/env python3
"""
Create three interactive biological sunburst charts for the Carbon genomic
enrichment project.

Charts
------
01_taxonomy_embedding_dispersion.html
    Taxonomic lineage -> median cosine distance to each taxonomic group's
    embedding centroid. Segment area is interval count; color is median
    within-group embedding dispersion.

02_taxonomy_cpu_composition.html
    Taxonomic lineage -> GC-content band -> coding status. Segment area is
    interval count; color is median Shannon entropy (or another CPU metric).

03_cpu_vs_gpu_profiles.html
    Taxonomic lineage -> CPU composition profile -> GPU likelihood profile.
    Segment area is interval count; color is a selected GPU metric.

The script follows the attached plot_carbon_likelihood.py conventions:
- local Parquet metadata cache and embeddings read through ClickHouseResource
  from remote Hugging Face Parquet shards in ArrowStream batches
- interval key: (record_id, start, end)
- semicolon-delimited taxonomy treated as an ordered lineage, not assumed
  to contain named ranks
- Plotly interactive HTML output and optional PNG output

Expected inputs
---------------
--metadata-parquet:
    The joined metadata cache produced by plot_carbon_likelihood.py, normally
    metrics/derived/likelihood_visualizations/likelihood_joined_metadata.parquet.
    It should contain record_id, start, end, taxonomy, gc_content,
    shannon_entropy, sequence_length, is_coding_region (or gene_type), and
    mean_log_prob where available.

--embedding-url-glob:
    Remote Hugging Face Parquet URL glob, read by ClickHouseResource through
    ArrowStream batches. No manual dataset download is performed.

No cluster assignments are invented. Chart 03 uses quantile-binned CPU and
likelihood profiles as an interpretable CPU-vs-GPU comparison. If an embedding
cluster column already exists in the metadata input, it can be selected with
--gpu-profile-column.

Example
-------
python scripts/plot_carbon_sunbursts.py \
    --repo-root /teamspace/studios/this_studio/hf-genomic-dataset-enrichment \
    --metadata-parquet metrics/derived/likelihood_visualizations/likelihood_joined_metadata.parquet \
    --output-dir metrics/derived/sunburst_visualizations

The embedding source is read directly from remote Hugging Face Parquet shards
through ClickHouseResource. Python retains centroid sums and per-row scalar
dispersion results, not the full embedding matrix.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


KEY = ["record_id", "start", "end"]
DEFAULT_REPO_ROOT = Path(
    "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment"
)
INVALID_TAXA = {"", "nan", "none", "<na>", "unknown", "unclassified"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create three interactive Carbon biological sunburst charts."
    )
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument(
        "--metadata-parquet",
        type=Path,
        default=Path(
            "metrics/derived/likelihood_visualizations/"
            "likelihood_joined_metadata.parquet"
        ),
        help="Joined CPU/cohort/likelihood metadata Parquet file.",
    )
    p.add_argument(
        "--embedding-url-glob",
        default=(
            "https://huggingface.co/datasets/AINovice2005/carbon-embeddings/"
            "resolve/main/*.parquet"
        ),
        help="Remote Hugging Face Parquet URL glob read by ClickHouse; no manual Hub download.",
    )
    p.add_argument(
        "--embedding-column",
        default="embedding",
        help="Vector column in the remote embeddings Parquet dataset.",
    )
    p.add_argument(
        "--clickhouse-binary",
        type=Path,
        default=None,
        help="Optional ClickHouse binary override; otherwise use ClickHouseConfig default.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional ClickHouse thread limit.",
    )
    p.add_argument(
        "--temp-directory",
        type=Path,
        default=None,
        help="Optional ClickHouse temporary directory.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/derived/sunburst_visualizations"),
    )
    p.add_argument("--minimum-group-size", type=int, default=20)
    p.add_argument("--max-taxonomy-depth", type=int, default=5)
    p.add_argument("--gc-bins", type=int, default=5)
    p.add_argument(
        "--cpu-color-column",
        choices=["shannon_entropy", "gc_content", "sequence_length", "gc_skew"],
        default="shannon_entropy",
        help="CPU metric used to color chart 02.",
    )
    p.add_argument(
        "--gpu-profile-column",
        default="mean_log_prob",
        help=(
            "GPU-derived numeric column used in chart 03. If this column is "
            "absent, the script falls back to mean_log_prob."
        ),
    )
    p.add_argument("--profile-bins", type=int, default=3)
    p.add_argument(
        "--stream-batch-rows",
        type=int,
        default=8192,
        help="ArrowStream batch size requested from ClickHouse.",
    )
    p.add_argument("--no-html", action="store_true")
    p.add_argument(
        "--png", action="store_true", help="Also export PNGs (requires kaleido)."
    )
    return p.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def clean_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required interval-key columns: {missing}"
        )
    out = df.dropna(subset=KEY).copy()
    out["record_id"] = out["record_id"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out = out.dropna(subset=["start", "end"])
    out["start"] = out["start"].astype("int64")
    out["end"] = out["end"].astype("int64")
    duplicate_count = int(out.duplicated(KEY).sum())
    if duplicate_count:
        print(
            f"[WARN] {name}: collapsing {duplicate_count:,} duplicate interval keys "
            "by keeping the first row."
        )
        out = out.drop_duplicates(KEY, keep="first")
    return out


def numeric_clean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~np.isfinite(df[col]), col] = np.nan
    return df


def split_taxonomy(value: Any, max_depth: int) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parts = [p.strip() for p in str(value).split(";")]
    parts = [p for p in parts if p.lower() not in INVALID_TAXA]
    # Remove repeated adjacent lineage components while preserving order.
    compact: list[str] = []
    for part in parts:
        if not compact or part != compact[-1]:
            compact.append(part)
    return compact[:max_depth]


def safe_component(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    text = str(value).strip()
    return text if text and text.lower() not in INVALID_TAXA else "Unknown"


def ensure_lineage(df: pd.DataFrame, max_depth: int) -> pd.DataFrame:
    if "taxonomy" not in df.columns:
        raise ValueError("Metadata must contain a 'taxonomy' column.")
    out = df.copy()
    out["_lineage"] = out["taxonomy"].map(lambda x: split_taxonomy(x, max_depth))
    out = out[out["_lineage"].map(bool)].copy()
    if out.empty:
        raise ValueError("No usable taxonomy lineages found after parsing taxonomy.")
    return out


def add_missing_parent_rows(
    d: pd.DataFrame,
    path_cols: list[str],
    metric_col: str,
    color_label: str,
    title: str,
    output: Path,
    min_group_size: int,
    no_html: bool,
    png: bool,
    hover_extra: list[str] | None = None,
) -> None:
    """Aggregate a hierarchical path into Plotly's ids/parents sunburst format."""
    hover_extra = hover_extra or []
    d = d.dropna(subset=[metric_col]).copy()
    if d.empty:
        print(f"[WARN] Skipping {output.stem}: no valid values for {metric_col}.")
        return

    # Count rows and metric summaries at every prefix, ensuring that each
    # parent value is >= the sum of its immediate children.
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    root_metrics: list[float] = []
    root_extras: dict[str, list[float]] = {c: [] for c in hover_extra}

    for row in d.to_dict("records"):
        path = tuple(safe_component(row[c]) for c in path_cols)
        if not path or path[0] == "Unknown":
            continue

        metric_val = float(row[metric_col])
        root_metrics.append(metric_val)
        for c in hover_extra:
            val = row.get(c)
            if val is not None and pd.notna(val):
                root_extras[c].append(float(val))

        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            rec = records.setdefault(
                prefix,
                {
                    "metrics": [],
                    "n": 0,
                    "extras": {c: [] for c in hover_extra},
                },
            )
            rec["n"] += 1
            rec["metrics"].append(metric_val)
            for c in hover_extra:
                val = row.get(c)
                if val is not None and pd.notna(val):
                    rec["extras"][c].append(float(val))

    if not records or not root_metrics:
        print(f"[WARN] Skipping {output.stem}: no valid taxonomy paths.")
        return

    candidate_rows = []
    for path, stats in records.items():
        n = stats["n"]
        if len(path) > 1 and n < min_group_size:
            continue
        node_id = "root/" + "/".join(path)
        parent = "root/" + "/".join(path[:-1]) if len(path) > 1 else "root"
        vals = np.asarray(stats["metrics"], dtype=float)
        row_dict: dict[str, Any] = {
            "node_id": node_id,
            "parent": parent,
            "label": path[-1],
            "records": n,
            "metric": float(np.median(vals)),
            "metric_q1": float(np.quantile(vals, 0.25)),
            "metric_q3": float(np.quantile(vals, 0.75)),
            "_depth": len(path),
        }
        for c, values in stats["extras"].items():
            row_dict[c] = float(np.median(values)) if values else np.nan
        candidate_rows.append(row_dict)

    # Walk top-down by depth so that any node whose parent/ancestor was pruned
    # is also cleanly excluded.
    candidate_rows.sort(key=lambda r: r["_depth"])
    valid_ids = {"root"}
    rows = []
    for r in candidate_rows:
        if r["parent"] in valid_ids:
            valid_ids.add(r["node_id"])
            r.pop("_depth", None)
            rows.append(r)

    if not rows:
        print(f"[WARN] Skipping {output.stem}: all groups below minimum size.")
        return

    root_arr = np.asarray(root_metrics, dtype=float)
    root: dict[str, Any] = {
        "node_id": "root",
        "parent": "",
        "label": "All intervals",
        "records": int(root_arr.size),
        "metric": float(np.median(root_arr)),
        "metric_q1": float(np.quantile(root_arr, 0.25)),
        "metric_q3": float(np.quantile(root_arr, 0.75)),
    }
    for c in hover_extra:
        vals = root_extras[c]
        root[c] = float(np.median(vals)) if vals else np.nan

    sun = pd.DataFrame([root] + rows)

    # Plotly branchvalues='total' expects each parent to be >= the sum of its
    # children. Use observed interval counts, and suppress tiny children rather
    # than changing their biological counts.
    fig = go.Figure(
        go.Sunburst(
            ids=sun["node_id"],
            labels=sun["label"],
            parents=sun["parent"],
            values=sun["records"],
            branchvalues="total",
            sort=False,
            maxdepth=len(path_cols),
            insidetextorientation="radial",
            marker={
                "colors": sun["metric"],
                "colorscale": "Viridis",
                "cmin": float(np.nanquantile(sun["metric"], 0.02)),
                "cmax": float(np.nanquantile(sun["metric"], 0.98)),
                "colorbar": {"title": color_label},
                "showscale": True,
                "line": {"color": "white", "width": 0.6},
            },
            customdata=np.column_stack(
                [
                    sun["records"].to_numpy(),
                    sun["metric"].to_numpy(),
                    sun["metric_q1"].to_numpy(),
                    sun["metric_q3"].to_numpy(),
                    *[
                        sun[c].to_numpy()
                        if c in sun.columns
                        else np.full(len(sun), np.nan)
                        for c in hover_extra
                    ],
                ]
            ),
            hovertemplate=(
                "<b>%{label}</b><br>Intervals: %{customdata[0]:,.0f}"
                f"<br>{color_label}: " + "%{customdata[1]:.4f}"
                "<br>Q1–Q3: %{customdata[2]:.4f}–%{customdata[3]:.4f}"
                + "".join(
                    f"<br>{c}: %{{customdata[{4 + i}]:.4f}}"
                    for i, c in enumerate(hover_extra)
                )
                + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        height=850,
        margin={"t": 80, "l": 15, "r": 25, "b": 15},
        uniformtext={"minsize": 9, "mode": "hide"},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not no_html:
        fig.write_html(output.with_suffix(".html"), include_plotlyjs="cdn")
        print(f"[DONE] {output.with_suffix('.html')}")
    if png:
        try:
            fig.write_image(output.with_suffix(".png"), scale=2)
            print(f"[DONE] {output.with_suffix('.png')}")
        except Exception as exc:
            print(
                f"[WARN] PNG export skipped for {output.stem}: {exc}. Install kaleido."
            )


# ---------------------------------------------------------------------------
# ClickHouse resource wiring
# ---------------------------------------------------------------------------


def import_clickhouse_resource(repo_root: Path):
    """Import the project's ClickHouse resource, falling back to the repo checkout."""
    try:
        from carbon_enrichment.resources.clickhouse import (
            ClickHouseConfig,
            ClickHouseResource,
        )

        return ClickHouseConfig, ClickHouseResource
    except ImportError as first_error:
        package_root = repo_root / "carbon-enrichment"
        if not (package_root / "carbon_enrichment").is_dir():
            raise RuntimeError(
                "carbon_enrichment is not importable and "
                f"{package_root / 'carbon_enrichment'} does not exist. "
                "Run `pip install -e carbon-enrichment` or fix --repo-root."
            ) from first_error
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        try:
            from carbon_enrichment.resources.clickhouse import (
                ClickHouseConfig,
                ClickHouseResource,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"Found {package_root} but importing carbon_enrichment.resources.clickhouse "
                f"still failed ({exc}). Check the project environment's dependencies."
            ) from exc
        return ClickHouseConfig, ClickHouseResource


def resolve_clickhouse_binary(
    repo_root: Path, explicit: Path | None
) -> str | None:
    if explicit is not None:
        return str(Path(explicit).expanduser())
    local = repo_root / "carbon-enrichment" / "clickhouse"
    if local.is_file():
        return str(local)
    return shutil.which("clickhouse")


def build_resource(repo_root: Path, args: argparse.Namespace):
    ClickHouseConfig, ClickHouseResource = import_clickhouse_resource(repo_root)
    config_kwargs: dict[str, Any] = {}
    binary = resolve_clickhouse_binary(repo_root, args.clickhouse_binary)
    if binary is not None:
        config_kwargs["binary_path"] = binary
    if args.threads is not None:
        config_kwargs["threads"] = args.threads
        config_kwargs["max_download_threads"] = args.threads
        config_kwargs["max_parsing_threads"] = args.threads
    if args.temp_directory is not None:
        config_kwargs["temp_directory"] = args.temp_directory
    return ClickHouseResource(ClickHouseConfig(**config_kwargs))

def stream_embedding_join(
    ch,
    metadata_path: Path,
    embedding_url_glob: str,
    embedding_column: str,
    batch_rows: int,
):
    """Yield Arrow batches from a ClickHouse join against remote Hub Parquet."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", embedding_column):
        raise ValueError(f"Invalid embedding column name: {embedding_column!r}")
    ch.register_dataset("sunburst_metadata", metadata_path)
    ch.register_hf_dataset(
        "sunburst_embeddings",
        embedding_url_glob,
        revision="main",
        pattern="*.parquet",
    )
    metadata_source = ch.source_expr("sunburst_metadata")
    embedding_source = ch.source_expr("sunburst_embeddings")
    sql = f"""
        SELECT
            toString(m.record_id) AS record_id,
            toInt64(m.start) AS start,
            toInt64(m.end) AS end,
            m.taxonomy AS taxonomy,
            e.`{embedding_column}` AS embedding
        FROM {embedding_source} AS e
        INNER JOIN (
            SELECT
                toString(src.record_id) AS record_id,
                toInt64(src.start) AS start,
                toInt64(src.end) AS end,
                any(toString(src.taxonomy)) AS taxonomy
            FROM {metadata_source} AS src
            WHERE src.taxonomy IS NOT NULL AND toString(src.taxonomy) != ''
            GROUP BY
                toString(src.record_id),
                toInt64(src.start),
                toInt64(src.end)
        ) AS m
            ON toString(e.record_id) = m.record_id
           AND toInt64(e.start) = m.start
           AND toInt64(e.end) = m.end
    """
    # Validate selected columns without collecting data rows.
    schema_table = ch.query_arrow(f"SELECT * FROM ({sql}) LIMIT 0")
    expected = {"record_id", "start", "end", "taxonomy", "embedding"}
    missing = expected.difference(schema_table.column_names)
    if missing:
        raise ValueError(
            f"ClickHouse embedding join missing columns: {sorted(missing)}"
        )
    yield from ch.stream_arrow_batches(
        f"SELECT * FROM ({sql}) SETTINGS max_block_size = {int(batch_rows)}"
    )


def _vector_from_arrow(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return None
    return vector / norm


def embedding_dispersion_stream(
    ch, metadata_path: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, int]:
    """Two-pass streaming cosine dispersion; never materializes the full embedding matrix."""
    prefix_sums: dict[tuple[str, ...], np.ndarray] = {}
    prefix_counts: dict[tuple[str, ...], int] = {}
    dimension: int | None = None
    matched_rows = 0

    print(
        "[INFO] Pass 1/2: streaming remote embeddings through ClickHouse to build centroids."
    )
    for batch in stream_embedding_join(
        ch,
        metadata_path,
        args.embedding_url_glob,
        args.embedding_column,
        args.stream_batch_rows,
    ):
        for row in batch.to_pylist():
            lineage = split_taxonomy(row.get("taxonomy"), args.max_taxonomy_depth)
            vector = _vector_from_arrow(row.get("embedding"))
            if not lineage or vector is None:
                continue
            if dimension is None:
                dimension = int(vector.size)
                print(f"[INFO] Embedding dimension: {dimension:,}")
            if vector.size != dimension:
                continue
            matched_rows += 1
            for depth in range(1, len(lineage) + 1):
                prefix = tuple(lineage[:depth])
                if prefix not in prefix_sums:
                    prefix_sums[prefix] = np.zeros(dimension, dtype=np.float64)
                    prefix_counts[prefix] = 0
                prefix_sums[prefix] += vector
                prefix_counts[prefix] += 1

    if matched_rows == 0 or dimension is None:
        raise ValueError(
            "No valid embedding rows matched metadata on (record_id, start, end)."
        )
    centroids: dict[tuple[str, ...], np.ndarray] = {}
    for prefix, vector_sum in prefix_sums.items():
        if prefix_counts[prefix] < args.minimum_group_size:
            continue
        norm = float(np.linalg.norm(vector_sum))
        if norm > 0:
            centroids[prefix] = (vector_sum / norm).astype(np.float32)
    if not centroids:
        raise ValueError(
            "No taxonomic groups met --minimum-group-size. Lower that threshold."
        )
    print(
        f"[INFO] First pass: {matched_rows:,} matched rows; "
        f"{len(centroids):,} qualifying taxonomy-prefix centroids."
    )

    print("[INFO] Pass 2/2: streaming again to calculate per-interval dispersion.")
    output_rows: list[dict[str, Any]] = []
    for batch in stream_embedding_join(
        ch,
        metadata_path,
        args.embedding_url_glob,
        args.embedding_column,
        args.stream_batch_rows,
    ):
        for row in batch.to_pylist():
            lineage = split_taxonomy(row.get("taxonomy"), args.max_taxonomy_depth)
            vector = _vector_from_arrow(row.get("embedding"))
            if not lineage or vector is None or vector.size != dimension:
                continue
            centroid = None
            for depth in range(len(lineage), 0, -1):
                centroid = centroids.get(tuple(lineage[:depth]))
                if centroid is not None:
                    break
            if centroid is None:
                continue
            distance = 1.0 - float(np.clip(np.dot(vector, centroid), -1.0, 1.0))
            output_rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "taxonomy": str(row["taxonomy"]),
                    "embedding_dispersion": distance,
                    "_lineage": lineage,
                }
            )

    dispersion = pd.DataFrame(output_rows)
    if dispersion.empty:
        raise ValueError(
            "No per-interval dispersion values were produced in the second pass."
        )
    dupes = int(dispersion.duplicated(KEY).sum())
    if dupes:
        print(
            f"[WARN] Streaming join returned {dupes:,} duplicate interval keys; keeping first."
        )
        dispersion = dispersion.drop_duplicates(KEY, keep="first").copy()
    print(f"[INFO] Dispersion rows retained: {len(dispersion):,}")
    return dispersion, dimension


def add_cpu_chart(
    metadata: pd.DataFrame, args: argparse.Namespace, output_dir: Path
) -> None:
    d = metadata.copy()
    d = numeric_clean(
        d, ["gc_content", "shannon_entropy", "sequence_length", "gc_skew"]
    )
    d = ensure_lineage(d, args.max_taxonomy_depth)
    metric = args.cpu_color_column
    if metric not in d.columns:
        raise ValueError(f"CPU color metric '{metric}' is absent from metadata.")
    d = d.dropna(subset=[metric, "gc_content"]).copy()
    if d.empty:
        raise ValueError("No rows with both GC content and selected CPU metric.")

    # Quantile bins are corpus-relative; duplicate edges are removed safely.
    labels = ["Low GC", "Lower-mid GC", "Mid GC", "Upper-mid GC", "High GC"]
    q = min(max(2, args.gc_bins), len(labels))
    try:
        # Let qcut drop duplicate edges first; then map the actual resulting
        # intervals to labels so low-cardinality GC values do not break bins.
        bands = pd.qcut(d["gc_content"], q=q, duplicates="drop")
        categories = list(bands.cat.categories)
        band_labels = labels[: len(categories)]
        mapping = {category: band_labels[i] for i, category in enumerate(categories)}
        d["gc_band"] = bands.map(mapping).astype("object")
    except (ValueError, IndexError):
        d["gc_band"] = "GC unbinned"

    coding_col = (
        "is_coding_region" if "is_coding_region" in d.columns else "gene_type"
    )
    d["coding_status"] = (
        d[coding_col].map(safe_component) if coding_col in d.columns else "Unknown"
    )
    d["gc_band"] = d["gc_band"].replace("nan", "Unknown").map(safe_component)
    d["coding_status"] = d["coding_status"].map(safe_component)
    d["_metric"] = d[metric]

    # Build named path columns to retain the taxonomy lineage depth and append
    # CPU composition categories as the final rings.
    depth = max(map(len, d["_lineage"]))
    for i in range(depth):
        d[f"path_{i + 1}"] = d["_lineage"].map(
            lambda x, idx=i: x[idx] if len(x) > idx else "Unclassified"
        )
    path_cols = [f"path_{i + 1}" for i in range(depth)] + [
        "gc_band",
        "coding_status",
    ]
    add_missing_parent_rows(
        d,
        path_cols,
        "_metric",
        metric,
        f"Taxonomy × CPU sequence composition (color: median {metric})",
        output_dir / "02_taxonomy_cpu_composition",
        args.minimum_group_size,
        args.no_html,
        args.png,
    )


def add_cpu_gpu_chart(
    metadata: pd.DataFrame, args: argparse.Namespace, output_dir: Path
) -> None:
    d = metadata.copy()
    numeric = [
        "gc_content",
        "shannon_entropy",
        "sequence_length",
        "mean_log_prob",
        "perplexity",
        "min_token_logprob",
        "per_token_logprob_std",
    ]
    d = numeric_clean(d, numeric)
    d = ensure_lineage(d, args.max_taxonomy_depth)
    gpu_metric = args.gpu_profile_column
    if gpu_metric not in d.columns:
        if "mean_log_prob" in d.columns:
            print(
                f"[WARN] GPU profile '{gpu_metric}' missing; using mean_log_prob."
            )
            gpu_metric = "mean_log_prob"
        else:
            raise ValueError(f"GPU profile column '{gpu_metric}' is absent.")
    cpu_metric = "gc_content"
    d = d.dropna(subset=[cpu_metric, gpu_metric]).copy()
    if d.empty:
        raise ValueError(
            "No rows have both CPU GC content and the selected GPU metric."
        )

    bins = max(2, min(args.profile_bins, 5))
    try:
        d["cpu_profile"] = pd.qcut(
            d[cpu_metric], q=bins, duplicates="drop"
        ).astype(str)
        d["gpu_profile"] = pd.qcut(
            d[gpu_metric], q=bins, duplicates="drop"
        ).astype(str)
    except ValueError:
        d["cpu_profile"] = "CPU profile"
        d["gpu_profile"] = "GPU profile"

    d["cpu_profile"] = "GC " + d["cpu_profile"].astype(str)
    d["gpu_profile"] = "GPU " + d["gpu_profile"].astype(str)
    d["_metric"] = d[gpu_metric]
    depth = max(map(len, d["_lineage"]))
    for i in range(depth):
        d[f"path_{i + 1}"] = d["_lineage"].map(
            lambda x, idx=i: x[idx] if len(x) > idx else "Unclassified"
        )
    path_cols = [f"path_{i + 1}" for i in range(depth)] + [
        "cpu_profile",
        "gpu_profile",
    ]
    add_missing_parent_rows(
        d,
        path_cols,
        "_metric",
        f"Median {gpu_metric}",
        f"CPU GC profile × GPU likelihood profile (color: median {gpu_metric})",
        output_dir / "03_cpu_vs_gpu_profiles",
        args.minimum_group_size,
        args.no_html,
        args.png,
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    metadata_path = resolve_path(args.metadata_parquet, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata cache not found: {metadata_path}\n"
            "Run plot_carbon_likelihood.py first, or pass --metadata-parquet."
        )

    print(f"[INFO] Reading joined metadata: {metadata_path}")
    metadata = pd.read_parquet(metadata_path)
    metadata = clean_keys(metadata, "joined metadata")
    numeric_clean(
        metadata,
        [
            "gc_content",
            "shannon_entropy",
            "sequence_length",
            "gc_skew",
            "mean_log_prob",
            "perplexity",
            "min_token_logprob",
            "per_token_logprob_std",
        ],
    )
    metadata = metadata.drop_duplicates(KEY, keep="first").copy()
    metadata = ensure_lineage(metadata, args.max_taxonomy_depth)

    print(f"[INFO] Metadata rows with taxonomy: {len(metadata):,}")
    ch = build_resource(repo_root, args)
    try:
        dispersion, embedding_dimension = embedding_dispersion_stream(
            ch, metadata_path, args
        )
    finally:
        ch.close()

    # Chart 01: taxonomy × embedding structure.
    depth = max(map(len, dispersion["_lineage"]))
    for i in range(depth):
        dispersion[f"path_{i + 1}"] = dispersion["_lineage"].map(
            lambda x, idx=i: x[idx] if len(x) > idx else "Unclassified"
        )
    add_missing_parent_rows(
        dispersion,
        [f"path_{i + 1}" for i in range(depth)],
        "embedding_dispersion",
        "Median cosine distance to taxonomic centroid",
        "Taxonomy × embedding structure: within-group embedding dispersion",
        output_dir / "01_taxonomy_embedding_dispersion",
        args.minimum_group_size,
        args.no_html,
        args.png,
    )

    # Chart 02: CPU-only taxonomy × sequence composition.
    add_cpu_chart(metadata, args, output_dir)

    # Chart 03: CPU composition profile × GPU likelihood profile.
    add_cpu_gpu_chart(metadata, args, output_dir)

    summary = {
        "metadata_rows": int(len(metadata)),
        "embedding_rows_matched": int(len(dispersion)),
        "embedding_dimension": int(embedding_dimension),
        "minimum_group_size": args.minimum_group_size,
        "taxonomy_depth": args.max_taxonomy_depth,
        "cpu_color_column": args.cpu_color_column,
        "gpu_profile_column": args.gpu_profile_column,
        "outputs": [
            "01_taxonomy_embedding_dispersion.html",
            "02_taxonomy_cpu_composition.html",
            "03_cpu_vs_gpu_profiles.html",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
        "notes": [
            "Taxonomy strings are treated as ordered semicolon-delimited lineage components, not guaranteed named ranks.",
            "Embedding dispersion is cosine distance to the deepest adequately sampled taxonomic ancestor centroid.",
            "Embedding Parquet shards are read remotely through ClickHouseResource in ArrowStream batches; no manual Hub dataset download is performed.",
            "Chart 02 GC bands and chart 03 CPU/GPU profiles use corpus-relative quantile bins.",
            "Chart 03 is a descriptive cross-tabulation, not proof that GPU embeddings add biological information.",
        ],
    }
    (output_dir / "sunburst_run_metadata.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[DONE] Outputs written to: {output_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())