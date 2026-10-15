#!/usr/bin/env python3
"""
Generate eight likelihood visualizations for AINovice2005/carbon-likelihood-stats.

Run from the repository root, or pass --repo-root:
    python scripts/plot_carbon_likelihood.py \
      --repo-root /teamspace/studios/this_studio/hf-genomic-dataset-enrichment \
      --output-dir metrics/derived/likelihood_visualizations

ClickHouse wiring
-----------------
All SQL goes through ``carbon_enrichment.resources.clickhouse.ClickHouseResource``
(a ``clickhouse local`` subprocess wrapper):

- The resource is imported normally. If ``carbon_enrichment`` is not installed,
  ``<repo-root>/carbon-enrichment`` is added to ``sys.path`` and the import is retried
  (or ``pip install -e carbon-enrichment`` once and skip the fallback).
- Sources are registered lazily, only when ``--reuse-metadata`` is not set, avoiding
  unnecessary Hugging Face tree-API calls on cached runs.
- Both the corpus-level likelihood table and the joined metadata table run as
  ``INSERT INTO FUNCTION file(..., Parquet) SELECT ...`` via ``execute_sql`` using atomic
  ``.partial`` file renames, so an interrupted run never leaves a truncated cache.
- The big CPU table is the *left* (streamed) side of the join and the small cohort
  and likelihood tables are the right (build) sides.
- Duplicate interval keys ``(record_id, start, end)`` in ``carbon-likelihood-stats``
  (251,427 rows / 244,046 unique keys) are collapsed in SQL with ``GROUP BY`` in BOTH
  the standalone likelihood export (Plot 01) and the joined metadata export (Plots 02–08).
  Copy counts and max-min score spreads are retained and reported under
  ``duplicate_key_handling`` in ``run_metadata.json``.
- ``--join_use_nulls 1`` is passed through ``extra_args`` so LEFT JOIN misses yield
  ``NULL`` rather than ``0.0`` for ``mean_log_prob``.

Eight outputs:
  01_likelihood_distribution.html
  02a_gc_vs_likelihood.html
  02b_entropy_vs_likelihood.html
  03_likelihood_by_biological_group.html
  04_taxonomic_sunburst.html
  06_likelihood_vs_duplication.html
  07_minimum_likelihood_position.html
  08_summary_correlation_matrix.html

Important interpretation notes:
- Plot 3 uses raw mean_log_prob unless a precomputed Markov baseline column is supplied.
  It must not be described as composition-corrected without that baseline.
- Plot 4 treats taxonomy as an ordered semicolon-delimited lineage, not as guaranteed
  named ranks. It therefore uses a sunburst of lineage components.
- Duplication is exact sequence-hash frequency within the joined common cohort, not
  across the entire 32.4M-row CPU corpus.
- Likelihood keys are aggregated by (record_id, start, end) using AVG(mean_log_prob);
  copy counts and score spread are retained for audit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_REPO_ROOT = Path(
    "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment"
)
DATASET_IDS = {
    "cpu": "AINovice2005/carbon-cpu-enriched-sequences",
    "cohort": "AINovice2005/carbon-common-cohort-taxonomy",
    "likelihood": "AINovice2005/carbon-likelihood-stats",
}
KEY = ["record_id", "start", "end"]
MAX_SCATTER_POINTS = 150_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/derived/likelihood_visualizations"),
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-scatter-points", type=int, default=MAX_SCATTER_POINTS)
    p.add_argument(
        "--minimum-group-size",
        type=int,
        default=20,
        help="Minimum records required to display a biological group in Plot 03.",
    )
    p.add_argument(
        "--correlation-method",
        choices=["spearman", "pearson"],
        default="spearman",
    )
    # ClickHouse resource configuration
    p.add_argument(
        "--clickhouse-binary",
        type=Path,
        default=None,
        help="Path to the clickhouse binary. Default: <repo-root>/carbon-enrichment/"
        "clickhouse if present, else `clickhouse` on PATH, else the resource default.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override ClickHouseConfig threads / download / parsing threads.",
    )
    p.add_argument(
        "--temp-directory",
        type=Path,
        default=None,
        help="Override ClickHouseConfig.temp_directory.",
    )
    # Optional local copies of the Hugging Face datasets
    p.add_argument(
        "--cpu-parquet",
        type=Path,
        default=None,
        help="Optional local Parquet file or directory; directory must contain *.parquet.",
    )
    p.add_argument(
        "--cohort-parquet",
        type=Path,
        default=None,
        help="Optional local Parquet file or directory.",
    )
    p.add_argument(
        "--likelihood-parquet",
        type=Path,
        default=None,
        help="Optional local Parquet file or directory.",
    )
    p.add_argument(
        "--reuse-metadata",
        action="store_true",
        help="Reuse cached likelihood_deduped.parquet and likelihood_joined_metadata.parquet.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Do not write interactive HTML figures.",
    )
    return p.parse_args()


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


def resolve_clickhouse_binary(repo_root: Path, explicit: Path | None) -> str | None:
    if explicit is not None:
        return str(explicit.expanduser())
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


def register_sources(ch, args: argparse.Namespace, names: tuple[str, ...]) -> None:
    """Register only the datasets a step needs (local copy if given, else Hugging Face)."""
    for name in names:
        local = getattr(args, f"{name}_parquet")
        if local is not None:
            ch.register_dataset(name, local)
        else:
            url_glob = (
                f"https://huggingface.co/datasets/{DATASET_IDS[name]}/resolve/main/*.parquet"
            )
            ch.register_hf_dataset(name, url_glob)


def sql_literal_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "''")


def write_sql_parquet(ch, sql: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    ch.execute_sql(
        sql,
        extra_args=["--join_use_nulls", "1"],
    )
    if not partial.exists():
        raise RuntimeError(f"ClickHouse reported success but wrote no file: {partial}")
    os.replace(partial, destination)


def get_table_columns(ch, name: str) -> set[str]:
    """Inspect the schema of a registered ClickHouse dataset to guard optional columns."""
    src = ch.source_expr(name)
    table = ch.query_arrow(f"DESCRIBE TABLE {src}")
    return set(table.column("name").to_pylist())


def build_likelihood_insert_sql(ch, destination: Path, l_cols: set[str]) -> str:
    """
    Collapse duplicate (record_id, start, end) keys in the standalone likelihood
    corpus so corpus-level distributions (Plot 01) are not skewed by repeated keys.
    Table alias `src.` prevents ClickHouse SELECT alias shadowing (Code: 184).
    """
    src = ch.source_expr("likelihood")

    def float_avg(col: str) -> str:
        return (
            f"avg(toFloat64(src.{col}))"
            if col in l_cols
            else "CAST(NULL, 'Nullable(Float64)')"
        )

    def int_round_avg(col: str) -> str:
        return (
            f"CAST(round(avg(toFloat64(src.{col}))), 'Nullable(Int64)')"
            if col in l_cols
            else "CAST(NULL, 'Nullable(Int64)')"
        )

    min_tok_expr = (
        "min(toFloat64(src.min_token_logprob))"
        if "min_token_logprob" in l_cols
        else "CAST(NULL, 'Nullable(Float64)')"
    )
    if "argmin_position" in l_cols and "min_token_logprob" in l_cols:
        argmin_expr = (
            "argMin(toInt64(src.argmin_position), toFloat64(src.min_token_logprob))"
        )
    elif "argmin_position" in l_cols:
        argmin_expr = "min(toInt64(src.argmin_position))"
    else:
        argmin_expr = "CAST(NULL, 'Nullable(Int64)')"

    return f"""
    INSERT INTO FUNCTION file('{sql_literal_path(destination)}', Parquet)
    SELECT
        toString(src.record_id) AS record_id,
        toInt64(src.start) AS start,
        toInt64(src.end) AS end,
        avg(toFloat64(src.mean_log_prob)) AS mean_log_prob,
        count() AS likelihood_copies,
        max(toFloat64(src.mean_log_prob)) - min(toFloat64(src.mean_log_prob)) AS likelihood_spread,
        {float_avg('sum_log_prob')} AS sum_log_prob,
        {float_avg('perplexity')} AS perplexity,
        {int_round_avg('supervised_position_count')} AS supervised_position_count,
        {min_tok_expr} AS min_token_logprob,
        {argmin_expr} AS argmin_position,
        {float_avg('per_token_logprob_std')} AS per_token_logprob_std
    FROM {src} AS src
    GROUP BY src.record_id, src.start, src.end
    """


def build_metadata_insert_sql(
    ch, destination: Path, l_cols: set[str], cpu_cols: set[str]
) -> str:
    """
    Stream the CPU dataset on the left and join against deduplicated cohort and
    likelihood tables on the right. Table aliases (`cpu.`, `c_src.`, `l_src.`)
    prevent ClickHouse SELECT alias shadowing (Code: 184).
    """
    cpu = ch.source_expr("cpu")
    cohort = ch.source_expr("cohort")
    likelihood = ch.source_expr("likelihood")

    def l_float_avg(col: str) -> str:
        return (
            f"avg(toFloat64(l_src.{col}))"
            if col in l_cols
            else "CAST(NULL, 'Nullable(Float64)')"
        )

    def l_int_round_avg(col: str) -> str:
        return (
            f"CAST(round(avg(toFloat64(l_src.{col}))), 'Nullable(Int64)')"
            if col in l_cols
            else "CAST(NULL, 'Nullable(Int64)')"
        )

    min_tok_expr = (
        "min(toFloat64(l_src.min_token_logprob))"
        if "min_token_logprob" in l_cols
        else "CAST(NULL, 'Nullable(Float64)')"
    )
    if "argmin_position" in l_cols and "min_token_logprob" in l_cols:
        argmin_expr = (
            "argMin(toInt64(l_src.argmin_position), toFloat64(l_src.min_token_logprob))"
        )
    elif "argmin_position" in l_cols:
        argmin_expr = "min(toInt64(l_src.argmin_position))"
    else:
        argmin_expr = "CAST(NULL, 'Nullable(Int64)')"

    markov_select = (
        "any(l.markov_residual) AS markov_residual,"
        if "markov_residual" in l_cols
        else ""
    )
    markov_sub = (
        "avg(toFloat64(l_src.markov_residual)) AS markov_residual,"
        if "markov_residual" in l_cols
        else ""
    )

    seq_col = (
        "cpu.strand_normalized_sequence"
        if "strand_normalized_sequence" in cpu_cols
        else ("cpu.sequence" if "sequence" in cpu_cols else None)
    )
    seq_hash_expr = (
        f"any(hex(SHA256(toString({seq_col}))))"
        if seq_col is not None
        else "CAST(NULL, 'Nullable(String)')"
    )

    return f"""
    INSERT INTO FUNCTION file('{sql_literal_path(destination)}', Parquet)
    SELECT
        toString(cpu.record_id) AS record_id,
        toInt64(cpu.start) AS start,
        toInt64(cpu.end) AS end,
        any(c.taxonomy) AS taxonomy,
        avg(toFloat64(cpu.gc_content)) AS gc_content,
        avg(toFloat64(cpu.shannon_entropy)) AS shannon_entropy,
        avg(toFloat64(cpu.sequence_length)) AS sequence_length,
        any(toString(cpu.gene_type)) AS gene_type,
        any(toString(cpu.is_coding_region)) AS is_coding_region,
        {seq_hash_expr} AS sequence_hash,
        any(l.mean_log_prob) AS mean_log_prob,
        {markov_select}
        any(l.likelihood_copies) AS likelihood_copies,
        any(l.likelihood_spread) AS likelihood_spread,
        any(l.sum_log_prob) AS sum_log_prob,
        any(l.perplexity) AS perplexity,
        any(l.supervised_position_count) AS supervised_position_count,
        any(l.min_token_logprob) AS min_token_logprob,
        any(l.argmin_position) AS argmin_position,
        any(l.per_token_logprob_std) AS per_token_logprob_std,
        count() AS cpu_cohort_copies
    FROM {cpu} AS cpu
    INNER JOIN
    (
        SELECT
            toString(c_src.record_id) AS record_id,
            toInt64(c_src.start) AS start,
            toInt64(c_src.end) AS end,
            any(c_src.taxonomy) AS taxonomy
        FROM {cohort} AS c_src
        GROUP BY c_src.record_id, c_src.start, c_src.end
    ) AS c
        ON toString(cpu.record_id) = c.record_id
       AND toInt64(cpu.start) = c.start
       AND toInt64(cpu.end) = c.end
    LEFT JOIN
    (
        SELECT
            toString(l_src.record_id) AS record_id,
            toInt64(l_src.start) AS start,
            toInt64(l_src.end) AS end,
            avg(toFloat64(l_src.mean_log_prob)) AS mean_log_prob,
            {markov_sub}
            count() AS likelihood_copies,
            max(toFloat64(l_src.mean_log_prob)) - min(toFloat64(l_src.mean_log_prob)) AS likelihood_spread,
            {l_float_avg('sum_log_prob')} AS sum_log_prob,
            {l_float_avg('perplexity')} AS perplexity,
            {l_int_round_avg('supervised_position_count')} AS supervised_position_count,
            {min_tok_expr} AS min_token_logprob,
            {argmin_expr} AS argmin_position,
            {l_float_avg('per_token_logprob_std')} AS per_token_logprob_std
        FROM {likelihood} AS l_src
        GROUP BY l_src.record_id, l_src.start, l_src.end
    ) AS l
        ON toString(cpu.record_id) = l.record_id
       AND toInt64(cpu.start) = l.start
       AND toInt64(cpu.end) = l.end
    GROUP BY
        cpu.record_id,
        cpu.start,
        cpu.end
    """


def load_or_build_caches(
    repo_root: Path, args: argparse.Namespace, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    likelihood_cache = output_dir / "likelihood_deduped.parquet"
    metadata_cache = output_dir / "likelihood_joined_metadata.parquet"

    if args.reuse_metadata:
        if not likelihood_cache.exists() or not metadata_cache.exists():
            raise FileNotFoundError(
                "--reuse-metadata set, but one or both caches are missing: "
                f"{likelihood_cache}, {metadata_cache}"
            )
        print(f"[INFO] Reading cached likelihood: {likelihood_cache}")
        print(f"[INFO] Reading cached metadata: {metadata_cache}")
        return pd.read_parquet(likelihood_cache), pd.read_parquet(metadata_cache)

    started = time.time()
    with build_resource(repo_root, args) as ch:
        register_sources(ch, args, ("cpu", "cohort", "likelihood"))

        l_cols = get_table_columns(ch, "likelihood")
        cpu_cols = get_table_columns(ch, "cpu")
        print(f"[INFO] Detected likelihood columns: {sorted(l_cols)}")

        partial_l = likelihood_cache.with_name(likelihood_cache.name + ".partial")
        print("[INFO] Exporting deduplicated corpus-level likelihood table via ClickHouse.")
        write_sql_parquet(
            ch, build_likelihood_insert_sql(ch, partial_l, l_cols), likelihood_cache
        )

        partial_m = metadata_cache.with_name(metadata_cache.name + ".partial")
        print(
            "[INFO] Joining CPU metadata, cohort taxonomy, and likelihood stats via ClickHouse. "
            "The first run may scan the CPU corpus."
        )
        write_sql_parquet(
            ch,
            build_metadata_insert_sql(ch, partial_m, l_cols, cpu_cols),
            metadata_cache,
        )

    likelihood = pd.read_parquet(likelihood_cache)
    metadata = pd.read_parquet(metadata_cache)
    print(
        f"[INFO] Cached {len(likelihood):,} unique likelihood keys and "
        f"{len(metadata):,} joined cohort rows in {time.time() - started:.1f}s."
    )
    return likelihood, metadata


# ---------------------------------------------------------------------------
# Data preparation & helpers
# ---------------------------------------------------------------------------

def write_figure(fig, output_dir: Path, stem: str, no_html: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not no_html:
        fig.write_html(output_dir / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(output_dir / f"{stem}.png", scale=2)
    except Exception as exc:
        print(
            f"[WARN] PNG export skipped for {stem}: {exc}. Install kaleido for PNG export."
        )


def split_taxonomy(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~np.isfinite(df[col]), col] = np.nan
    return df


def prepare_frame(df: pd.DataFrame, name: str, numeric_cols: list[str]) -> pd.DataFrame:
    """Normalize interval keys, enforce strict key uniqueness, and clean numeric columns."""
    df = df.dropna(subset=KEY).copy()
    df["record_id"] = df["record_id"].astype(str)
    for c in ("start", "end"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")

    duplicate_keys = int(df.duplicated(KEY).sum())
    if duplicate_keys:
        raise ValueError(
            f"{name} contains {duplicate_keys:,} duplicate interval keys {KEY}. "
            "Resolve join/aggregation cardinality before plotting."
        )

    df = clean_numeric(df, numeric_cols)
    if "sequence_hash" in df.columns:
        valid_hash = df["sequence_hash"].notna() & (df["sequence_hash"].astype(str) != "")
        counts = df.loc[valid_hash, "sequence_hash"].map(
            df.loc[valid_hash, "sequence_hash"].value_counts()
        )
        df["duplicate_count_in_common_cohort"] = np.nan
        df.loc[valid_hash, "duplicate_count_in_common_cohort"] = counts
    return df.reset_index(drop=True)


def build_dedupe_report(likelihood: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if "likelihood_copies" in likelihood.columns:
        copies = pd.to_numeric(likelihood["likelihood_copies"], errors="coerce").fillna(1)
        spread = pd.to_numeric(likelihood["likelihood_spread"], errors="coerce").fillna(0.0)
        report["corpus_likelihood_total_raw_rows"] = int(copies.sum())
        report["corpus_likelihood_unique_keys"] = int(len(likelihood))
        report["corpus_likelihood_keys_with_duplicates"] = int((copies > 1).sum())
        report["corpus_likelihood_disagreeing_keys"] = int((spread > 0).sum())
        report["corpus_likelihood_max_spread"] = float(spread.max()) if len(spread) else 0.0

    if "likelihood_copies" in metadata.columns:
        m_copies = pd.to_numeric(metadata["likelihood_copies"], errors="coerce")
        m_spread = pd.to_numeric(metadata["likelihood_spread"], errors="coerce")
        report["joined_keys_with_likelihood_duplicates"] = int((m_copies > 1).sum())
        report["joined_keys_missing_likelihood"] = int(m_copies.isna().sum())
        report["joined_likelihood_max_spread"] = (
            float(m_spread.max()) if m_spread.notna().any() else 0.0
        )

    if "cpu_cohort_copies" in metadata.columns:
        c_copies = pd.to_numeric(metadata["cpu_cohort_copies"], errors="coerce").fillna(1)
        report["joined_keys_with_cpu_cohort_duplicates"] = int((c_copies > 1).sum())

    return report


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_01_distribution(likelihood: pd.DataFrame, args: argparse.Namespace, out: Path) -> None:
    d = likelihood.dropna(subset=["mean_log_prob"]).copy()
    if d.empty:
        print("[WARN] Plot 01 skipped: no valid mean_log_prob values.")
        return

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Mean log probability", "Perplexity")
    )
    fig.add_trace(
        go.Histogram(
            x=d["mean_log_prob"],
            nbinsx=100,
            name="Mean log probability",
            marker={"opacity": 0.78},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=d["perplexity"].dropna(),
            nbinsx=100,
            name="Perplexity",
            marker={"opacity": 0.78},
        ),
        row=1,
        col=2,
    )
    for col, series in [
        (1, d["mean_log_prob"].dropna()),
        (2, d["perplexity"].dropna()),
    ]:
        if len(series):
            q1, med, q3 = series.quantile([0.25, 0.5, 0.75])
            fig.add_vline(x=float(med), line_dash="dash", line_width=2, row=1, col=col)
            fig.add_annotation(
                x=float(med),
                y=1,
                yref="y domain",
                xref="x",
                text=f"median={med:.4g}<br>IQR={q1:.4g}–{q3:.4g}<br>n={len(series):,}",
                showarrow=False,
                yanchor="top",
                row=1,
                col=col,
            )
    fig.update_layout(
        title=f"Likelihood distributions across deduplicated keys (n={len(d):,})",
        height=520,
        bargap=0.04,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Mean log probability", row=1, col=1)
    fig.update_xaxes(title_text="Perplexity (lower = more predictable)", row=1, col=2)
    fig.update_yaxes(title_text="Unique interval count", row=1, col=1)
    fig.update_yaxes(title_text="Unique interval count", row=1, col=2)
    write_figure(fig, out, "01_likelihood_distribution", args.no_html)


def plot_02_composition(joined: pd.DataFrame, args: argparse.Namespace, out: Path) -> None:
    # 02a: GC content vs likelihood
    d_gc = joined.dropna(subset=["gc_content", "mean_log_prob"]).copy()
    if not d_gc.empty:
        fig = px.density_heatmap(
            d_gc,
            x="gc_content",
            y="mean_log_prob",
            nbinsx=60,
            nbinsy=60,
            histfunc="count",
            color_continuous_scale="Viridis",
            title=f"GC content vs mean log probability (n={len(d_gc):,})",
            labels={
                "gc_content": "GC content",
                "mean_log_prob": "Mean log probability",
                "count": "Records",
            },
        )
        write_figure(fig, out, "02a_gc_vs_likelihood", args.no_html)
    else:
        print("[WARN] Plot 02a skipped: no joined GC/likelihood values.")

    # 02b: Shannon entropy vs likelihood
    d_ent = joined.dropna(subset=["shannon_entropy", "mean_log_prob"]).copy()
    if not d_ent.empty:
        fig = px.density_heatmap(
            d_ent,
            x="shannon_entropy",
            y="mean_log_prob",
            nbinsx=60,
            nbinsy=60,
            histfunc="count",
            color_continuous_scale="Plasma",
            title=f"Shannon entropy vs mean log probability (n={len(d_ent):,})",
            labels={
                "shannon_entropy": "Shannon entropy",
                "mean_log_prob": "Mean log probability",
                "count": "Records",
            },
        )
        write_figure(fig, out, "02b_entropy_vs_likelihood", args.no_html)
    else:
        print("[WARN] Plot 02b skipped: no joined entropy/likelihood values.")


def plot_03_biological_groups(
    joined: pd.DataFrame, args: argparse.Namespace, out: Path
) -> str | None:
    score_col = "markov_residual" if "markov_residual" in joined.columns else "mean_log_prob"
    score_label = (
        "Markov-corrected mean log-probability residual"
        if score_col == "markov_residual"
        else "Raw mean log probability (not Markov-corrected)"
    )
    group_frames = []
    for group_col in ("gene_type", "is_coding_region"):
        if group_col not in joined.columns:
            continue
        tmp = joined.dropna(subset=[group_col, score_col]).copy()
        tmp[group_col] = tmp[group_col].astype(str).str.strip()
        tmp = tmp[~tmp[group_col].isin(["", "nan", "None", "<NA>"])]
        counts = tmp[group_col].value_counts()
        keep = counts[counts >= args.minimum_group_size].index
        tmp = tmp[tmp[group_col].isin(keep)]
        if not tmp.empty:
            tmp["group_variable"] = group_col
            tmp["group_value"] = tmp[group_col]
            group_frames.append(tmp[["group_variable", "group_value", score_col]])

    if not group_frames:
        print("[WARN] Plot 03 skipped: no biological groups with enough observations.")
        return None

    gd = pd.concat(group_frames, ignore_index=True).rename(columns={score_col: "score"})
    summary = (
        gd.groupby(["group_variable", "group_value"])["score"]
        .agg(
            records="count",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    summary.to_csv(out / "03_biological_group_summary.csv", index=False)

    fig = px.box(
        gd,
        x="group_value",
        y="score",
        color="group_value",
        facet_col="group_variable",
        points=False,
        title=f"Likelihood by biological group — {score_label}",
        labels={
            "group_value": "Group",
            "score": score_label,
            "group_variable": "Grouping variable",
        },
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=35, matches=None)
    write_figure(fig, out, "03_likelihood_by_biological_group", args.no_html)
    return score_label


def plot_04_taxonomic_sunburst(
    joined: pd.DataFrame, args: argparse.Namespace, out: Path
) -> None:
    d = joined.dropna(subset=["taxonomy", "mean_log_prob"]).copy()
    if d.empty:
        print("[WARN] Plot 04 skipped: no taxonomy joined to valid likelihood.")
        return

    d["lineage"] = d["taxonomy"].map(split_taxonomy)
    d = d[d["lineage"].map(len) > 0].copy()
    if d.empty:
        print("[WARN] Plot 04 skipped: all taxonomy strings were empty after splitting.")
        return

    rows = []
    for lineage, score, length in zip(
        d["lineage"], d["mean_log_prob"], d["sequence_length"]
    ):
        bp_val = float(length) if pd.notna(length) else 0.0
        score_val = float(score)
        for depth in range(1, len(lineage) + 1):
            rows.append((tuple(lineage[:depth]), score_val, bp_val))

    agg = pd.DataFrame(rows, columns=["path", "score", "bp"])
    agg["node_id"] = agg["path"].map(lambda x: "root/" + "/".join(x))
    agg["parent"] = agg["path"].map(
        lambda x: "root/" + "/".join(x[:-1]) if len(x) > 1 else "root"
    )
    agg["label"] = agg["path"].map(lambda x: x[-1])

    stats = agg.groupby(["node_id", "parent", "label"], as_index=False).agg(
        median_log_prob=("score", "median"),
        records=("score", "size"),
        total_bp=("bp", "sum"),
        score_std=("score", "std"),
    )
    root = pd.DataFrame(
        [
            {
                "node_id": "root",
                "parent": "",
                "label": "All taxa",
                "median_log_prob": float(d["mean_log_prob"].median()),
                "records": int(len(d)),
                "total_bp": float(d["sequence_length"].fillna(0).sum()),
                "score_std": float(d["mean_log_prob"].std()),
            }
        ]
    )
    sun = pd.concat([root, stats], ignore_index=True)

    fig = go.Figure(
        go.Sunburst(
            ids=sun["node_id"],
            labels=sun["label"],
            parents=sun["parent"],
            values=sun["records"],
            branchvalues="total",
            marker={
                "colors": sun["median_log_prob"],
                "colorscale": "Viridis",
                "colorbar": {"title": "Median mean_log_prob"},
                "showscale": True,
            },
            customdata=np.column_stack(
                [
                    sun["records"].fillna(0),
                    sun["total_bp"].fillna(0),
                    sun["median_log_prob"].fillna(np.nan),
                    sun["score_std"].fillna(np.nan),
                ]
            ),
            hovertemplate=(
                "<b>%{label}</b><br>Records: %{customdata[0]:,.0f}"
                "<br>Total bp: %{customdata[1]:,.0f}"
                "<br>Median mean_log_prob: %{customdata[2]:.4f}"
                "<br>Score SD: %{customdata[3]:.4f}<extra></extra>"
            ),
            maxdepth=6,
        )
    )
    fig.update_layout(
        title="Taxonomic sunburst coloured by median mean log probability",
        margin={"t": 65, "l": 10, "r": 10, "b": 10},
        height=800,
    )
    write_figure(fig, out, "04_taxonomic_sunburst", args.no_html)


def plot_06_duplication(meta: pd.DataFrame, args: argparse.Namespace, out: Path) -> None:
    col = "duplicate_count_in_common_cohort"
    if col not in meta.columns:
        print("[WARN] Plot 06 skipped: sequence hashes are unavailable.")
        return

    dup = meta.dropna(subset=[col, "mean_log_prob"]).copy()
    if dup.empty:
        print("[WARN] Plot 06 skipped: no rows with both sequence_hash and mean_log_prob.")
        return

    dup["copy_category"] = pd.cut(
        dup[col],
        bins=[0, 1, 2, np.inf],
        labels=["1 copy", "2 copies", "3+ copies"],
        include_lowest=True,
    )
    dup = dup.dropna(subset=["copy_category"])

    dup_summary = (
        dup.groupby("copy_category", observed=False)["mean_log_prob"]
        .agg(
            records="count",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    dup_summary.to_csv(out / "06_duplication_summary.csv", index=False)

    fig = px.box(
        dup,
        x="copy_category",
        y="mean_log_prob",
        points=False,
        category_orders={"copy_category": ["1 copy", "2 copies", "3+ copies"]},
        title="Mean log probability by exact-sequence copy count (common cohort)",
        labels={
            "copy_category": "Exact copies in common cohort",
            "mean_log_prob": "Mean log probability",
        },
    )
    fig.update_layout(showlegend=False)
    write_figure(fig, out, "06_likelihood_vs_duplication", args.no_html)


def plot_07_min_position(
    joined: pd.DataFrame, args: argparse.Namespace, out: Path
) -> None:
    if "supervised_position_count" not in joined.columns:
        print("[WARN] Plot 07 skipped: supervised_position_count is unavailable.")
        return

    pos = joined.dropna(subset=["argmin_position", "supervised_position_count"]).copy()
    denom = pos["supervised_position_count"]
    valid = denom.gt(1) & pos["argmin_position"].ge(0) & pos["argmin_position"].lt(denom)
    pos = pos.loc[valid].copy()
    denom = denom.loc[valid]

    pos["normalized_min_position"] = pos["argmin_position"] / (denom - 1.0)
    pos = pos[pos["normalized_min_position"].between(0.0, 1.0)]
    if pos.empty:
        print("[WARN] Plot 07 skipped: no in-bound argmin_position rows.")
        return

    fig = px.histogram(
        pos,
        x="normalized_min_position",
        nbins=60,
        histnorm="probability density",
        title=f"Normalized position of minimum token log-probability (n={len(pos):,})",
        labels={
            "normalized_min_position": "Relative position (0 = start, 1 = end)",
            "density": "Density",
        },
    )
    fig.update_layout(bargap=0.03)
    write_figure(fig, out, "07_minimum_likelihood_position", args.no_html)
    pos[KEY + ["argmin_position", "supervised_position_count", "normalized_min_position"]].to_parquet(
        out / "07_normalized_min_position.parquet", index=False
    )


def plot_08_correlation_matrix(
    joined: pd.DataFrame, args: argparse.Namespace, out: Path
) -> list[str]:
    corr_cols = [
        "mean_log_prob",
        "perplexity",
        "sum_log_prob",
        "min_token_logprob",
        "per_token_logprob_std",
        "gc_content",
        "shannon_entropy",
        "sequence_length",
    ]
    available = [c for c in corr_cols if c in joined.columns]
    corr_data = joined[available].apply(pd.to_numeric, errors="coerce")
    available = [
        c
        for c in available
        if corr_data[c].notna().sum() >= 3 and corr_data[c].nunique(dropna=True) > 1
    ]
    corr_data = corr_data[available]

    if len(available) < 2:
        print("[WARN] Plot 08 skipped: fewer than two usable numeric variables.")
        return available

    corr = corr_data.corr(method=args.correlation_method, min_periods=3)
    corr.to_csv(out / "08_summary_correlation_matrix.csv")

    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=f"Summary-statistic correlation matrix ({args.correlation_method.title()})",
        labels={"color": "Correlation"},
    )
    fig.update_layout(height=720, width=900)
    write_figure(fig, out, "08_summary_correlation_matrix", args.no_html)
    return available


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    out = args.output_dir.expanduser()
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    likelihood_raw, metadata_raw = load_or_build_caches(repo_root, args, out)

    numeric_likelihood = [
        "mean_log_prob",
        "sum_log_prob",
        "perplexity",
        "supervised_position_count",
        "min_token_logprob",
        "argmin_position",
        "per_token_logprob_std",
        "likelihood_copies",
        "likelihood_spread",
    ]
    numeric_metadata = numeric_likelihood + [
        "gc_content",
        "shannon_entropy",
        "sequence_length",
        "cpu_cohort_copies",
    ]

    likelihood = prepare_frame(likelihood_raw, "deduplicated likelihood", numeric_likelihood)
    metadata = prepare_frame(metadata_raw, "joined metadata", numeric_metadata)
    dedupe_report = build_dedupe_report(likelihood, metadata)

    joined = metadata.dropna(subset=["mean_log_prob"]).copy()

    plot_01_distribution(likelihood, args, out)
    plot_02_composition(joined, args, out)
    plot_03_label = plot_03_biological_groups(joined, args, out)
    plot_04_taxonomic_sunburst(joined, args, out)
    plot_06_duplication(metadata, args, out)
    plot_07_min_position(joined, args, out)
    corr_cols = plot_08_correlation_matrix(joined, args, out)

    summary = {
        "likelihood_unique_keys": int(len(likelihood)),
        "metadata_rows": int(len(metadata)),
        "joined_rows_with_likelihood": int(len(joined)),
        "duplicate_key_handling": dedupe_report,
        "correlation_columns": corr_cols,
        "correlation_method": args.correlation_method,
        "plot_03_score": plot_03_label,
        "elapsed_seconds": round(time.time() - started, 2),
        "notes": [
            "Duplicate (record_id, start, end) keys in carbon-likelihood-stats are collapsed via SQL GROUP BY in both standalone and joined caches.",
            "Plot 03 uses raw mean_log_prob unless a precomputed markov_residual column is present.",
            "Plot 04 treats semicolon-delimited taxonomy strings as ordered lineage prefixes rather than named taxonomic ranks.",
            "Plot 06 duplication counts are exact SHA-256 sequence-hash frequencies within the common cohort only.",
        ],
    }
    (out / "run_metadata.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Outputs written to: {out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())