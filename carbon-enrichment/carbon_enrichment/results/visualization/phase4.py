#!/usr/bin/env python3
"""
Generate the nine Carbon embedding visualizations.

Run from the repository root, or pass --repo-root:
    python scripts/plot_carbon_embeddings.py \
      --repo-root /teamspace/studios/this_studio/hf-genomic-dataset-enrichment \
      --output-dir metrics/derived/embedding_visualizations \
      --knn-csv metrics/derived/phase4/knn_neighbors.csv \
      --sample-size 20000

ClickHouse wiring
-----------------
All SQL goes through ``carbon_enrichment.resources.clickhouse.ClickHouseResource``
(a ``clickhouse local`` subprocess wrapper):

- The resource is imported normally. If ``carbon_enrichment`` is not installed,
  ``<repo-root>/carbon-enrichment`` is added to ``sys.path`` and the import is retried
  (or ``pip install -e carbon-enrichment`` once and skip the fallback).
- Sources are registered lazily, only for the datasets a run actually needs:
  ``--reuse-metadata`` never touches the cpu/cohort/likelihood datasets, so it
  makes no Hugging Face tree-API calls for them.
- The metadata join runs as ``INSERT INTO FUNCTION file(..., Parquet) SELECT ...``
  via ``execute_sql``, so ClickHouse writes the cache itself instead of piping the
  whole result through ``query_arrow`` into Python memory. The file is written to a
  ``.partial`` name and renamed, so an interrupted run cannot leave a truncated
  cache for ``--reuse-metadata`` to pick up.
- The big CPU table is the *left* side of the join and the small cohort table the
  right (build) side, so ClickHouse hashes the cohort rather than 32M sequences.
- Duplicate likelihood keys are collapsed in SQL with avg() per key (see
  build_metadata_insert_sql); what was collapsed is reported under
  duplicate_key_handling in run_metadata.json. CPU duplicates do not overlap the
  cohort, so the CPU side is not deduplicated.
- ``--join_use_nulls 1`` is passed through ``extra_args``. Without it a LEFT JOIN miss
  yields 0.0 instead of NULL for ``mean_log_prob`` and plot 6 would colour
  unmatched rows as if they had a real log-probability of zero.
- Embeddings are streamed with ``stream_arrow_batches``; the resource is used as a
  context manager so a missing binary fails immediately with its own message.

The script:
- Joins CPU metadata, common-cohort taxonomy, and likelihood statistics.
- Streams embeddings in Arrow batches for dimension diagnostics and to collect
  only a reproducible sample for PCA/UMAP.
- Reuses an existing kNN CSV; it does not recompute vector search. The CSV is
  read once and feeds both plot 4 and plot 9.
- Caches the joined metadata as Parquet so subsequent runs can skip the
  expensive source join with --reuse-metadata.

Important:
- The first metadata extraction can scan the large CPU Parquet corpus once.
  If possible, register a local copy of the CPU Parquet shards with --cpu-parquet.
- Plot 4 reports lineage-prefix agreement, not held-out classification accuracy.
  Taxonomy strings do not carry explicit rank labels, so plot 9 uses shared lineage
  depth rather than guessing genus/family/order ranks.
- kNN edges whose query and neighbor are the same interval are excluded from plots
  4 and 9 (a self-match is trivially in the same lineage); the count is reported
  in run_metadata.json.
- Duplicate counts are exact-sequence SHA-256 groups within the full common
  cohort (computed before sampling), not the full 32.4M-row corpus.
- Whitening/UMAP is exploratory. The script whitens the retained PCA components,
  not all 3,072 raw dimensions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


DEFAULT_REPO_ROOT = Path(
    "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment"
)
DATASET_IDS = {
    "embeddings": "AINovice2005/carbon-embeddings",
    "cpu": "AINovice2005/carbon-cpu-enriched-sequences",
    "cohort": "AINovice2005/carbon-common-cohort-taxonomy",
    "likelihood": "AINovice2005/carbon-likelihood-stats",
}
KEY = ["record_id", "start", "end"]
MAX_LINEAGE_DEPTH = 12
STATS_BLOCK_ROWS = 8192
MAX_SCATTER_POINTS = 100_000
KNN_CHUNK_ROWS = 200_000
DISTANCE_CATEGORIES = [
    "0 shared", "1-2 shared", "3-4 shared", "5-6 shared", "7-8 shared", "9+ shared",
]
DISTANCE_SAMPLE_PER_CATEGORY = 50_000


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
        default=Path("metrics/derived/embedding_visualizations"),
    )
    p.add_argument("--sample-size", type=int, default=20000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--diagnostic-dim", type=int, default=3,
                   help="Zero-based embedding coordinate used for the channel diagnostic.")
    p.add_argument("--umap-neighbors", type=int, default=25)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--knn-csv", type=Path,
                   default=Path("metrics/derived/phase4/knn_neighbors.csv"))
    # ClickHouse resource configuration
    p.add_argument("--clickhouse-binary", type=Path, default=None,
                   help="Path to the clickhouse binary. Default: <repo-root>/carbon-enrichment/"
                        "clickhouse if present, else `clickhouse` on PATH, else the resource default.")
    p.add_argument("--threads", type=int, default=None,
                   help="Override ClickHouseConfig threads / download / parsing threads.")
    p.add_argument("--temp-directory", type=Path, default=None,
                   help="Override ClickHouseConfig.temp_directory.")
    # Optional local copies of the Hugging Face datasets
    p.add_argument("--cpu-parquet", type=Path, default=None,
                   help="Optional local Parquet file or directory; directory must contain *.parquet.")
    p.add_argument("--embeddings-parquet", type=Path, default=None,
                   help="Optional local Parquet file or directory.")
    p.add_argument("--cohort-parquet", type=Path, default=None,
                   help="Optional local Parquet file or directory.")
    p.add_argument("--likelihood-parquet", type=Path, default=None,
                   help="Optional local Parquet file or directory.")
    p.add_argument("--reuse-metadata", action="store_true",
                   help="Reuse output-dir/common_cohort_metadata.parquet.")
    p.add_argument("--skip-knn", action="store_true",
                   help="Skip plots 4 and 9 if the kNN CSV is unavailable or too large.")
    p.add_argument("--no-html", action="store_true",
                   help="Do not write interactive HTML figures.")
    return p.parse_args()


def require_dependencies() -> None:
    """Fail before the expensive passes, not after them."""
    try:
        import umap  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install umap-learn to generate UMAP figures: pip install umap-learn"
        ) from exc


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
    # hf_token is intentionally not set: the resource falls back to $HF_TOKEN.
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


def build_metadata_insert_sql(ch, destination: Path) -> str:
    cohort = ch.source_expr("cohort")
    cpu = ch.source_expr("cpu")
    likelihood = ch.source_expr("likelihood")
    # CPU is the streamed (left) side; the small cohort and likelihood tables are
    # the right-hand build sides.
    #
    # Likelihood has repeated keys (251,427 rows / 244,046 keys; 10 keys disagree,
    # max spread 0.0093), so it is collapsed to one row per key with avg(). The copy
    # count and max-min spread are kept as audit columns.
    # CPU also has repeated keys (364), but none fall inside the cohort (checked with
    # check_duplicate_keys.py), so no CPU dedupe is needed; prepare_metadata still
    # raises if that ever stops being true.
    return f"""
    INSERT INTO FUNCTION file('{sql_literal_path(destination)}', Parquet)
    SELECT
        cpu.record_id AS record_id,
        cpu.start AS start,
        cpu.end AS end,
        c.taxonomy AS taxonomy,
        cpu.gc_content AS gc_content,
        cpu.shannon_entropy AS shannon_entropy,
        cpu.sequence_length AS sequence_length,
        cpu.gene_type AS gene_type,
        cpu.is_coding_region AS is_coding_region,
        cpu.kmer_frequency_vector AS kmer_frequency_vector,
        hex(SHA256(cpu.strand_normalized_sequence)) AS sequence_hash,
        l.mean_log_prob_avg AS mean_log_prob,
        l.likelihood_copies AS likelihood_copies,
        l.likelihood_spread AS likelihood_spread
    FROM {cpu} AS cpu
    INNER JOIN {cohort} AS c
        ON cpu.record_id = c.record_id
       AND cpu.start = c.start
       AND cpu.end = c.end
    LEFT JOIN
    (
        SELECT
            record_id,
            start,
            end,
            avg(mean_log_prob) AS mean_log_prob_avg,
            count() AS likelihood_copies,
            max(mean_log_prob) - min(mean_log_prob) AS likelihood_spread
        FROM {likelihood}
        GROUP BY record_id, start, end
    ) AS l
        ON cpu.record_id = l.record_id
       AND cpu.start = l.start
       AND cpu.end = l.end
    """


def load_or_build_metadata(ch, args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    cache = output_dir / "common_cohort_metadata.parquet"
    if args.reuse_metadata:
        if not cache.exists():
            raise FileNotFoundError(f"--reuse-metadata set but cache does not exist: {cache}")
        print(f"[INFO] Reading metadata cache: {cache}")
        return pd.read_parquet(cache)

    register_sources(ch, args, ("cpu", "cohort", "likelihood"))
    partial = cache.with_name(cache.name + ".partial")
    partial.unlink(missing_ok=True)
    print("[INFO] Building common-cohort metadata through ClickHouse. "
          "The first run may scan the CPU corpus.")
    ch.execute_sql(
        build_metadata_insert_sql(ch, partial),
        extra_args=["--join_use_nulls", "1"],
    )
    if not partial.exists():
        raise RuntimeError(f"ClickHouse reported success but wrote no file: {partial}")
    os.replace(partial, cache)
    df = pd.read_parquet(cache)
    print(f"[INFO] Cached {len(df):,} aligned metadata rows to {cache}")
    return df


# ---------------------------------------------------------------------------
# Taxonomy / metadata helpers
# ---------------------------------------------------------------------------

def write_figure(fig, output_dir: Path, stem: str, no_html: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not no_html:
        fig.write_html(output_dir / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(output_dir / f"{stem}.png", scale=2)
    except Exception as exc:
        print(f"[WARN] PNG export skipped for {stem}: {exc}. Install kaleido for PNG export.")


def key_tuple(record_id: Any, start: Any, end: Any) -> tuple[str, int, int]:
    return (str(record_id), int(start), int(end))


def split_taxonomy(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def kingdom_label(taxonomy: Any) -> str | None:
    # Published taxonomy strings are semicolon-delimited lineages, not a
    # rank-annotated taxonomy table. For this eukaryotic corpus, the second
    # lineage component is commonly the kingdom (e.g. Eukaryota;Fungi).
    parts = split_taxonomy(taxonomy)
    if not parts:
        return None
    candidate = parts[1] if len(parts) > 1 and parts[0].lower() == "eukaryota" else parts[0]
    return candidate or None


def shared_prefix_len(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    depth = 0
    for x, y in zip(a, b):
        if x != y:
            break
        depth += 1
    return depth


def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise keys/types, derive kingdom, and count duplicates over the FULL cohort."""
    df = df.dropna(subset=["record_id", "start", "end"]).copy()
    df["record_id"] = df["record_id"].astype(str)
    for c in ("start", "end"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    if "sequence_length" in df:
        df["sequence_length"] = pd.to_numeric(df["sequence_length"], errors="coerce")
    duplicate_keys = int(df.duplicated(KEY).sum())
    if duplicate_keys:
        raise ValueError(
            f"Metadata contains {duplicate_keys:,} duplicate interval keys. "
            "Resolve join cardinality before plotting."
        )
    df["kingdom"] = df["taxonomy"].map(kingdom_label)
    # Counted before sampling so the plot-8 colour scale reflects the whole cohort,
    # not just the ~20k projection sample.
    if "sequence_hash" in df:
        counts = df["sequence_hash"].map(df["sequence_hash"].value_counts())
        df["duplicate_count_in_common_cohort"] = counts.fillna(1).astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Embedding stream
# ---------------------------------------------------------------------------

class RunningMoments:
    """Per-column mean and variance via Chan et al.'s parallel update."""

    def __init__(self) -> None:
        self.n = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, block: np.ndarray) -> None:
        bn = block.shape[0]
        if bn == 0:
            return
        bmean = block.mean(axis=0)
        bm2 = np.square(block - bmean).sum(axis=0)
        if self.mean is None or self.m2 is None:
            self.n, self.mean, self.m2 = bn, bmean, bm2
            return
        delta = bmean - self.mean
        total = self.n + bn
        self.mean = self.mean + delta * (bn / total)
        self.m2 = self.m2 + bm2 + np.square(delta) * (self.n * bn / total)
        self.n = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.m2 is None:
            raise RuntimeError("No embedding rows were read.")
        return self.mean, self.m2 / max(self.n - 1, 1)


def embedding_matrix(column: Any) -> np.ndarray:
    """Convert an Arrow list<float> column to an (n, dim) array without Python lists."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if column.null_count:
        raise ValueError("Embedding column contains nulls.")
    n = len(column)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)
    if hasattr(column, "value_lengths"):
        lengths = column.value_lengths().to_numpy(zero_copy_only=False)
        if lengths.min() != lengths.max():
            raise ValueError("Embedding batch has ragged vector lengths.")
    flat = column.flatten().to_numpy(zero_copy_only=False)
    if flat.size % n:
        raise ValueError(f"Cannot reshape {flat.size} values into {n} equal rows.")
    return flat.reshape(n, -1)


@dataclass
class EmbeddingPass:
    diagnostic: pd.DataFrame
    norms: np.ndarray
    abs_dim: np.ndarray
    norm_table: pd.DataFrame
    sample_meta: pd.DataFrame
    sample_vectors: np.ndarray


def stream_embedding_pass(ch, args: argparse.Namespace, metadata: pd.DataFrame) -> EmbeddingPass:
    register_sources(ch, args, ("embeddings",))
    emb_src = ch.source_expr("embeddings")

    sample_meta = metadata.sample(
        n=min(args.sample_size, len(metadata)), random_state=args.random_state
    )
    ordered_keys = [
        key_tuple(*row) for row in sample_meta[KEY].itertuples(index=False, name=None)
    ]
    sample_keys = set(ordered_keys)
    selected: dict[tuple[str, int, int], np.ndarray] = {}

    moments = RunningMoments()
    norm_frames: list[pd.DataFrame] = []
    all_norms: list[np.ndarray] = []
    all_abs_dim: list[np.ndarray] = []
    dim = args.diagnostic_dim
    embedding_dim = 0

    sql = f"""
    SELECT record_id, start, end, embedding, embedding_norm
    FROM {emb_src}
    """
    print("[INFO] Streaming embeddings in Arrow batches for dimension diagnostics.")
    for batch in ch.stream_arrow_batches(sql):
        def col(name: str):
            return batch.column(batch.schema.get_field_index(name))

        matrix = embedding_matrix(col("embedding"))
        if matrix.shape[0] == 0:
            continue
        embedding_dim = matrix.shape[1]
        if dim < 0 or dim >= embedding_dim:
            raise ValueError(f"--diagnostic-dim {dim} outside embedding dimension {embedding_dim}")

        ids = [str(v) for v in col("record_id").to_pylist()]
        starts = [int(v) for v in col("start").to_pylist()]
        ends = [int(v) for v in col("end").to_pylist()]
        norms = np.asarray(col("embedding_norm").to_numpy(zero_copy_only=False), dtype=np.float64)

        # Block the float64 statistics so a 65k-row batch of 3,072-d vectors does
        # not materialise multi-GB temporaries.
        for lo in range(0, matrix.shape[0], STATS_BLOCK_ROWS):
            moments.update(matrix[lo:lo + STATS_BLOCK_ROWS].astype(np.float64))

        all_norms.append(norms)
        all_abs_dim.append(np.abs(matrix[:, dim].astype(np.float64)))
        norm_frames.append(pd.DataFrame({
            "record_id": ids, "start": starts, "end": ends, "embedding_norm": norms,
        }))
        for i, key in enumerate(zip(ids, starts, ends)):
            if key in sample_keys and key not in selected:
                selected[key] = matrix[i].astype(np.float32, copy=True)

    mean, variance = moments.finalize()
    diagnostic = pd.DataFrame({
        "dimension_index": np.arange(len(mean)),
        "mean": mean,
        "variance": variance,
        "std": np.sqrt(np.maximum(variance, 0)),
    })
    norms = np.concatenate(all_norms)
    abs_dim = np.concatenate(all_abs_dim)

    norm_table = pd.concat(norm_frames, ignore_index=True)
    duplicated = int(norm_table.duplicated(KEY).sum())
    if duplicated:
        print(f"[WARN] {duplicated:,} duplicate embedding keys; keeping the first of each.")
        norm_table = norm_table.drop_duplicates(KEY, keep="first")

    print(f"[INFO] Read {len(norms):,} embeddings; selected {len(selected):,} for projections.")
    keep = np.fromiter((k in selected for k in ordered_keys), dtype=bool, count=len(ordered_keys))
    if not keep.all():
        print(f"[WARN] {int((~keep).sum()):,} sampled cohort keys had no embedding row.")
    # Row order follows the seeded sample, never set iteration order, so PCA/UMAP
    # inputs are identical between runs.
    sampled_meta = sample_meta.loc[keep].reset_index(drop=True)
    kept_keys = [k for k, flag in zip(ordered_keys, keep) if flag]
    vec_matrix = (
        np.vstack([selected[k] for k in kept_keys])
        if kept_keys else np.empty((0, embedding_dim), dtype=np.float32)
    )
    return EmbeddingPass(diagnostic, norms, abs_dim, norm_table, sampled_meta, vec_matrix)


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def add_projection_columns(meta: pd.DataFrame, vectors: np.ndarray, args: argparse.Namespace):
    if len(meta) != len(vectors):
        raise ValueError("Metadata and embedding sample lengths differ.")
    valid = np.isfinite(vectors).all(axis=1)
    meta = meta.loc[valid].reset_index(drop=True).copy()
    vectors = vectors[valid]
    if len(vectors) < 20:
        raise ValueError("Fewer than 20 valid vectors remain for projection.")
    n_components = min(args.pca_components, vectors.shape[1], len(vectors) - 1)
    print(f"[INFO] Fitting PCA/UMAP on {len(vectors):,} sampled embeddings, {n_components} PCA components.")
    pca_raw = PCA(n_components=n_components, svd_solver="randomized",
                  random_state=args.random_state)
    raw_pc = pca_raw.fit_transform(vectors)
    pca_white = PCA(n_components=n_components, whiten=True, svd_solver="randomized",
                    random_state=args.random_state)
    white_pc = pca_white.fit_transform(vectors)
    import umap
    n_neighbors = min(args.umap_neighbors, len(vectors) - 1)

    def make_reducer():
        return umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=args.umap_min_dist,
            metric="cosine", random_state=args.random_state, n_jobs=1,
        )

    meta["umap_raw_x"], meta["umap_raw_y"] = make_reducer().fit_transform(raw_pc).T
    meta["umap_white_x"], meta["umap_white_y"] = make_reducer().fit_transform(white_pc).T
    # The leading two components of the raw PCA are the embedding PCA plane.
    meta["embedding_pca_x"] = raw_pc[:, 0]
    meta["embedding_pca_y"] = raw_pc[:, 1]
    return meta, vectors


# ---------------------------------------------------------------------------
# kNN edge scan (feeds plots 4 and 9 in a single pass over the CSV)
# ---------------------------------------------------------------------------

@dataclass
class KnnScan:
    agreement: pd.DataFrame
    distances: pd.DataFrame | None
    category_counts: pd.DataFrame | None
    edges_read: int
    edges_missing_key: int
    self_edges: int
    edges_without_lineage: int
    valid_edges: int


def scan_knn(args: argparse.Namespace, metadata: pd.DataFrame) -> KnnScan | None:
    if args.skip_knn or not args.knn_csv.exists():
        print(f"[WARN] Plots 4 and 9 skipped: kNN CSV not found or --skip-knn set: {args.knn_csv}")
        return None

    labels = metadata[KEY + ["taxonomy"]].reset_index(drop=True)
    index = pd.MultiIndex.from_frame(labels[KEY])
    codes, uniques = pd.factorize(labels["taxonomy"])
    parts_by_code = [tuple(split_taxonomy(u)) for u in uniques]
    codes = codes.astype(np.int64)
    if parts_by_code:
        nonempty = np.array([len(p) > 0 for p in parts_by_code], dtype=bool)
        codes = np.where((codes >= 0) & nonempty[np.maximum(codes, 0)], codes, -1)
    else:
        codes = np.full(len(codes), -1, dtype=np.int64)
    n_codes = max(len(parts_by_code), 1)
    max_len = max((len(p) for p in parts_by_code), default=1)
    max_depth = max(1, min(MAX_LINEAGE_DEPTH, max_len))

    total = np.zeros(MAX_LINEAGE_DEPTH, dtype=np.int64)
    match = np.zeros(MAX_LINEAGE_DEPTH, dtype=np.int64)
    pair_cache: dict[int, tuple[int, int]] = {}
    rng = np.random.default_rng(args.random_state)
    reservoirs: dict[int, tuple[np.ndarray, np.ndarray]] = {
        c: (np.empty(0), np.empty(0)) for c in range(len(DISTANCE_CATEGORIES))
    }
    category_counts = np.zeros(len(DISTANCE_CATEGORIES), dtype=np.int64)
    saw_distance = False
    edges_read = edges_missing_key = self_edges = edges_without_lineage = valid_edges = 0

    wanted = {
        "query_record_id", "query_start", "query_end",
        "neighbor_record_id", "neighbor_start", "neighbor_end", "cosine_distance",
    }
    print(f"[INFO] Scanning kNN edges once for plots 4 and 9: {args.knn_csv}")
    reader = pd.read_csv(
        args.knn_csv,
        usecols=lambda c: c in wanted,
        dtype={"query_record_id": str, "neighbor_record_id": str},
        chunksize=KNN_CHUNK_ROWS,
    )
    for chunk in reader:
        edges_read += len(chunk)
        qmi = pd.MultiIndex.from_arrays([
            chunk["query_record_id"].to_numpy(),
            chunk["query_start"].to_numpy(np.int64),
            chunk["query_end"].to_numpy(np.int64),
        ])
        nmi = pd.MultiIndex.from_arrays([
            chunk["neighbor_record_id"].to_numpy(),
            chunk["neighbor_start"].to_numpy(np.int64),
            chunk["neighbor_end"].to_numpy(np.int64),
        ])
        qpos = index.get_indexer(qmi)
        npos = index.get_indexer(nmi)
        both = (qpos >= 0) & (npos >= 0)
        edges_missing_key += int((~both).sum())
        is_self = both & (qpos == npos)
        self_edges += int(is_self.sum())
        use = both & ~is_self
        use_idx = np.flatnonzero(use)
        qc = codes[qpos[use_idx]]
        nc = codes[npos[use_idx]]
        has_lineage = (qc >= 0) & (nc >= 0)
        edges_without_lineage += int((~has_lineage).sum())
        use_idx, qc, nc = use_idx[has_lineage], qc[has_lineage], nc[has_lineage]
        if use_idx.size == 0:
            continue
        valid_edges += int(use_idx.size)

        pair = qc * n_codes + nc
        uniq, inverse = np.unique(pair, return_inverse=True)
        shared_u = np.empty(len(uniq), dtype=np.int64)
        minlen_u = np.empty(len(uniq), dtype=np.int64)
        for j, p in enumerate(uniq.tolist()):
            cached = pair_cache.get(p)
            if cached is None:
                a, b = parts_by_code[p // n_codes], parts_by_code[p % n_codes]
                cached = (shared_prefix_len(a, b), min(len(a), len(b)))
                pair_cache[p] = cached
            shared_u[j], minlen_u[j] = cached
        shared = shared_u[inverse.reshape(-1)]
        minlen = minlen_u[inverse.reshape(-1)]

        # Descriptive neighborhood consistency, not held-out classification accuracy.
        for k in range(1, MAX_LINEAGE_DEPTH + 1):
            total[k - 1] += int((minlen >= k).sum())
            match[k - 1] += int((shared >= k).sum())

        if "cosine_distance" in chunk.columns:
            saw_distance = True
            dist = chunk["cosine_distance"].to_numpy(np.float64)[use_idx]
            finite = np.isfinite(dist)
            sh, dv = shared[finite], dist[finite]
            cat = np.where(sh == 0, 0, np.minimum((sh + 1) // 2, len(DISTANCE_CATEGORIES) - 1))
            for c in range(len(DISTANCE_CATEGORIES)):
                vals = dv[cat == c]
                if vals.size == 0:
                    continue
                category_counts[c] += vals.size
                # Uniform sample without replacement in bounded memory: keep the
                # cap smallest of i.i.d. uniform keys per category.
                keys = np.concatenate([reservoirs[c][0], rng.random(vals.size)])
                values = np.concatenate([reservoirs[c][1], vals])
                if keys.size > DISTANCE_SAMPLE_PER_CATEGORY:
                    keep = np.argpartition(keys, DISTANCE_SAMPLE_PER_CATEGORY)[:DISTANCE_SAMPLE_PER_CATEGORY]
                    keys, values = keys[keep], values[keep]
                reservoirs[c] = (keys, values)

    depths = np.arange(1, max_depth + 1)
    agreement = pd.DataFrame({
        "lineage_prefix_depth": depths,
        "neighbor_prefix_agreement": np.divide(
            match[:max_depth], total[:max_depth],
            out=np.full(max_depth, np.nan, dtype=float), where=total[:max_depth] > 0,
        ),
        "valid_edges": total[:max_depth],
        "interpretation": "descriptive lineage-prefix agreement; not held-out classification accuracy",
    })

    distances = counts_df = None
    if saw_distance:
        frames = [
            pd.DataFrame({
                "shared_lineage_depth": DISTANCE_CATEGORIES[c],
                "cosine_distance": reservoirs[c][1].astype(np.float32),
            })
            for c in range(len(DISTANCE_CATEGORIES)) if reservoirs[c][1].size
        ]
        distances = pd.concat(frames, ignore_index=True) if frames else None
        counts_df = pd.DataFrame({
            "shared_lineage_depth": DISTANCE_CATEGORIES,
            "edges": category_counts,
            "sampled_rows": [int(reservoirs[c][1].size) for c in range(len(DISTANCE_CATEGORIES))],
        })

    return KnnScan(
        agreement=agreement, distances=distances, category_counts=counts_df,
        edges_read=edges_read, edges_missing_key=edges_missing_key, self_edges=self_edges,
        edges_without_lineage=edges_without_lineage, valid_edges=valid_edges,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_1_channel(diagnostic, norms, abs_dim, args, out):
    diagnostic.to_csv(out / "plot01_dimension_diagnostics.csv", index=False)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Per-dimension mean", "Per-dimension variance", f"Norm vs |dimension {args.diagnostic_dim}|"),
    )
    fig.add_trace(go.Scattergl(
        x=diagnostic.dimension_index, y=diagnostic["mean"], mode="lines",
        name="Mean", line={"width": 1},
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=diagnostic.dimension_index, y=diagnostic["variance"], mode="lines",
        name="Variance", line={"width": 1},
    ), row=1, col=2)
    count = min(len(norms), 60000)
    rng = np.random.default_rng(args.random_state)
    ix = rng.choice(len(norms), size=count, replace=False) if count < len(norms) else np.arange(len(norms))
    fig.add_trace(go.Scattergl(
        x=abs_dim[ix], y=norms[ix], mode="markers", name="Embedding rows",
        marker={"size": 3, "opacity": 0.35},
    ), row=1, col=3)
    fig.update_layout(title="Embedding channel diagnostic", height=500, showlegend=False)
    fig.update_xaxes(title_text="Dimension index", row=1, col=1)
    fig.update_xaxes(title_text="Dimension index", row=1, col=2)
    fig.update_xaxes(title_text=f"|embedding[{args.diagnostic_dim}]|", row=1, col=3)
    fig.update_yaxes(title_text="Mean", row=1, col=1)
    fig.update_yaxes(title_text="Variance", row=1, col=2)
    fig.update_yaxes(title_text="Embedding L2 norm", row=1, col=3)
    write_figure(fig, out, "plot01_outlier_channel", args.no_html)


def plot_2_umap(meta, args, out):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Raw embedding PCA → UMAP", "PCA-whitened embedding → UMAP"))
    color = meta["kingdom"].fillna("Unknown") if "kingdom" in meta else meta["taxonomy"].fillna("Unknown")
    for col, x, y, name in [
        (1, "umap_raw_x", "umap_raw_y", "Raw"),
        (2, "umap_white_x", "umap_white_y", "Whitened"),
    ]:
        fig.add_trace(go.Scattergl(
            x=meta[x], y=meta[y], mode="markers",
            marker={"size": 4, "opacity": 0.65, "color": pd.factorize(color)[0], "showscale": False},
            text=color, customdata=np.column_stack([meta["record_id"], meta["start"], meta["end"]]),
            hovertemplate="Taxon=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            name=name,
        ), row=1, col=col)
    fig.update_layout(title="UMAP before and after PCA whitening (exploratory)", height=600, showlegend=False)
    write_figure(fig, out, "plot02_umap_raw_vs_whitened", args.no_html)


def plot_3_norm_length(metadata, args, out):
    d = metadata.dropna(subset=["embedding_norm", "sequence_length"]).copy()
    if d.empty:
        print("[WARN] Plot 3 skipped: no joined embedding_norm/sequence_length.")
        return
    corr = d[["embedding_norm", "sequence_length"]].corr(method="spearman").iloc[0, 1]
    d[["record_id", "start", "end", "sequence_length", "embedding_norm"]].to_parquet(
        out / "plot03_norm_vs_length.parquet", index=False
    )
    shown = d.sample(n=MAX_SCATTER_POINTS, random_state=args.random_state) if len(d) > MAX_SCATTER_POINTS else d
    fig = px.scatter(
        shown, x="sequence_length", y="embedding_norm",
        color="kingdom" if "kingdom" in shown else None,
        opacity=0.55, render_mode="webgl",
        title=f"Embedding norm vs sequence length (Spearman ρ={corr:.3f}, n={len(d):,}; {len(shown):,} shown)",
        labels={"sequence_length": "Sequence length (nt)", "embedding_norm": "Embedding L2 norm"},
        hover_data=["record_id", "start", "end"],
    )
    write_figure(fig, out, "plot03_embedding_norm_vs_length", args.no_html)


def plot_4_knn(scan: KnnScan | None, args, out):
    if scan is None:
        return
    scan.agreement.to_csv(out / "plot04_knn_lineage_agreement.csv", index=False)
    fig = px.line(
        scan.agreement, x="lineage_prefix_depth", y="neighbor_prefix_agreement", markers=True,
        title="Taxonomic lineage-prefix agreement among existing kNN neighbors",
        labels={
            "lineage_prefix_depth": "Number of shared lineage components tested",
            "neighbor_prefix_agreement": "Fraction of neighbor edges sharing the full prefix",
        },
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    write_figure(fig, out, "plot04_knn_lineage_agreement", args.no_html)


def plot_5_kmer_vs_embedding_pca(meta, vectors, args, out):
    if "kmer_frequency_vector" not in meta:
        print("[WARN] Plot 5 skipped: kmer_frequency_vector unavailable.")
        return
    valid_idx = []
    kmer_rows = []
    for i, val in enumerate(meta["kmer_frequency_vector"]):
        try:
            arr = np.asarray(val, dtype=np.float32)
            if arr.shape == (64,) and np.isfinite(arr).all():
                valid_idx.append(i)
                kmer_rows.append(arr)
        except Exception:
            continue
    if len(valid_idx) < 3:
        print("[WARN] Plot 5 skipped: insufficient valid 64-element k-mer vectors.")
        return
    km = np.vstack(kmer_rows)
    emb = vectors[valid_idx]
    km_xy = PCA(n_components=2, random_state=args.random_state).fit_transform(km)
    emb_xy = PCA(n_components=2, svd_solver="randomized", random_state=args.random_state).fit_transform(emb)
    d = meta.iloc[valid_idx].copy()
    d["kmer_pc1"], d["kmer_pc2"] = km_xy[:, 0], km_xy[:, 1]
    d["embedding_pc1"], d["embedding_pc2"] = emb_xy[:, 0], emb_xy[:, 1]
    color = d["kingdom"].fillna("Unknown") if "kingdom" in d else d["taxonomy"].fillna("Unknown")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Trinucleotide frequency PCA", "Carbon embedding PCA"))
    fig.add_trace(go.Scattergl(
        x=d.kmer_pc1, y=d.kmer_pc2, mode="markers",
        marker={"size": 4, "opacity": 0.6, "color": pd.factorize(color)[0]},
        text=color, name="3-mer PCA",
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=d.embedding_pc1, y=d.embedding_pc2, mode="markers",
        marker={"size": 4, "opacity": 0.6, "color": pd.factorize(color)[0]},
        text=color, name="Embedding PCA",
    ), row=1, col=2)
    fig.update_layout(title="Trinucleotide composition vs learned embedding (same intervals)", height=600, showlegend=False)
    write_figure(fig, out, "plot05_trinucleotide_vs_embedding_pca", args.no_html)
    d[KEY + ["kmer_pc1", "kmer_pc2", "embedding_pc1", "embedding_pc2"]].to_parquet(
        out / "plot05_pca_coordinates.parquet", index=False
    )


def plot_6_likelihood(meta, args, out):
    d = meta.dropna(subset=["mean_log_prob"]).copy()
    if d.empty:
        print("[WARN] Plot 6 skipped: mean_log_prob not available after join.")
        return
    fig = px.scatter(
        d, x="umap_raw_x", y="umap_raw_y", color="mean_log_prob",
        color_continuous_scale="Viridis", opacity=0.7, render_mode="webgl",
        title="Embedding UMAP coloured by mean log-probability",
        labels={"mean_log_prob": "Mean log-probability", "umap_raw_x": "UMAP 1", "umap_raw_y": "UMAP 2"},
        hover_data=["record_id", "start", "end", "taxonomy"],
    )
    write_figure(fig, out, "plot06_umap_by_likelihood", args.no_html)


def plot_7_composition(meta, args, out):
    d = meta.dropna(subset=["gc_content", "shannon_entropy"]).copy()
    if d.empty:
        print("[WARN] Plot 7 skipped: composition metadata not available.")
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=("GC content", "Shannon entropy"))
    fig.add_trace(go.Scattergl(
        x=d.umap_raw_x, y=d.umap_raw_y, mode="markers",
        marker={"size": 4, "opacity": 0.7, "color": d.gc_content, "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": "GC content", "x": 0.46}},
        text=d.taxonomy, name="GC",
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=d.umap_raw_x, y=d.umap_raw_y, mode="markers",
        marker={"size": 4, "opacity": 0.7, "color": d.shannon_entropy, "colorscale": "Plasma", "showscale": True,
                "colorbar": {"title": "Shannon entropy", "x": 1.02}},
        text=d.taxonomy, name="Entropy",
    ), row=1, col=2)
    fig.update_layout(title="Embedding UMAP coloured by sequence composition", height=600, showlegend=False)
    write_figure(fig, out, "plot07_umap_by_composition", args.no_html)


def plot_8_duplication(sample_meta, metadata, args, out):
    col = "duplicate_count_in_common_cohort"
    if col not in metadata or col not in sample_meta:
        print("[WARN] Plot 8 skipped: sequence_hash missing from metadata.")
        return
    # Labels for the whole cohort; the scatter only shows the projected sample.
    labels = metadata[KEY + ["sequence_hash", col]].copy()
    labels["duplicate_status"] = np.where(
        labels[col] > 1, "Repeated in common cohort", "Unique in common cohort"
    )
    labels.to_parquet(out / "plot08_duplication_labels.parquet", index=False)
    d = sample_meta.copy()
    d["duplicate_status"] = np.where(
        d[col] > 1, "Repeated in common cohort", "Unique in common cohort"
    )
    fig = px.scatter(
        d, x="umap_raw_x", y="umap_raw_y", color=col,
        color_continuous_scale="Reds", opacity=0.75, render_mode="webgl",
        title="Embedding UMAP coloured by exact-sequence duplication (counted over the full common cohort)",
        labels={col: "Exact sequence copies"},
        hover_data=["record_id", "start", "end", "duplicate_status", "taxonomy"],
    )
    write_figure(fig, out, "plot08_umap_by_duplication", args.no_html)


def plot_9_clade_distances(scan: KnnScan | None, args, out):
    if scan is None:
        return
    if scan.distances is None or scan.distances.empty:
        print("[WARN] Plot 9 skipped: no kNN rows with cosine_distance matched the cohort taxonomy.")
        return
    d = scan.distances
    order = [c for c in DISTANCE_CATEGORIES if c in set(d.shared_lineage_depth)]
    fig = px.violin(
        d, x="shared_lineage_depth", y="cosine_distance",
        category_orders={"shared_lineage_depth": order},
        box=True, points=False, color="shared_lineage_depth",
        title="Cosine distance by deepest shared taxonomy-lineage prefix",
        labels={
            "shared_lineage_depth": "Shared lineage-prefix depth (not a named rank)",
            "cosine_distance": "Cosine distance",
        },
    )
    fig.update_layout(showlegend=False)
    d.to_parquet(out / "plot09_clade_distance_sample.parquet", index=False)
    if scan.category_counts is not None:
        scan.category_counts.to_csv(out / "plot09_clade_distance_counts.csv", index=False)
    write_figure(fig, out, "plot09_intra_vs_inter_clade_distance", args.no_html)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.sample_size < 100:
        raise ValueError("--sample-size should be at least 100 for stable projections.")
    repo_root = args.repo_root.expanduser().resolve()
    out = args.output_dir.expanduser()
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.knn_csv.is_absolute():
        args.knn_csv = (repo_root / args.knn_csv).resolve()

    require_dependencies()
    started = time.time()

    # ClickHouse is only needed for the metadata join and the embedding stream;
    # everything after this block is pandas/numpy/plotly.
    with build_resource(repo_root, args) as ch:
        metadata = prepare_metadata(load_or_build_metadata(ch, args, out))
        emb = stream_embedding_pass(ch, args, metadata)

    metadata = metadata.merge(emb.norm_table, on=KEY, how="left", validate="one_to_one")
    dedupe_report = {}
    if "likelihood_copies" in metadata:
        dedupe_report["likelihood_keys_with_copies"] = int((metadata["likelihood_copies"] > 1).sum())
        dedupe_report["likelihood_max_spread"] = float(metadata["likelihood_spread"].max())
        dedupe_report["likelihood_keys_missing"] = int(metadata["likelihood_copies"].isna().sum())

    plot_1_channel(emb.diagnostic, emb.norms, emb.abs_dim, args, out)
    sample_meta, sample_vectors = add_projection_columns(emb.sample_meta, emb.sample_vectors, args)
    sample_meta.to_parquet(out / "embedding_projection_sample.parquet", index=False)
    plot_2_umap(sample_meta, args, out)
    plot_3_norm_length(metadata, args, out)
    knn = scan_knn(args, metadata)
    plot_4_knn(knn, args, out)
    plot_5_kmer_vs_embedding_pca(sample_meta, sample_vectors, args, out)
    plot_6_likelihood(sample_meta, args, out)
    plot_7_composition(sample_meta, args, out)
    plot_8_duplication(sample_meta, metadata, args, out)
    plot_9_clade_distances(knn, args, out)

    summary = {
        "metadata_rows": int(len(metadata)),
        "embedding_rows_streamed": int(len(emb.norms)),
        "projection_rows": int(len(sample_meta)),
        "embedding_dimension": int(len(emb.diagnostic)),
        "diagnostic_dimension_zero_based": int(args.diagnostic_dim),
        "knn_csv": str(args.knn_csv),
        "duplicate_key_handling": dedupe_report,
        "knn_edges": None if knn is None else {
            "read": knn.edges_read,
            "missing_from_cohort": knn.edges_missing_key,
            "self_edges_excluded": knn.self_edges,
            "without_lineage": knn.edges_without_lineage,
            "used": knn.valid_edges,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "notes": [
            "UMAP is exploratory and was fitted on a sample.",
            "Whitening applies to retained PCA components.",
            "kNN plots use lineage-prefix depth because taxonomy strings lack explicit rank annotations.",
            "Plot 4 is descriptive neighbor agreement, not held-out classification accuracy.",
            "Self-matching kNN edges are excluded from plots 4 and 9.",
            "Duplication counts are over the full common cohort only (not the 32.4M-row corpus).",
        ],
    }
    (out / "run_metadata.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Outputs written to: {out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())