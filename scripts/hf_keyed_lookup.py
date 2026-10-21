#!/usr/bin/env python3
"""Candidate-first retrieval and evidence audit for Carbon genomic intervals (ClickHouse, optimized).

Tuned for a small box (4 cores, 16 GB RAM, 32 GB disk) reading 600+ remote shards.

What changed versus the original:
  * Two-phase retrieval. Phase 1 reads ONLY record_id/start/end and returns the
    shards that contain candidate keys (SELECT DISTINCT _path). Phase 2 reads the
    wide columns (sequence, embedding, ...) from those shards only.
  * No casts on the big side when the native types allow it, so Parquet row-group
    min/max pruning works. Casts are a fallback only.
  * No 3-column grace_hash join: a tuple IN (subquery) against the key file, with
    candidate_key attached in pandas afterwards.
  * Shard-hit cache (shard_index.json), keyed by candidate keys + shard list, so
    reruns skip phase 1 entirely.
  * The three sources run concurrently (threads; each query is its own
    clickhouse-local subprocess).
  * Optional --assume-same-shard-names: run phase 1 on the CPU repo only and
    reuse its hit list for the other repos (verify on a sample first).
  * Per-process memory cap, no --progress output, parquet filter push-down.
  * Pairwise step (opt-in) is vectorized: one matmul for cosines, Biopython
    counts(), process pool for sequence identity, optional alignment gate.
  * Every stage is timed; see timings.json.

start/end are treated as corpus coordinates. The script does NOT assume
end-start equals nucleotide sequence length.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

try:
    from carbon_enrichment.resources.clickhouse import (
        ClickHouseConfig,
        ClickHouseResource,
    )
except ModuleNotFoundError as exc:
    raise ImportError(
        "Could not import ClickHouseResource. Ensure that "
        "'carbon-enrichment' is on PYTHONPATH and that "
        "'carbon_enrichment/resources/clickhouse.py' exists."
    ) from exc

KEYS = ["record_id", "start", "end"]
LIKELIHOOD_FIELDS = [
    "mean_log_prob", "sum_log_prob", "perplexity",
    "supervised_position_count", "min_token_logprob",
    "argmin_position", "per_token_logprob_std",
]
CPU_FIELDS = [
    "begin_of_sequence", "end_of_sequence", "begin_of_gene", "end_of_gene",
    "gene_type", "species_type", "strand", "sequence", "molecule_type",
    "topology", "taxonomy", "gc_content", "gc_skew", "sequence_length",
    "gene_length", "shannon_entropy", "strand_normalized_sequence",
    "taxonomy_domain", "taxonomy_depth", "is_coding_region", "qc_flag",
]
SOURCE_FIELDS = {
    "cpu": CPU_FIELDS,
    "likelihood": LIKELIHOOD_FIELDS,
    "embeddings": ["embedding", "embedding_norm"],
}
SOURCES = ("cpu", "likelihood", "embeddings")


# --------------------------------------------------------------------------- #
# Logging / timing
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


class Timings:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            dur = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + dur
            log(f"[time] {name}: {dur:,.2f}s")

    def total(self) -> float:
        return time.perf_counter() - self.t0


# --------------------------------------------------------------------------- #
# ClickHouse wrapper: no --progress, memory cap, parquet filter push-down
# --------------------------------------------------------------------------- #
class FastClickHouse(ClickHouseResource):
    def __init__(self, config: ClickHouseConfig, extra_flags: list[str]) -> None:
        super().__init__(config)
        self._extra_flags = extra_flags

    def _base_args(self, extra_args: list[str] | None = None) -> list[str]:
        args = [a for a in super()._base_args(extra_args) if a != "--progress"]
        return args + self._extra_flags


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ch_sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def ch_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def digest(items) -> str:
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Shard discovery
# --------------------------------------------------------------------------- #
def parquet_urls(api: HfApi, repo_id: str, revision: str = "main") -> list[str]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    paths = sorted(x for x in files if x.lower().endswith(".parquet"))
    if not paths:
        raise RuntimeError(f"No Parquet files found in {repo_id}@{revision}")
    return [
        f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quote(path, safe='/')}"
        for path in paths
    ]


def repo_relative_path(url: str) -> str:
    """Path of a shard inside its repo (after /resolve/<rev>/)."""
    m = re.search(r"/resolve/[^/]+/(.+)$", urlparse(url).path)
    return unquote(m.group(1)) if m else unquote(urlparse(url).path)


# --------------------------------------------------------------------------- #
# Candidate normalization (unchanged behaviour)
# --------------------------------------------------------------------------- #
def normalize_candidate_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")
    raw = pd.read_csv(path, low_memory=False)

    membership_cols = {"panel", "record_id", "start", "end", "eps_1_min20_cluster"}
    if membership_cols.issubset(raw.columns):
        raw = raw.loc[raw["panel"].astype("string").str.strip().eq("raw")].copy()
        raw["eps_1_min20_cluster"] = pd.to_numeric(raw["eps_1_min20_cluster"], errors="coerce")
        raw = raw.loc[raw["eps_1_min20_cluster"].eq(2)].copy()
        if "candidate_definition" not in raw.columns:
            raw["candidate_definition"] = "panel=raw; eps_1_min20_cluster=2"
        log(
            "[candidates] Recognized UMAP memberships; selected "
            f"{len(raw):,} raw-panel intervals where eps_1_min20_cluster == 2."
        )

    aliases = {
        "record_id": ["record_id", "accession", "sequence_id", "source_record_id"],
        "start": ["start", "begin", "interval_start"],
        "end": ["end", "stop", "interval_end"],
    }
    rename = {}
    for target, choices in aliases.items():
        found = next((c for c in choices if c in raw.columns), None)
        if found is None:
            raise ValueError(
                f"Candidate file must contain interval key {target!r}. "
                f"Accepted aliases: {choices}. Found columns: {list(raw.columns)}"
            )
        if found != target:
            rename[found] = target
    raw = raw.rename(columns=rename)
    raw["record_id"] = raw["record_id"].astype("string").str.strip()
    for col in ("start", "end"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("Int64")
    invalid = raw[raw[KEYS].isna().any(axis=1)].copy()
    valid = raw.dropna(subset=KEYS).copy()
    valid["start"] = valid["start"].astype("int64")
    valid["end"] = valid["end"].astype("int64")
    invalid["candidate_input_row"] = invalid.index
    valid["candidate_input_row"] = valid.index
    invalid_key = valid[(valid["start"] < 0) | (valid["end"] <= valid["start"])].copy()
    valid = valid[(valid["start"] >= 0) & (valid["end"] > valid["start"])].copy()
    invalid = pd.concat([invalid, invalid_key], ignore_index=True).drop_duplicates()
    key_counts = valid.groupby(KEYS, dropna=False).size().rename("candidate_key_input_rows").reset_index()
    unique = valid.drop_duplicates(KEYS, keep="first").copy()
    unique = unique.merge(key_counts, on=KEYS, how="left", validate="one_to_one")
    unique["candidate_key"] = [
        hashlib.sha256(f"{r}|{s}|{e}".encode()).hexdigest()[:16]
        for r, s, e in unique[KEYS].itertuples(index=False, name=None)
    ]
    return unique, invalid


# --------------------------------------------------------------------------- #
# Retrieval: schema, filter, phase 1 (hit shards), phase 2 (fetch rows)
# --------------------------------------------------------------------------- #
def describe_schema(ch: ClickHouseResource, sample_url: str) -> dict[str, str]:
    """Schema of ONE shard (not all 600+)."""
    src = ch._build_hf_url_source(sample_url, token=ch.get_hf_token())
    df = ch.query_arrow(f"DESCRIBE TABLE {src}").to_pandas()
    return dict(zip(df["name"].astype(str), df["type"].astype(str)))


def key_exprs(schema: dict[str, str]) -> dict:
    rid_native = "String" in schema["record_id"]
    st_native = bool(re.search(r"U?Int\d+", schema["start"]))
    en_native = bool(re.search(r"U?Int\d+", schema["end"]))
    return {
        "rid_native": rid_native, "st_native": st_native, "en_native": en_native,
        "rid": "`record_id`" if rid_native else "toString(`record_id`)",
        "st": "`start`" if st_native else "toInt64OrNull(toString(`start`))",
        "en": "`end`" if en_native else "toInt64OrNull(toString(`end`))",
    }


def build_where(ex: dict, stats: dict, keys_path: Path) -> str:
    """Prefilters on native columns enable row-group pruning; the tuple IN is exact."""
    where = []
    if ex["rid_native"]:
        where.append(f"`record_id` >= {ch_sql_string(stats['rid_lo'])} "
                     f"AND `record_id` <= {ch_sql_string(stats['rid_hi'])}")
    if ex["st_native"]:
        where.append(f"`start` >= {stats['min_start']}")
    if ex["en_native"]:
        where.append(f"`end` <= {stats['max_end']}")
    key_src = f"file({ch_sql_string(str(keys_path.resolve()))}, Parquet)"
    where.append(
        f"({ex['rid']}, {ex['st']}, {ex['en']}) IN "
        f"(SELECT `record_id`, `start`, `end` FROM {key_src})"
    )
    return "WHERE " + " AND ".join(where)


def find_hit_shards(ch: ClickHouseResource, name: str, urls: list[str], schema: dict,
                    stats: dict, keys_path: Path) -> list[str]:
    """Phase 1: narrow scan of key columns only; returns shards holding >=1 candidate."""
    rel = f"scan_{name}"
    ch.register_hf_shards(rel, urls)
    ex = key_exprs(schema)
    sql = f"SELECT DISTINCT _path AS p FROM {ch.source_expr(rel)} {build_where(ex, stats, keys_path)}"
    paths = set(ch.query_arrow(sql).to_pandas()["p"].astype(str))
    if not paths:
        return []
    by_variants = {}
    for u in urls:
        for v in {u, u.split("?")[0], unquote(u), unquote(u.split("?")[0])}:
            by_variants[v] = u
    hits = sorted({by_variants[p] for p in paths if p in by_variants})
    if not hits:
        log(f"[{name}] WARNING: could not map _path values back to URLs "
            f"(sample: {next(iter(paths))!r}); falling back to all shards in phase 2.")
        return list(urls)
    return hits


def fetch_rows(ch: ClickHouseResource, name: str, hit_urls: list[str], schema: dict,
               stats: dict, keys_path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: wide-column read from hit shards only."""
    fields = SOURCE_FIELDS[name]
    cols = KEYS + ["candidate_key"] + fields
    if not hit_urls:
        return pd.DataFrame(columns=cols + ["source_shard"])
    rel = f"hit_{name}"
    ch.register_hf_shards(rel, hit_urls)
    ex = key_exprs(schema)

    proj = [
        "`record_id`" if ex["rid_native"] else "toString(`record_id`) AS `record_id`",
        "`start`" if ex["st_native"] else f"{ex['st']} AS `start`",
        "`end`" if ex["en_native"] else f"{ex['en']} AS `end`",
    ]
    for col in fields:
        if col in schema:
            proj.append(ch_ident(col))
        else:
            proj.append(f"CAST(NULL, 'Nullable(String)') AS {ch_ident(col)}")
    proj.append("_path AS `source_shard`")

    sql = f"SELECT {', '.join(proj)} FROM {ch.source_expr(rel)} {build_where(ex, stats, keys_path)}"
    df = ch.query_arrow(sql).to_pandas()
    if df.empty:
        return pd.DataFrame(columns=cols + ["source_shard"])
    df["record_id"] = df["record_id"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype("int64")
    df["end"] = df["end"].astype("int64")
    merged = df.merge(candidates[KEYS + ["candidate_key"]], on=KEYS, how="inner")
    return merged[[c for c in cols if c in merged.columns] + ["source_shard"]]


def load_prior_index(path: Path, enabled: bool) -> dict:
    if enabled and path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def cache_hit(prior: dict, name: str, cand_digest: str, urls_digest: str, revision: str):
    p = prior.get(name)
    if (p and p.get("candidate_digest") == cand_digest and p.get("urls_digest") == urls_digest
            and p.get("revision") == revision):
        return list(p["shards_with_hits"])
    return None


# --------------------------------------------------------------------------- #
# Duplicate audit and collapse (unchanged behaviour)
# --------------------------------------------------------------------------- #
def duplicate_audit(df: pd.DataFrame, source: str, outdir: Path) -> pd.DataFrame:
    if df.empty:
        audit = pd.DataFrame(columns=KEYS + ["source", "source_rows"])
    else:
        audit = df.groupby(KEYS, dropna=False).size().rename("source_rows").reset_index()
        audit = audit[audit["source_rows"] > 1].copy()
        audit["source"] = source
        audit = audit[["source"] + KEYS + ["source_rows"]]
    audit.to_csv(outdir / f"duplicate_keys_{source}.csv", index=False)
    return audit


def scalar_unique_or_none(series: pd.Series):
    vals = series.dropna()
    if len(vals) == 0:
        return None, 0
    distinct = vals.astype(str).nunique()
    if distinct > 1:
        return None, distinct
    return vals.iloc[0], distinct


def collapse_cpu(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    numeric_cols = {"gc_content", "gc_skew", "sequence_length", "gene_length",
                    "shannon_entropy", "taxonomy_depth"}
    for key, g in df.groupby(KEYS, dropna=False, sort=False):
        row = dict(zip(KEYS, key))
        row["candidate_key"] = g["candidate_key"].iloc[0]
        row["cpu_source_rows"] = len(g)
        seqs = g["sequence"].dropna().astype(str).drop_duplicates() if "sequence" in g else pd.Series(dtype=str)
        row["cpu_distinct_sequences"] = len(seqs)
        row["sequence_conflict"] = len(seqs) > 1
        for col in CPU_FIELDS:
            if col not in g.columns:
                continue
            if col == "sequence":
                row[col] = seqs.iloc[0] if len(seqs) == 1 else None
            elif col == "strand_normalized_sequence":
                vals = g[col].dropna().astype(str).drop_duplicates()
                row[col] = vals.iloc[0] if len(vals) == 1 else None
            elif col in numeric_cols:
                vals = pd.to_numeric(g[col], errors="coerce").dropna().drop_duplicates()
                row[col] = vals.iloc[0] if len(vals) == 1 else (float(vals.mean()) if len(vals) else np.nan)
            else:
                row[col], distinct = scalar_unique_or_none(g[col])
                if distinct > 1:
                    row[f"{col}_conflict"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_likelihood(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    numeric = [c for c in LIKELIHOOD_FIELDS if c in df.columns]
    for key, g in df.groupby(KEYS, dropna=False, sort=False):
        row = dict(zip(KEYS, key))
        row["candidate_key"] = g["candidate_key"].iloc[0]
        row["likelihood_source_rows"] = len(g)
        row["likelihood_duplicate_key"] = len(g) > 1
        for col in numeric:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[col] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{col}_max"] = float(vals.max()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for key, g in df.groupby(KEYS, dropna=False, sort=False):
        row = dict(zip(KEYS, key))
        row["candidate_key"] = g["candidate_key"].iloc[0]
        row["embedding_source_rows"] = len(g)
        vectors = []
        for value in g.get("embedding", pd.Series(dtype=object)).tolist():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size and np.isfinite(arr).all():
                    vectors.append(arr)
            except Exception:
                pass
        unique_vectors = []
        for v in vectors:
            if not any(v.shape == old.shape and np.array_equal(v, old) for old in unique_vectors):
                unique_vectors.append(v)
        row["embedding_distinct_vectors"] = len(unique_vectors)
        row["embedding_conflict"] = len(unique_vectors) > 1
        row["embedding"] = unique_vectors[0].tolist() if len(unique_vectors) == 1 else None
        norms = pd.to_numeric(g.get("embedding_norm", pd.Series(dtype=float)), errors="coerce").dropna()
        row["embedding_norm"] = (
            float(norms.iloc[0]) if len(norms) and norms.nunique() == 1
            else (float(norms.mean()) if len(norms) else np.nan)
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pairwise comparisons (vectorized cosine, parallel sequence identity)
# --------------------------------------------------------------------------- #
_CANON = np.zeros(256, dtype=bool)
_CANON[[ord(c) for c in "ACGT"]] = True
_ALIGNER = None
_SEQS: list = []


def shared_taxonomy_prefix(a, b) -> int:
    if not isinstance(a, str) or not isinstance(b, str):
        return 0
    aa = [x.strip() for x in a.split(";") if x.strip()]
    bb = [x.strip() for x in b.split(";") if x.strip()]
    n = 0
    for x, y in zip(aa, bb):
        if x != y:
            break
        n += 1
    return n


def _get_aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from Bio.Align import PairwiseAligner
        al = PairwiseAligner()
        al.mode = "global"
        al.match_score = 2
        al.mismatch_score = -1
        al.open_gap_score = -5
        al.extend_gap_score = -1
        _ALIGNER = al
    return _ALIGNER


def sequence_identity(seq_a, seq_b, allow_align: bool = True) -> tuple[float | None, str]:
    if not isinstance(seq_a, str) or not isinstance(seq_b, str) or not seq_a or not seq_b:
        return None, "missing_sequence"
    a, b = seq_a.upper(), seq_b.upper()
    if a == b:
        return 1.0, "exact_match"
    if len(a) == len(b):
        xa = np.frombuffer(a.encode("ascii", "replace"), dtype=np.uint8)
        xb = np.frombuffer(b.encode("ascii", "replace"), dtype=np.uint8)
        valid = _CANON[xa] & _CANON[xb]
        n = int(valid.sum())
        if n == 0:
            return None, "no_canonical_positions"
        return float((xa[valid] == xb[valid]).sum() / n), "same_length_position_identity"
    if not allow_align:
        return None, "alignment_skipped_below_cosine_gate"
    try:
        aln = _get_aligner().align(a, b)[0]
        c = aln.counts()
        cols = c.identities + c.mismatches + c.gaps
        return (c.identities / cols if cols else None), "global_alignment_identity"
    except ImportError:
        return None, "different_lengths_install_biopython_for_alignment"
    except Exception as exc:
        return None, f"alignment_error:{type(exc).__name__}"


def _init_worker(seqs: list) -> None:
    global _SEQS
    _SEQS = seqs


def _identity_chunk(pairs: list[tuple[int, int, bool]]):
    return [sequence_identity(_SEQS[i], _SEQS[j], allow) for i, j, allow in pairs]


def compute_pairwise(integrated: pd.DataFrame, workers: int, align_min_cosine: float | None) -> pd.DataFrame:
    colnames = list(integrated.columns)

    def pick(*names):
        return next((n for n in names if n in colnames), None)

    seq_col = pick("sequence", "cpu_sequence")
    norm_seq_col = pick("strand_normalized_sequence", "cpu_strand_normalized_sequence")
    tax_col = pick("taxonomy", "cpu_taxonomy")
    vec_col = pick("embedding", "embedding_embedding")
    score_col = pick("mean_log_prob", "likelihood_mean_log_prob")
    bracket_col = pick("score_bracket", "score_bracket_candidate")

    rows = integrated.to_dict("records")
    n = len(rows)
    if n < 2:
        return pd.DataFrame()

    seqs = []
    for r in rows:
        ns = r.get(norm_seq_col) if norm_seq_col else None
        seqs.append(ns if isinstance(ns, str) else (r.get(seq_col) if seq_col else None))

    vecs: list[np.ndarray | None] = []
    for r in rows:
        v = r.get(vec_col) if vec_col else None
        arr = None
        if v is not None:
            try:
                x = np.asarray(v, dtype=np.float64)
                if x.ndim == 1 and x.size:
                    nx = np.linalg.norm(x)
                    if nx > 0 and np.isfinite(nx):
                        arr = x / nx
            except Exception:
                arr = None
        vecs.append(arr)
    dims = Counter(v.shape[0] for v in vecs if v is not None)
    pos = [-1] * n
    cos = None
    if dims:
        dim = dims.most_common(1)[0][0]
        idx = [i for i, v in enumerate(vecs) if v is not None and v.shape[0] == dim]
        for k, i in enumerate(idx):
            pos[i] = k
        mat = np.vstack([vecs[i] for i in idx])
        cos = mat @ mat.T

    def cos_ij(i: int, j: int):
        if cos is not None and pos[i] >= 0 and pos[j] >= 0:
            return float(cos[pos[i], pos[j]])
        return None

    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        allow = True
        if align_min_cosine is not None:
            c = cos_ij(i, j)
            allow = c is not None and c >= align_min_cosine
        pairs.append((i, j, allow))
    log(f"[pairwise] {len(pairs):,} pairs; identity workers={workers}")

    if workers > 1 and len(pairs) > 2000:
        chunk = max(500, len(pairs) // (workers * 8))
        chunks = [pairs[i:i + chunk] for i in range(0, len(pairs), chunk)]
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(seqs,)) as pool:
            identities = [x for part in pool.map(_identity_chunk, chunks) for x in part]
    else:
        _init_worker(seqs)
        identities = _identity_chunk(pairs)

    pair_rows = []
    for (i, j, _), (identity, method) in zip(pairs, identities):
        a, b = rows[i], rows[j]
        pair_rows.append({
            "candidate_key_a": a.get("candidate_key"),
            "candidate_key_b": b.get("candidate_key"),
            "record_id_a": a.get("record_id"), "start_a": a.get("start"), "end_a": a.get("end"),
            "record_id_b": b.get("record_id"), "start_b": b.get("start"), "end_b": b.get("end"),
            "sequence_identity": identity, "sequence_identity_method": method,
            "embedding_cosine_similarity": cos_ij(i, j),
            "shared_taxonomy_prefix_depth": shared_taxonomy_prefix(
                a.get(tax_col) if tax_col else None, b.get(tax_col) if tax_col else None),
            "taxonomy_a": a.get(tax_col) if tax_col else None,
            "taxonomy_b": b.get(tax_col) if tax_col else None,
            "mean_log_prob_a": a.get(score_col) if score_col else None,
            "mean_log_prob_b": b.get(score_col) if score_col else None,
            "score_bracket_a": a.get(bracket_col) if bracket_col else None,
            "score_bracket_b": b.get(bracket_col) if bracket_col else None,
            "sequence_comparison_note": "strand-normalized sequence used when available; identity is not proof of duplication",
        })
    return pd.DataFrame(pair_rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    cpu_n = os.cpu_count() or 2
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, help="CSV containing record_id,start,end candidate keys")
    parser.add_argument("--output-dir", default="metrics/derived/biological_annotation")
    parser.add_argument("--cpu-repo", default="AINovice2005/carbon-cpu-enriched-sequences")
    parser.add_argument("--likelihood-repo", default="AINovice2005/carbon-likelihood-stats")
    parser.add_argument("--embedding-repo", default="AINovice2005/carbon-embeddings")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-candidates", type=int, default=500,
                        help="Safety limit for pairwise comparisons; set 0 to disable")
    parser.add_argument("--clickhouse-binary", default=os.environ.get("CLICKHOUSE_BINARY"))
    parser.add_argument("--clickhouse-threads", type=int, default=cpu_n,
                        help="Threads per clickhouse-local process (3 run concurrently)")
    parser.add_argument("--clickhouse-temp-dir", default="data/tmp/clickhouse")
    parser.add_argument("--download-threads", type=int, default=12,
                        help="Download threads per process. 3 concurrent processes => 3x this many connections")
    parser.add_argument("--ch-memory-gb", type=float, default=4.0,
                        help="max_memory_usage per clickhouse process (3 run concurrently)")
    parser.add_argument("--sequential-sources", action="store_true",
                        help="Run the three sources one after another instead of concurrently")
    parser.add_argument("--assume-same-shard-names", action="store_true",
                        help="Run phase 1 on the CPU repo only and reuse its hit shards for the other "
                             "repos by repo-relative path. Only valid if all three repos are sharded "
                             "identically; verify on a sample first.")
    parser.add_argument("--no-shard-cache", action="store_true",
                        help="Ignore/overwrite shard_index.json instead of reusing hit shards")
    parser.add_argument("--workers", type=int, default=max(1, cpu_n - 1),
                        help="Processes for pairwise sequence identity")
    parser.add_argument("--align-min-cosine", type=float, default=None,
                        help="Only run different-length global alignment for pairs with embedding "
                             "cosine >= this value (skips the quadratic alignment otherwise)")
    parser.add_argument("--pairwise", action="store_true",
                        help="Enable O(n^2) all-pairs sequence/vector comparisons (off by default)")
    parser.add_argument("--skip-pairwise", action="store_true",
                        help="Compatibility option: skip all-pairs comparisons even if --pairwise is supplied")
    args = parser.parse_args()

    tm = Timings()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(args.candidates)

    with tm.stage("load_candidates"):
        candidates, invalid_candidates = normalize_candidate_file(candidate_path)
        candidates.to_csv(outdir / "candidate_keys_frozen.csv", index=False)
        invalid_candidates.to_csv(outdir / "invalid_candidate_rows.csv", index=False)
        if candidates.empty:
            raise RuntimeError("No valid candidate keys remain after validation")
        do_pairwise = bool(args.pairwise and not args.skip_pairwise)
        if do_pairwise and args.max_candidates and len(candidates) > args.max_candidates:
            raise RuntimeError(
                f"{len(candidates)} candidates exceed --max-candidates={args.max_candidates}. "
                "Pairwise work is O(n^2). Raise the limit deliberately if this is intended."
            )

    token = os.environ.get("HF_TOKEN")
    if not token:
        log("[warn] HF_TOKEN not set; anonymous requests are throttled much sooner.")
    api = HfApi(token=token)
    repos = {"cpu": args.cpu_repo, "likelihood": args.likelihood_repo, "embeddings": args.embedding_repo}

    with tm.stage("discover_shards"):
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {n: ex.submit(parquet_urls, api, r, args.revision) for n, r in repos.items()}
            urls = {n: f.result() for n, f in futs.items()}
        for name, repo in repos.items():
            log(f"[{name}] {repo}@{args.revision}: discovered {len(urls[name])} Parquet file(s)")
        (outdir / "source_inventory.json").write_text(json.dumps(
            {k: {"repo": repos[k], "revision": args.revision, "parquet_files": urls[k]} for k in urls}, indent=2))

    with tm.stage("write_key_file"):
        keys_path = outdir / "candidate_keys_for_clickhouse.parquet"
        key_export = candidates[KEYS + ["candidate_key"]].copy()
        key_export["record_id"] = key_export["record_id"].astype(str)
        key_export["start"] = key_export["start"].astype("int64")
        key_export["end"] = key_export["end"].astype("int64")
        key_export.to_parquet(keys_path, index=False)
        rids = sorted(key_export["record_id"].unique().tolist())
        stats = {
            "n": len(key_export), "rid_lo": rids[0], "rid_hi": rids[-1],
            "min_start": int(key_export["start"].min()), "max_end": int(key_export["end"].max()),
        }

    mem_bytes = int(args.ch_memory_gb * 1024**3)
    ch_config = ClickHouseConfig(
        binary_path=args.clickhouse_binary or ClickHouseConfig().binary_path,
        threads=args.clickhouse_threads,
        temp_directory=args.clickhouse_temp_dir,
        hf_token=token,
        join_algorithm="grace_hash",
        max_download_threads=args.download_threads,
        max_parsing_threads=max(2, cpu_n),
        max_download_buffer_size=8 * 1024 * 1024,
    )
    ch = FastClickHouse(ch_config, [
        "--max_memory_usage", str(mem_bytes),
        "--input_format_parquet_filter_push_down", "1",
    ])

    cand_digest = digest(sorted(candidates["candidate_key"]))
    urls_digest = {n: digest(urls[n]) for n in urls}
    index_path = outdir / "shard_index.json"
    prior_index = load_prior_index(index_path, enabled=not args.no_shard_cache)

    raw: dict[str, pd.DataFrame] = {}
    scan_stats: dict[str, dict] = {}
    hits: dict[str, list[str]] = {}
    schemas: dict[str, dict] = {}

    try:
        ch.get_binary()

        def phase1(name: str) -> list[str]:
            cached = cache_hit(prior_index, name, cand_digest, urls_digest[name], args.revision)
            if cached is not None:
                log(f"[{name}] shard cache hit: {len(cached)}/{len(urls[name])} shards")
                return cached
            t = time.perf_counter()
            log(f"[{name}] phase 1: scanning key columns of {len(urls[name])} shard(s)...")
            h = find_hit_shards(ch, name, urls[name], schemas[name], stats, keys_path)
            log(f"[{name}] phase 1: {len(h)} shard(s) with hits in {time.perf_counter() - t:,.1f}s")
            return h

        def phase1_map_from_cpu(name: str, cpu_hits: list[str]) -> list[str]:
            wanted = {repo_relative_path(u) for u in cpu_hits}
            mapped = [u for u in urls[name] if repo_relative_path(u) in wanted]
            if not mapped and cpu_hits:
                log(f"[{name}] no shard names matched the CPU repo; falling back to a full phase 1")
                return phase1(name)
            log(f"[{name}] reusing CPU hit list: {len(mapped)}/{len(urls[name])} shards")
            return mapped

        with tm.stage("schemas"):
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {n: ex.submit(describe_schema, ch, urls[n][0]) for n in SOURCES}
                schemas = {n: f.result() for n, f in futs.items()}
            for n in SOURCES:
                missing = set(KEYS) - set(schemas[n])
                if missing:
                    raise ValueError(f"{n} dataset missing required columns: {sorted(missing)}")
                log(f"[{n}] key types: " + ", ".join(f"{k}={schemas[n][k]}" for k in KEYS))

        with tm.stage("phase1_find_hit_shards"):
            if args.assume_same_shard_names:
                log("[warn] --assume-same-shard-names: assuming all repos share shard layout.")
                hits["cpu"] = phase1("cpu")
                for n in ("likelihood", "embeddings"):
                    hits[n] = phase1_map_from_cpu(n, hits["cpu"])
            elif args.sequential_sources:
                hits = {n: phase1(n) for n in SOURCES}
            else:
                with ThreadPoolExecutor(max_workers=3) as ex:
                    futs = {n: ex.submit(phase1, n) for n in SOURCES}
                    hits = {n: f.result() for n, f in futs.items()}

        index = dict(prior_index)
        for n in SOURCES:
            index[n] = {
                "revision": args.revision, "candidate_digest": cand_digest,
                "urls_digest": urls_digest[n], "shards_with_hits": hits[n],
                "assumed_same_shard_names": bool(args.assume_same_shard_names and n != "cpu"),
            }
        index_path.write_text(json.dumps(index, indent=2))

        with tm.stage("phase2_fetch_rows"):
            def phase2(name: str):
                t = time.perf_counter()
                df = fetch_rows(ch, name, hits[name], schemas[name], stats, keys_path, candidates)
                secs = time.perf_counter() - t
                st = {
                    "shards_total": len(urls[name]), "shards_scanned_phase2": len(hits[name]),
                    "rows": int(len(df)), "fetch_seconds": round(secs, 3),
                }
                log(f"[{name}] phase 2: {len(df):,} row(s) from {len(hits[name])} shard(s) in {secs:,.1f}s")
                return df, st

            if args.sequential_sources:
                results = {n: phase2(n) for n in SOURCES}
            else:
                with ThreadPoolExecutor(max_workers=3) as ex:
                    futs = {n: ex.submit(phase2, n) for n in SOURCES}
                    results = {n: f.result() for n, f in futs.items()}
            for n in SOURCES:
                raw[n], scan_stats[n] = results[n]
                raw[n].to_parquet(outdir / f"retrieved_{n}_raw.parquet", index=False)

        with tm.stage("duplicate_audit"):
            dup_audits = [duplicate_audit(raw[n], n, outdir) for n in SOURCES]
            audit = candidates[["candidate_key"] + KEYS + ["candidate_key_input_rows"]].copy()
            for name, df in raw.items():
                counts = (df.groupby("candidate_key").size().rename(f"{name}_match_rows")
                          if not df.empty else pd.Series(dtype="int64", name=f"{name}_match_rows"))
                audit = audit.merge(counts.reset_index(), on="candidate_key", how="left")
                audit[f"{name}_match_rows"] = audit[f"{name}_match_rows"].fillna(0).astype(int)
                audit[f"{name}_found"] = audit[f"{name}_match_rows"] > 0
            audit["found_in_all_three"] = audit[["cpu_found", "likelihood_found", "embeddings_found"]].all(axis=1)
            audit.to_csv(outdir / "retrieval_audit.csv", index=False)
            pd.concat(dup_audits, ignore_index=True).to_csv(outdir / "duplicate_key_audit.csv", index=False)

        with tm.stage("collapse_sources"):
            cpu = collapse_cpu(raw["cpu"].drop(columns=["source_shard"], errors="ignore"))
            likelihood = collapse_likelihood(raw["likelihood"].drop(columns=["source_shard"], errors="ignore"))
            embeddings = collapse_embeddings(raw["embeddings"].drop(columns=["source_shard"], errors="ignore"))

        with tm.stage("integrate_and_write_evidence"):
            meta_cols = [c for c in candidates.columns if c not in KEYS + ["candidate_key"]]
            meta = candidates[["candidate_key"] + meta_cols].copy()
            integrated = candidates[["candidate_key"] + KEYS].merge(
                meta, on="candidate_key", how="left", suffixes=("", "_candidate"))
            for df, prefix in ((cpu, "cpu"), (likelihood, "likelihood"), (embeddings, "embedding")):
                if df.empty:
                    continue
                drop = [c for c in KEYS + ["candidate_key"] if c in df.columns]
                other = df.drop(columns=drop, errors="ignore")
                rename = {c: f"{prefix}_{c}" for c in other.columns if c in integrated.columns}
                if rename:
                    other = other.rename(columns=rename)
                integrated = integrated.merge(
                    df[["candidate_key"]].join(other), on="candidate_key", how="left", validate="one_to_one")
            integrated.to_parquet(outdir / "candidate_evidence.parquet", index=False)
            csv_df = integrated.copy()
            if "embedding" in csv_df.columns:
                csv_df["embedding"] = csv_df["embedding"].map(
                    lambda x: json.dumps(x) if isinstance(x, (list, tuple, np.ndarray)) else None)
            csv_df.to_csv(outdir / "candidate_evidence.csv", index=False)

        with tm.stage("pairwise"):
            pair_df = (compute_pairwise(integrated, args.workers, args.align_min_cosine)
                       if do_pairwise else pd.DataFrame())
            pair_df.to_csv(outdir / "candidate_pairwise_comparisons.csv", index=False)
    finally:
        ch.close()

    with tm.stage("write_summary"):
        def count(df, col, pred):
            return int(pred(df[col]).sum()) if (not df.empty and col in df) else 0

        summary = {
            "candidate_input": str(candidate_path),
            "query_engine": "ClickHouseResource / clickhouse local (two-phase)",
            "clickhouse_binary": ch_config.binary_path,
            "clickhouse_threads_per_process": args.clickhouse_threads,
            "download_threads_per_process": args.download_threads,
            "memory_cap_gb_per_process": args.ch_memory_gb,
            "sources_run_concurrently": not args.sequential_sources,
            "assume_same_shard_names": args.assume_same_shard_names,
            "candidate_rows_after_key_deduplication": int(len(candidates)),
            "invalid_candidate_rows": int(len(invalid_candidates)),
            "candidates_found_in_cpu": int(audit["cpu_found"].sum()),
            "candidates_found_in_likelihood": int(audit["likelihood_found"].sum()),
            "candidates_found_in_embeddings": int(audit["embeddings_found"].sum()),
            "candidates_found_in_all_three": int(audit["found_in_all_three"].sum()),
            "cpu_duplicate_keys_in_candidate_hits": count(cpu, "cpu_source_rows", lambda s: s > 1),
            "cpu_sequence_conflict_keys": count(cpu, "sequence_conflict", lambda s: s.fillna(False)),
            "likelihood_duplicate_keys_in_candidate_hits": count(likelihood, "likelihood_duplicate_key", lambda s: s.fillna(False)),
            "embedding_duplicate_keys_in_candidate_hits": count(embeddings, "embedding_source_rows", lambda s: s > 1),
            "embedding_conflict_keys": count(embeddings, "embedding_conflict", lambda s: s.fillna(False)),
            "pairwise_comparisons_enabled": do_pairwise,
            "pairwise_comparisons": int(len(pair_df)),
            "scan_stats": scan_stats,
            "hit_shard_counts": {n: len(hits.get(n, [])) for n in SOURCES},
            "stage_timings_seconds": {k: round(v, 3) for k, v in tm.stages.items()},
            "notes": [
                "When the input is a UMAP membership table, candidates are filtered to panel='raw' and eps_1_min20_cluster=2.",
                "All-pairs comparisons are disabled by default because their cost grows quadratically with candidate count.",
                "start/end are treated as corpus coordinates; no assumption that end-start equals nucleotide length.",
                "Duplicate likelihood rows are averaged and min/max retained; interpret these keys as ambiguous.",
                "Conflicting CPU sequences and conflicting embeddings are nulled rather than arbitrarily selected.",
                "Embedding similarity and model likelihood are not evidence of biological duplication by themselves.",
                "Retrieval is two-phase: phase 1 finds shards holding candidate keys, phase 2 reads wide columns from those shards only.",
                "Shard-hit cache assumes shard contents have not changed for an identical shard list and revision.",
            ],
        }
        (outdir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))

        report = [
            "# Candidate Retrieval and Biological Evidence Audit", "",
            f"- Candidate interval keys: **{summary['candidate_rows_after_key_deduplication']}**",
            f"- Invalid candidate rows excluded: **{summary['invalid_candidate_rows']}**",
            f"- Found in CPU dataset: **{summary['candidates_found_in_cpu']}**",
            f"- Found in likelihood dataset: **{summary['candidates_found_in_likelihood']}**",
            f"- Found in embedding dataset: **{summary['candidates_found_in_embeddings']}**",
            f"- Found in all three datasets: **{summary['candidates_found_in_all_three']}**",
            f"- Pairwise comparisons enabled: **{summary['pairwise_comparisons_enabled']}**",
            f"- Pairwise comparisons generated: **{summary['pairwise_comparisons']}**", "",
            "## Interpretation constraints", "",
            "1. `start` and `end` are corpus coordinates in the supplied schemas. Their difference is not assumed to equal nucleotide sequence length.",
            "2. Duplicate-key reports must be reviewed. Likelihood duplicates are summarized with means and ranges; conflicting sequence/vector values are not silently selected.",
            "3. Sequence identity and embedding cosine similarity are comparative signals, not standalone proof of duplication, homology, function, or evolutionary mechanism.",
            "4. Taxonomy is inherited from the CPU dataset. Missing taxonomic lineage is not inferred.",
            "5. Inspect `retrieval_audit.csv`, `duplicate_key_audit.csv`, and `candidate_pairwise_comparisons.csv` before making biological claims.", "",
            "## Scan performance", "",
            "| source | shards total | phase-2 shards | rows | fetch s |",
            "|---|---|---|---|---|",
        ]
        for n, s in scan_stats.items():
            report.append(f"| {n} | {s['shards_total']} | {s['shards_scanned_phase2']} | {s['rows']:,} | {s['fetch_seconds']} |")
        report += ["", "## Stage timings (s)", "", "| stage | seconds |", "|---|---|"]
        report += [f"| {k} | {v:,.2f} |" for k, v in tm.stages.items()]
        report += [
            "", "## Outputs", "",
            "- `candidate_keys_frozen.csv`: frozen candidate keys and retained input metadata",
            "- `candidate_keys_for_clickhouse.parquet`: key relation used by ClickHouse filters",
            "- `retrieval_audit.csv`: per-key availability across all three datasets",
            "- `duplicate_key_audit.csv` and `duplicate_keys_*.csv`: duplicate-key evidence",
            "- `retrieved_*_raw.parquet`: matching source rows (with `source_shard`) before duplicate handling",
            "- `candidate_evidence.parquet` / `.csv`: integrated candidate-level evidence",
            "- `candidate_pairwise_comparisons.csv`: sequence, embedding, and taxonomy comparisons",
            "- `shard_index.json`: shards that held hits, reused on later runs",
            "- `timings.json`: per-stage timings",
            "- `pipeline_summary.json`: machine-readable run summary", "",
        ]
        (outdir / "biological_facts_for_writeup.md").write_text("\n".join(report))

    total = tm.total()
    (outdir / "timings.json").write_text(json.dumps({
        "total_seconds": round(total, 3),
        "stages_seconds": {k: round(v, 3) for k, v in tm.stages.items()},
        "sources": scan_stats,
    }, indent=2))
    log("\nPipeline completed. Review outputs in: " + str(outdir.resolve()))
    log("\n=== Timing summary ===")
    for k, v in tm.stages.items():
        log(f"  {k:<32}{v:>10,.2f}s")
    log(f"  {'TOTAL':<32}{total:>10,.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("DEBUG_PIPELINE") == "1":
            traceback.print_exc()
        raise SystemExit(1)