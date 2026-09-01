#!/usr/bin/env python3
"""
build_cohort_pipeline.py

Reads the three carbon-* HF datasets from Hugging Face through clickhouse-local, resolves their shard lists via the HF dataset API, uses an explicit cohort-key schema, and restricts
the join to the record_id/start/end keys in a user-supplied
common_cohort.csv, computes the candidate-screening table (within-taxon
percentiles, likelihood z-scores), and writes CSV — with wall-clock time
and throughput (rows/sec) logged per stage.

Requires the `clickhouse` binary (clickhouse-local mode) on PATH.
Install: curl https://clickhouse.com/ | sh

Usage:
    python build_cohort_pipeline.py \
        --cohort-keys common_cohort.csv \
        --output candidate_screening.csv \
        --cpu-glob 'https://huggingface.co/datasets/AINovice2005/carbon-pilot-corpus-dedup/resolve/main/*.parquet' \
        --likelihood-glob 'https://huggingface.co/datasets/AINovice2005/carbon-likelihood-stats/resolve/main/*.parquet' \
        --embeddings-glob 'https://huggingface.co/datasets/AINovice2005/carbon-embeddings/resolve/main/*.parquet'
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json
import re

CLICKHOUSE_BIN = "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment/clickhouse"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def check_binary() -> None:
    if shutil.which(CLICKHOUSE_BIN) is None:
        sys.exit(
            "clickhouse binary not found on PATH. Install with:\n"
            "  curl https://clickhouse.com/ | sh\n"
            "then re-run (this script uses `clickhouse local`)."
        )


def discover_hf_shards(url_glob: str) -> list[str]:
    if "*" not in url_glob and "?" not in url_glob and "{" not in url_glob:
        return [url_glob]

    match = re.match(
        r"^(https://huggingface\.co/datasets/[^/]+/[^/]+)/resolve/main/(.+)$",
        url_glob,
    )
    if not match:
        raise ValueError(
            "Wildcard expansion is only implemented for Hugging Face dataset URLs "
            "of the form https://huggingface.co/datasets/<org>/<repo>/resolve/main/..."
        )

    repo = match.group(1).split("/datasets/", 1)[1]
    pattern = match.group(2)

    api_url = (
        f"https://huggingface.co/api/datasets/{repo}/tree/main"
        "?recursive=true&limit=1000"
    )

    req = Request(api_url, headers={"User-Agent": "build_cohort_pipeline/1.0"})
    try:
        with urlopen(req, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query Hugging Face dataset tree for {repo}: {exc}"
        ) from exc

    shard_paths = []
    for entry in payload:
        path = entry.get("path", "")
        if path.endswith(".parquet") and re.fullmatch(
            r"shard-\d+\.parquet", path
        ):
            if pattern == "*.parquet":
                shard_paths.append(path)

    if not shard_paths:
        raise RuntimeError(
            f"No Parquet shards matched {url_glob} in Hugging Face dataset {repo}."
        )

    shard_paths.sort()
    return [
        f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
        for path in shard_paths
    ]


def clickhouse_shard_url(url_glob: str) -> str:
    urls = discover_hf_shards(url_glob)
    if len(urls) == 1:
        return urls[0]

    parsed = [urlparse(u) for u in urls]
    names = [Path(p.path).name for p in parsed]

    match = re.fullmatch(r"shard-(\d+)\.parquet", names[0])
    if not match:
        return "{" + ",".join(urls) + "}"

    width = len(match.group(1))
    numbers = [int(re.fullmatch(r"shard-(\d+)\.parquet", name).group(1)) for name in names]
    contiguous = numbers == list(range(numbers[0], numbers[-1] + 1))

    base = urls[0].rsplit(names[0], 1)[0]
    if contiguous:
        return f"{base}shard-{{{numbers[0]:0{width}d}..{numbers[-1]:0{width}d}}}.parquet"

    names_text = ",".join(f"{n:0{width}d}" for n in numbers)
    return f"{base}shard-{{{names_text}}}.parquet"


def count_csv_rows(path: Path) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in f) - 1


def validate_cohort_header(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header is None:
        raise ValueError(f"cohort keys file is empty: {path}")
    if len(header) < 3:
        raise ValueError("cohort keys file must have at least 3 columns")

    expected = ["record_id", "start", "end"]
    normalized = [name.strip().lstrip("\ufeff") for name in header[:3]]
    if normalized != expected:
        raise ValueError(
            f"cohort keys file must begin with columns "
            f"record_id,start,end; found: {header[:3]}"
        )
    return len(header)


def run_clickhouse_local(query: str, extra_args=None) -> subprocess.CompletedProcess:
    cmd = [CLICKHOUSE_BIN, "local", "--query", query]
    if extra_args:
        cmd += extra_args
    return subprocess.run(cmd, capture_output=True, text=True)


def stage(name: str, fn):
    log(f"START  {name}")
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    log(f"DONE   {name}  ({elapsed:.2f}s)")
    return result, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-keys", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--cpu-glob", required=True)
    ap.add_argument("--likelihood-glob", required=True)
    ap.add_argument("--embeddings-glob", required=True)
    ap.add_argument("--max-threads", default="8")
    ap.add_argument("--join-algorithm", default="grace_hash")
    ap.add_argument("--knn", action="store_true")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--knn-top-n", type=int, default=200)
    ap.add_argument("--knn-output", type=Path, default=None)
    args = ap.parse_args()

    if args.knn_output is None:
        args.knn_output = args.output.with_suffix(".knn.csv")

    check_binary()

    if not args.cohort_keys.exists():
        sys.exit(f"cohort keys file not found: {args.cohort_keys}")

    try:
        num_cols = validate_cohort_header(args.cohort_keys)
    except ValueError as exc:
        sys.exit(str(exc))

    n_keys, t_count = stage("count cohort keys", lambda: count_csv_rows(args.cohort_keys))
    log(f"  cohort keys: {n_keys} rows  ({n_keys / t_count:,.0f} rows/s to count)")

    cpu_url = clickhouse_shard_url(args.cpu_glob)
    likelihood_url = clickhouse_shard_url(args.likelihood_glob)
    embeddings_url = clickhouse_shard_url(args.embeddings_glob)

    log(f"  CPU source       : {cpu_url}")
    log(f"  Likelihood source: {likelihood_url}")
    log(f"  Embeddings source: {embeddings_url}")

    cohort_schema = ", ".join(f"c{i} String" for i in range(num_cols))

    # ---- Stage 1: Explicit projection & USING limits AST ambiguity ----
    query = f"""
    WITH cohort_keys AS (
        SELECT
            toString(c0) AS record_id,
            toInt64(c1)  AS start,
            toInt64(c2)  AS end
        FROM file('{args.cohort_keys}', CSV, '{cohort_schema}')
        WHERE toInt64OrNull(c1) IS NOT NULL
    ),
    cpu AS (
        SELECT 
            record_id, start, end, 
            taxonomy, sequence_length, gc_content, shannon_entropy, is_coding_region
        FROM url('{cpu_url}', Parquet)
        INNER JOIN cohort_keys USING (record_id, start, end)
    ),
    lik AS (
        SELECT 
            record_id, start, end, 
            mean_log_prob, sum_log_prob, perplexity
        FROM url('{likelihood_url}', Parquet)
        INNER JOIN cohort_keys USING (record_id, start, end)
    ),
    emb AS (
        SELECT 
            record_id, start, end, 
            embedding
        FROM url('{embeddings_url}', Parquet)
        INNER JOIN cohort_keys USING (record_id, start, end)
    )
    SELECT
        record_id, start, end, taxonomy,
        sequence_length, gc_content, shannon_entropy, is_coding_region,
        mean_log_prob, sum_log_prob, perplexity,
        percent_rank() OVER (PARTITION BY taxonomy ORDER BY sequence_length)  AS within_taxon_length_percentile,
        percent_rank() OVER (PARTITION BY taxonomy ORDER BY gc_content)      AS within_taxon_gc_percentile,
        percent_rank() OVER (PARTITION BY taxonomy ORDER BY shannon_entropy) AS within_taxon_entropy_percentile,
        percent_rank() OVER (PARTITION BY taxonomy ORDER BY mean_log_prob)  AS within_taxon_likelihood_percentile,
        (mean_log_prob - avg(mean_log_prob) OVER (PARTITION BY taxonomy))
            / nullIf(stddevPop(mean_log_prob) OVER (PARTITION BY taxonomy), 0) AS likelihood_zscore,
        greatest(
            abs(percent_rank() OVER (PARTITION BY taxonomy ORDER BY sequence_length) - 0.5),
            abs(percent_rank() OVER (PARTITION BY taxonomy ORDER BY gc_content) - 0.5),
            abs(percent_rank() OVER (PARTITION BY taxonomy ORDER BY shannon_entropy) - 0.5)
        ) AS bio_extremeness
    FROM cpu
    INNER JOIN lik USING (record_id, start, end)
    INNER JOIN emb USING (record_id, start, end)
    ORDER BY record_id, start, end
    INTO OUTFILE '{args.output}'
    FORMAT CSVWithNames
    """

    extra_args = [
        "--max_threads", args.max_threads,
        "--join_algorithm", args.join_algorithm,
        "--joined_subquery_requires_alias", "0",
        "--allow_experimental_url_wildcard_from_index_pages", "1",
        "--max_http_get_redirects", "10",
        "--progress",
    ]

    result, t_build = stage("build candidate-screening table (clickhouse-local)",
                             lambda: run_clickhouse_local(query, extra_args))

    if result.returncode != 0:
        log("clickhouse-local FAILED:")
        log(result.stderr)
        sys.exit(1)

    if not args.output.exists():
        sys.exit("clickhouse-local reported success but no output file was written.")

    n_out = count_csv_rows(args.output)
    throughput = n_out / t_build if t_build > 0 else float("inf")

    log("---- summary -------------------------------------------------")
    log(f"  cohort keys           : {n_keys}")
    log(f"  output rows            : {n_out}")
    log(f"  wall clock (join+score): {t_build:.2f}s")
    log(f"  throughput              : {throughput:,.1f} rows/s")
    log(f"  output file             : {args.output}")

    # ---- Stage 2 (optional): embedding-neighbor search for top candidates --
    if args.knn:
        knn_query = f"""
        WITH candidates AS (
            SELECT record_id, start, end, taxonomy,
                   (bio_extremeness + abs(coalesce(likelihood_zscore, 0))) AS candidate_score
            FROM file('{args.output}', CSVWithNames)
            ORDER BY candidate_score DESC
            LIMIT {args.knn_top_n}
        ),
        candidate_emb AS (
            SELECT
                c.record_id,
                c.start,
                c.end,
                c.taxonomy,
                arrayMap(
                    x -> ifNull(x, toFloat32(0)),
                    e.embedding
                ) AS embedding
            FROM candidates AS c
        INNER JOIN url('{embeddings_url}', Parquet) AS e
            ON c.record_id = e.record_id
            AND c.start = e.start
            AND c.end = e.end
        ),
        full_emb AS (
            SELECT
            e.record_id,
            e.start,
            e.end,
            arrayMap(
                x -> ifNull(x, toFloat32(0)),
                e.embedding
            ) AS embedding,
            t.taxonomy
        FROM url('{embeddings_url}', Parquet) AS e
        INNER JOIN url('{cpu_url}', Parquet) AS t
            ON e.record_id = t.record_id
            AND e.start = t.start
            AND e.end = t.end
        ),
        neighbors AS (
            SELECT
                c.record_id  AS candidate_record_id,
                c.start      AS candidate_start,
                c.end        AS candidate_end,
                c.taxonomy   AS candidate_taxonomy,
                f.record_id  AS neighbor_record_id,
                f.taxonomy   AS neighbor_taxonomy,
                cosineDistance(c.embedding, f.embedding) AS distance
            FROM candidate_emb AS c
            CROSS JOIN full_emb AS f
            WHERE NOT (c.record_id = f.record_id AND c.start = f.start AND c.end = f.end)
            ORDER BY candidate_record_id, candidate_start, candidate_end, distance
            LIMIT {args.knn_k} BY candidate_record_id, candidate_start, candidate_end
        )
        SELECT
            candidate_record_id, candidate_start, candidate_end, candidate_taxonomy,
            count() AS k,
            countIf(neighbor_taxonomy = candidate_taxonomy) AS same_taxon_neighbors,
            countIf(neighbor_taxonomy = candidate_taxonomy) / count() AS same_taxon_fraction,
            groupArray(neighbor_record_id) AS neighbor_record_ids,
            groupArray(neighbor_taxonomy) AS neighbor_taxonomies,
            groupArray(round(distance, 4)) AS neighbor_distances
        FROM neighbors
        GROUP BY candidate_record_id, candidate_start, candidate_end, candidate_taxonomy
        ORDER BY same_taxon_fraction ASC
        INTO OUTFILE '{args.knn_output}'
        FORMAT CSVWithNames
        """

        knn_extra_args = [
            "--max_threads", args.max_threads,
            "--joined_subquery_requires_alias", "0",
            "--max_http_get_redirects", "10",
            "--progress",
        ]

        knn_result, t_knn = stage(
            f"embedding KNN for top {args.knn_top_n} candidates (k={args.knn_k})",
            lambda: run_clickhouse_local(knn_query, knn_extra_args),
        )

        if knn_result.returncode != 0:
            log("clickhouse-local FAILED (knn stage):")
            log(knn_result.stderr)
            sys.exit(1)

        if not args.knn_output.exists():
            sys.exit("KNN stage reported success but no output file was written.")

        n_knn = count_csv_rows(args.knn_output)
        knn_throughput = n_knn / t_knn if t_knn > 0 else float("inf")

        log("---- knn summary ----------------------------------------------")
        log(f"  candidates screened     : {args.knn_top_n}")
        log(f"  k                       : {args.knn_k}")
        log(f"  candidates written      : {n_knn}")
        log(f"  wall clock (knn)        : {t_knn:.2f}s")
        log(f"  throughput              : {knn_throughput:,.1f} candidates/s")
        log(f"  output file             : {args.knn_output}")


if __name__ == "__main__":
    main()