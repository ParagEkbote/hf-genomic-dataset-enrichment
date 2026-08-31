"""
Build the 100K embeddings fixture for validate_resources.py.

Source
------
The full ``AINovice2005/carbon-embeddings`` dataset on the Hugging Face
Hub (remote Parquet shards). Schema (per the dataset README):

    record_id       string
    string_lengths  int64
    start           int64
    end             int64
    embedding       list
    embedding_norm  float32

Row identity
------------
The embeddings dataset carries the same logical key as the validation
cohort: (record_id, start, end). This is the important correction from
just joining on record_id alone -- record_id repeats within the 100K
cohort (100,000 rows over ~44,180 distinct record_id values), so a
record_id-only join would either fan out or require extra de-duplication
bookkeeping. Joining on the full (record_id, start, end) triple instead
gives an exact 1:1 match against validation_row_id by construction, with
no ambiguity.

Shard discovery
----------------
The dataset's file listing shows a "Load more files" control beyond
shard-00047.parquet, and this has been confirmed in practice: a prior
run's HTTP 429 came from shard-00118.parquet, so the source has well
over 48 shards. This script discovers shards via DuckDB's hf:// glob
support rather than hardcoding a shard count or enumerating URLs by
hand. If your DuckDB build predates hf:// support, pass --shard-glob
with an explicit HTTPS glob/URL list instead (see --help).

Rate limiting
-------------
Hugging Face's CDN rate-limits anonymous requests. A 100+-shard glob
scan can trip HTTP 429 partway through even within a single run, and
DuckDB has no cross-run resume for a failed read_parquet() glob -- a
failure aborts the whole query, so the scan restarts from shard 0 next
time. Two mitigations, both exposed as CLI flags: (1) pass --hf-token
(or set HF_TOKEN) to authenticate, which raises the rate limit
substantially; (2) --http-retries/--http-retry-wait-ms/
--http-retry-backoff configure automatic retry with exponential backoff
for transient 429s within a run.

Output
------
data/validation/100k/embeddings.parquet, with columns:
    validation_row_id
    record_id
    embedding

This is the exact fixture contract validate_resources.py's
load_embeddings() expects.

The script verifies invariants on both sides of the join (source
dedup/coverage, cohort match completeness/uniqueness) and refuses to
write the fixture unless all pass.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import duckdb


LOGGER = logging.getLogger("create_embeddings_fixture")

EXPECTED_ROWS = 100_000

DEFAULT_HF_GLOB = "hf://datasets/AINovice2005/carbon-embeddings/*.parquet"


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Join the 100K validation cohort against the remote "
            "carbon-embeddings dataset to build embeddings.parquet."
        )
    )

    parser.add_argument(
        "--cpu-path",
        type=Path,
        required=True,
        help=(
            "Path to the validation cohort's cpu.parquet (must contain "
            "validation_row_id, record_id, start, end)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/100k/embeddings.parquet"),
        help="Output path for the embeddings fixture.",
    )

    parser.add_argument(
        "--shard-glob",
        default=DEFAULT_HF_GLOB,
        help=(
            "DuckDB-readable glob/URL for the source embeddings shards. "
            f"Default uses DuckDB's hf:// support: {DEFAULT_HF_GLOB!r}. "
            "If your DuckDB build lacks hf:// support, pass an explicit "
            "HTTPS glob such as "
            "'https://huggingface.co/datasets/AINovice2005/"
            "carbon-embeddings/resolve/main/shard-*.parquet' (requires "
            "the httpfs extension) or a local path if you've already "
            "downloaded the shards."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "DuckDB worker threads. Note: higher thread counts issue "
            "more concurrent HTTP requests to Hugging Face's CDN during "
            "the remote shard scan, which increases the chance of "
            "hitting rate limits (HTTP 429). If you hit 429s, try "
            "lowering this (e.g. 1-2) rather than raising it."
        ),
    )

    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Optional DuckDB memory limit, e.g. 8GB.",
    )

    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face access token. Anonymous requests to the HF "
            "CDN are rate-limited more aggressively than authenticated "
            "ones, so passing a token (or setting the HF_TOKEN "
            "environment variable, which is picked up automatically) "
            "substantially reduces the chance of HTTP 429s during the "
            "shard scan. Get one at https://huggingface.co/settings/tokens"
        ),
    )

    parser.add_argument(
        "--http-retries",
        type=int,
        default=6,
        help=(
            "Number of automatic retries DuckDB performs on transient "
            "HTTP errors (including 429) during the remote scan. "
            "Default: 6."
        ),
    )

    parser.add_argument(
        "--http-retry-wait-ms",
        type=int,
        default=2_000,
        help="Initial wait between HTTP retries, in ms. Default: 2000.",
    )

    parser.add_argument(
        "--http-retry-backoff",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to the retry wait after each failed "
            "attempt (exponential backoff). Default: 2.0."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


def configure_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    threads: int | None,
    memory_limit: str | None,
    http_retries: int,
    http_retry_wait_ms: int,
    http_retry_backoff: float,
) -> None:
    """Install/load extensions and apply resource limits.

    http_retries/http_retry_wait_ms/http_retry_backoff matter in
    practice: Hugging Face's CDN rate-limits anonymous requests (HTTP
    429), and a 48+-shard glob scan can trip that limit partway through
    even on a single run. DuckDB has no cross-run resume for a failed
    read_parquet() glob -- a failure aborts the whole query -- so
    surviving *transient* 429s via retry is the only way to avoid
    restarting the entire scan from shard 0.
    """
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")

    try:
        connection.execute("INSTALL hf")
        connection.execute("LOAD hf")
    except duckdb.Error:
        # Older DuckDB builds route hf:// through httpfs directly and
        # don't have a separate 'hf' extension; that's fine.
        LOGGER.debug(
            "No separate 'hf' extension available; continuing with "
            "httpfs only."
        )

    connection.execute(f"SET http_retries={int(http_retries)}")
    connection.execute(f"SET http_retry_wait_ms={int(http_retry_wait_ms)}")
    connection.execute(
        f"SET http_retry_backoff={float(http_retry_backoff)}"
    )

    if threads is not None:
        connection.execute(f"SET threads={int(threads)}")

    if memory_limit is not None:
        connection.execute(f"SET memory_limit='{memory_limit}'")


def register_cohort(
    connection: duckdb.DuckDBPyConnection,
    cpu_path: Path,
) -> int:
    """Register the validation cohort and verify required columns exist."""
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW cohort AS
        SELECT validation_row_id, record_id, start, "end"
        FROM read_parquet('{cpu_path.as_posix()}')
        """
    )

    row_count = int(
        connection.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    )

    if row_count != EXPECTED_ROWS:
        raise RuntimeError(
            f"Cohort at {cpu_path} has {row_count:,} rows; expected "
            f"{EXPECTED_ROWS:,}."
        )

    distinct_ids = int(
        connection.execute(
            "SELECT COUNT(DISTINCT validation_row_id) FROM cohort"
        ).fetchone()[0]
    )

    if distinct_ids != EXPECTED_ROWS:
        raise RuntimeError(
            f"Cohort at {cpu_path} has {distinct_ids:,} distinct "
            f"validation_row_id values; expected {EXPECTED_ROWS:,}."
        )

    return row_count


def register_source(
    connection: duckdb.DuckDBPyConnection,
    shard_glob: str,
) -> dict[str, Any]:
    """Register the remote embeddings source and report its invariants."""
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW source_embeddings AS
        SELECT *
        FROM read_parquet('{shard_glob}')
        """
    )

    columns = {
        row[0]
        for row in connection.execute(
            "DESCRIBE source_embeddings"
        ).fetchall()
    }

    required = {"record_id", "start", "end", "embedding"}
    missing = required - columns
    if missing:
        raise ValueError(
            "Source embeddings dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    total_rows = int(
        connection.execute(
            "SELECT COUNT(*) FROM source_embeddings"
        ).fetchone()[0]
    )

    distinct_keys = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT DISTINCT record_id, start, "end"
                FROM source_embeddings
            )
            """
        ).fetchone()[0]
    )

    null_embeddings = int(
        connection.execute(
            "SELECT COUNT(*) FROM source_embeddings WHERE embedding IS NULL"
        ).fetchone()[0]
    )

    return {
        "total_rows": total_rows,
        "distinct_keys": distinct_keys,
        "duplicate_keys": total_rows - distinct_keys,
        "null_embeddings": null_embeddings,
    }


def join_and_verify(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Join cohort against source on (record_id, start, end) and verify."""
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE matched AS
        SELECT
            cohort.validation_row_id,
            cohort.record_id,
            source_embeddings.embedding
        FROM cohort
        INNER JOIN source_embeddings
            ON cohort.record_id = source_embeddings.record_id
            AND cohort.start = source_embeddings.start
            AND cohort."end" = source_embeddings."end"
        """
    )

    matched_rows = int(
        connection.execute("SELECT COUNT(*) FROM matched").fetchone()[0]
    )

    distinct_matched_ids = int(
        connection.execute(
            "SELECT COUNT(DISTINCT validation_row_id) FROM matched"
        ).fetchone()[0]
    )

    missing = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT validation_row_id FROM cohort
                EXCEPT
                SELECT validation_row_id FROM matched
            )
            """
        ).fetchone()[0]
    )

    null_embeddings_matched = int(
        connection.execute(
            "SELECT COUNT(*) FROM matched WHERE embedding IS NULL"
        ).fetchone()[0]
    )

    checks = {
        "matched_rows": matched_rows,
        "distinct_matched_validation_row_id": distinct_matched_ids,
        "missing_from_source": missing,
        "null_embeddings_in_match": null_embeddings_matched,
    }

    if matched_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Join produced {matched_rows:,} rows; expected "
            f"{EXPECTED_ROWS:,}. This means either the join fanned out "
            "(duplicate (record_id, start, end) keys in the source) or "
            "under-matched (missing rows). See 'missing_from_source' and "
            "the source's duplicate_keys count for diagnosis."
        )

    if distinct_matched_ids != EXPECTED_ROWS:
        raise RuntimeError(
            f"Join has {distinct_matched_ids:,} distinct "
            f"validation_row_id values; expected {EXPECTED_ROWS:,}. "
            "The join fanned out for at least one cohort row."
        )

    if missing:
        raise RuntimeError(
            f"{missing:,} cohort rows have no matching embedding in the "
            "source dataset. The embeddings dataset does not fully "
            "cover this validation cohort."
        )

    if null_embeddings_matched:
        raise RuntimeError(
            f"{null_embeddings_matched:,} matched rows have a NULL "
            "embedding."
        )

    return checks


def write_fixture(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    *,
    force: bool,
) -> None:
    """Write the embeddings fixture."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not force:
            raise FileExistsError(
                f"{output_path} already exists. Pass --force to overwrite."
            )
        output_path.unlink()

    connection.execute(
        f"""
        COPY (
            SELECT validation_row_id, record_id, embedding
            FROM matched
            ORDER BY validation_row_id
        )
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def main() -> int:
    """Build and verify the 100K embeddings fixture."""
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)

    if not args.cpu_path.exists():
        LOGGER.error("Cohort file does not exist: %s", args.cpu_path)
        return 2

    if args.output.exists() and not args.force:
        LOGGER.error(
            "%s already exists. Pass --force to overwrite.",
            args.output,
        )
        return 2

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    elif not os.environ.get("HF_TOKEN"):
        LOGGER.warning(
            "No HF_TOKEN set. Anonymous requests to the Hugging Face CDN "
            "are rate-limited more aggressively; if this run hits HTTP "
            "429 errors, pass --hf-token or set the HF_TOKEN environment "
            "variable."
        )

    connection = duckdb.connect(database=":memory:")

    try:
        configure_connection(
            connection,
            threads=args.threads,
            memory_limit=args.memory_limit,
            http_retries=args.http_retries,
            http_retry_wait_ms=args.http_retry_wait_ms,
            http_retry_backoff=args.http_retry_backoff,
        )

        LOGGER.info("Registering validation cohort | path=%s", args.cpu_path)
        cohort_rows = register_cohort(connection, args.cpu_path)
        LOGGER.info("Cohort registered | rows=%s", f"{cohort_rows:,}")

        LOGGER.info(
            "Registering remote embeddings source | glob=%s",
            args.shard_glob,
        )

        source_start = time.perf_counter()
        source_stats = register_source(connection, args.shard_glob)
        LOGGER.info(
            "Source registered | total_rows=%s | distinct_keys=%s | "
            "duplicate_keys=%s | null_embeddings=%s | elapsed=%.3fs",
            f"{source_stats['total_rows']:,}",
            f"{source_stats['distinct_keys']:,}",
            f"{source_stats['duplicate_keys']:,}",
            f"{source_stats['null_embeddings']:,}",
            time.perf_counter() - source_start,
        )

        if source_stats["duplicate_keys"]:
            LOGGER.warning(
                "Source has %s duplicate (record_id, start, end) keys. "
                "If any duplicated key also appears in the cohort, the "
                "join below will fail with a fan-out error rather than "
                "silently picking one.",
                f"{source_stats['duplicate_keys']:,}",
            )

        LOGGER.info("Joining cohort against source on (record_id, start, end)")
        join_start = time.perf_counter()
        join_stats = join_and_verify(connection)
        LOGGER.info(
            "Join verified | matched_rows=%s | elapsed=%.3fs",
            f"{join_stats['matched_rows']:,}",
            time.perf_counter() - join_start,
        )

        LOGGER.info("Writing fixture | path=%s", args.output)
        write_fixture(connection, args.output, force=args.force)

    except Exception:
        LOGGER.exception("Embeddings fixture creation failed.")
        return 1

    finally:
        connection.close()

    print()
    print("=" * 64)
    print("100K EMBEDDINGS FIXTURE")
    print("=" * 64)
    print(f"Cohort rows:            {cohort_rows:,}")
    print(f"Source total rows:      {source_stats['total_rows']:,}")
    print(f"Source distinct keys:   {source_stats['distinct_keys']:,}")
    print(f"Source duplicate keys:  {source_stats['duplicate_keys']:,}")
    print(f"Matched rows:           {join_stats['matched_rows']:,}")
    print(f"Output:                 {args.output}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())