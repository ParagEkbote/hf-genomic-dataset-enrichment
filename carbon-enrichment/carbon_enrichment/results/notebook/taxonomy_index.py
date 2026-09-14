"""
Batch builder for the local record_id -> taxonomy index.

Why this exists
----------------
NCBIClient's in-memory cache (see ncbi_client.py) only helps once a
tax_id has been looked up at least once in the running process. It
does nothing for questions like "which of these 30,000 records share
a genus with the selected record?" — answering that requires knowing
every record's tax_id *before* any record is selected.

This module resolves record_id -> tax_id / lineage for a full cohort
once, offline, and writes the result to a local parquet file. The
Marimo notebook then loads that file directly (no network calls) and
uses it for the taxonomic-cohort comparisons in the specimen explorer.

This is a batch job, not part of the interactive path:

    build once (this module, run ahead of time / on a schedule)
        -> taxonomy_index.parquet
            -> loaded by the notebook at startup (cheap, local, static)

Usage
-----
    from taxonomy_index import build_taxonomy_index, load_taxonomy_index

    build_taxonomy_index(
        record_ids=common_record_ids,
        output_path="taxonomy_index.parquet",
    )

    index_df = load_taxonomy_index("taxonomy_index.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ncbi_client import NCBIClient, NCBIError

logger = logging.getLogger(__name__)

# How often to flush progress to disk. A crash or interruption partway
# through a large cohort should not lose all prior work.
CHECKPOINT_EVERY = 200

# Columns written for every resolved record. classification_columns are
# added dynamically based on whichever ranks NCBI actually returned.
BASE_COLUMNS = [
    "record_id",
    "tax_id",
    "scientific_name",
    "common_name",
    "rank",
    "assembly_accession",
    "status",
    "error",
]


def _row_for_record(client: NCBIClient, record_id: str) -> dict:
    """Resolve one record_id and flatten it into an index row."""

    try:
        specimen = client.resolve_record(record_id)

    except NCBIError as exc:
        return {
            "record_id": record_id,
            "tax_id": None,
            "scientific_name": None,
            "common_name": None,
            "rank": None,
            "assembly_accession": None,
            "status": "error",
            "error": str(exc),
        }

    row = {
        "record_id": record_id,
        "tax_id": specimen.tax_id,
        "scientific_name": specimen.scientific_name,
        "common_name": specimen.common_name,
        "rank": specimen.rank,
        "assembly_accession": specimen.assembly.accession,
        "status": "ok",
        "error": None,
    }

    # Flatten classification (e.g. genus_name, genus_tax_id, ...) so the
    # notebook can filter cohorts with a plain column comparison instead
    # of parsing a nested structure per row.
    row.update(specimen.classification)

    return row


def build_taxonomy_index(
    record_ids: list[str],
    *,
    output_path: str | Path,
    api_key: str | None = None,
    resume: bool = True,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> pd.DataFrame:
    """
    Resolve record_id -> taxonomy for every id in record_ids and write
    the result to output_path as parquet.

    Resumable: if output_path already exists and resume=True, record
    ids already present (regardless of ok/error status) are skipped.
    Re-run the same call to pick up where a previous run left off, or
    to add newly-arrived record ids to an existing index.

    Records that fail NCBI resolution are still written with
    status="error" and the error message, rather than being silently
    dropped — that way a later re-run can distinguish "not yet
    attempted" from "attempted and failed", and the notebook can show
    NCBI-unavailable records explicitly instead of them just vanishing
    from cohort comparisons.
    """

    output_path = Path(output_path)

    existing_df: pd.DataFrame | None = None
    already_done: set[str] = set()

    if resume and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        already_done = set(existing_df["record_id"])

    todo = [rid for rid in record_ids if rid not in already_done]

    logger.info(
        "taxonomy index: %d total, %d already resolved, %d to resolve",
        len(record_ids),
        len(already_done),
        len(todo),
    )

    if not todo:
        return existing_df if existing_df is not None else pd.DataFrame(
            columns=BASE_COLUMNS
        )

    rows: list[dict] = []

    with NCBIClient(api_key=api_key) as client:
        for i, record_id in enumerate(todo, start=1):
            rows.append(_row_for_record(client, record_id))

            if i % checkpoint_every == 0:
                _flush(existing_df, rows, output_path)
                logger.info("taxonomy index: checkpointed at %d/%d", i, len(todo))

    return _flush(existing_df, rows, output_path)


def _flush(
    existing_df: pd.DataFrame | None,
    new_rows: list[dict],
    output_path: Path,
) -> pd.DataFrame:
    """Merge new_rows into existing_df (if any) and write to output_path."""

    new_df = pd.DataFrame(new_rows)

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="record_id", keep="last")
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    return combined


def load_taxonomy_index(path: str | Path) -> pd.DataFrame:
    """Load a previously built taxonomy index. Pure local read, no network."""

    return pd.read_parquet(path)


def cohort_for_rank(
    index_df: pd.DataFrame,
    *,
    record_id: str,
    rank: str,
) -> pd.DataFrame:
    """
    Return the subset of index_df sharing the given record's value at
    `rank` (e.g. rank="genus_name" -> all records in the same genus).

    Raises KeyError if the rank column isn't present, and ValueError if
    the record_id isn't in the index or has no value at that rank.
    """

    if rank not in index_df.columns:
        raise KeyError(f"No '{rank}' column in taxonomy index.")

    match = index_df.loc[index_df["record_id"] == record_id]

    if match.empty:
        raise ValueError(f"record_id {record_id!r} not found in taxonomy index.")

    value = match.iloc[0][rank]

    if pd.isna(value):
        raise ValueError(
            f"record_id {record_id!r} has no value for rank '{rank}'."
        )

    return index_df.loc[index_df[rank] == value]


__all__ = [
    "build_taxonomy_index",
    "load_taxonomy_index",
    "cohort_for_rank",
]