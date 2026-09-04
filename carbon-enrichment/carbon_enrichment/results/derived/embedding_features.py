from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
from datasets import load_dataset

from carbon_enrichment.resources.lancedb import (
    LanceDBConfig,
    LanceDBResource,
)

# ============================================================================
# Configuration
# ============================================================================

LOCAL_DATASET_PATH = Path(
    "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment/"
    "carbon-enrichment/data/carbon-embeddings"
)

LOCAL_DATASET_GLOB = "**/*.parquet"

LANCEDB_URI = "./data/lancedb_store"
LANCEDB_TABLE = "carbon_embeddings"

VECTOR_SIZE = 3072

# ============================================================================
# Ingestion
# ============================================================================

INGEST_BATCH_SIZE = 4096

# Define the strict PyArrow schema required by LanceDB
SCHEMA = pa.schema([
    pa.field("source_row_index", pa.int64()),
    pa.field("record_id", pa.string()),
    pa.field("start", pa.int64()),
    pa.field("end", pa.int64()),
    pa.field("string_length", pa.int64(), nullable=True),
    pa.field("embedding_norm", pa.float64(), nullable=True),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_SIZE)),
])

# ============================================================================
# Analysis
# ============================================================================

ANALYSIS_BATCH_SIZE = 256
K = 10
DENSITY_THRESHOLD = 0.85
DENSITY_LIMIT = 10_000

# LanceDB releases the GIL during search. This ThreadPool will truly parallelize
# vector searches across your CPU cores while CSV I/O happens on the main thread.
ANALYSIS_PREFETCH = 4

# ============================================================================
# Output
# ============================================================================

OUTPUT_DIR = Path("results/case_study/phase4")
METRICS_OUTPUT = OUTPUT_DIR / "lancedb_metrics.csv"
KNN_OUTPUT = OUTPUT_DIR / "knn_neighbors.csv"

RECREATE_TABLE = True

# ============================================================================
# Helpers
# ============================================================================

def normalize_embedding(value: Any) -> List[float]:
    """Convert an HF embedding into a plain Python float list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(x) for x in value]

def open_local_dataset():
    if not LOCAL_DATASET_PATH.exists():
        print(f"\nERROR: LOCAL_DATASET_PATH does not exist: {LOCAL_DATASET_PATH}")
        sys.exit(1)

    parquet_files = sorted(str(p) for p in LOCAL_DATASET_PATH.glob(LOCAL_DATASET_GLOB))
    if not parquet_files:
        print(f"\nERROR: No parquet files found under {LOCAL_DATASET_PATH}.")
        sys.exit(1)

    print(f"Loading {len(parquet_files)} local parquet file(s)...")
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    print(f"Loaded {len(dataset):,} rows.")
    return dataset

# ============================================================================
# Phase 4 analyzer
# ============================================================================

class Phase4EmbeddingAnalyzer:
    """Phase 4 embedding analysis layer using LanceDB."""

    def __init__(self, db_resource: LanceDBResource) -> None:
        self.db_resource = db_resource
        self.table = self.db_resource.get_db().open_table(LANCEDB_TABLE)

    def compute_density_and_knn_batch(
        self, query_batch: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
        """
        Calculates both Density and KNN instantly using PyArrow and NumPy.
        Avoids deserializing 10,000 dicts per query into pure Python memory.
        """
        densities: Dict[int, int] = {}
        knn_results: List[Dict[str, Any]] = []

        for item in query_batch:
            own_row_index = item["source_row_index"]

            # Pull as PyArrow table
            hits = (
                self.table.search(item["vector"])
                .metric("cosine")
                .limit(DENSITY_LIMIT)
                .select(["source_row_index", "record_id", "start", "end", "_distance"])
                .to_arrow()
            )

            # Fast vector math using NumPy
            hit_indices = hits["source_row_index"].to_numpy()
            cosine_distances = hits["_distance"].to_numpy()
            cosine_similarities = 1.0 - cosine_distances

            # Mask out the query observation itself
            mask = hit_indices != own_row_index
            
            valid_similarities = cosine_similarities[mask]
            valid_indices = hit_indices[mask]
            
            # 1. Density Check: sum the boolean array instantly
            density_count = np.sum(valid_similarities >= DENSITY_THRESHOLD)
            densities[own_row_index] = int(density_count)

            # 2. Top-K Check: Slice the top K results (already sorted by LanceDB)
            top_k_sims = valid_similarities[:K]
            top_k_dists = cosine_distances[mask][:K]
            top_k_indices = valid_indices[:K]
            
            # Map back to the original Arrow table index to fetch metadata
            valid_arrow_indices = np.where(mask)[0]

            for rank in range(len(top_k_indices)):
                arrow_idx = valid_arrow_indices[rank] 
                
                knn_results.append({
                    "query_row_index": own_row_index,
                    "query_record_id": item["record_id"],
                    "query_start": item["start"],
                    "query_end": item["end"],
                    "neighbor_rank": rank + 1,
                    "neighbor_row_index": top_k_indices[rank],
                    "neighbor_record_id": str(hits["record_id"][arrow_idx].as_py()),
                    "neighbor_start": hits["start"][arrow_idx].as_py(),
                    "neighbor_end": hits["end"][arrow_idx].as_py(),
                    "cosine_similarity": float(top_k_sims[rank]),
                    "cosine_distance": float(top_k_dists[rank]),
                })

        return densities, knn_results

# ============================================================================
# Ingestion
# ============================================================================

def ingest_embeddings(db_resource: LanceDBResource, dataset) -> tuple[int, int]:
    print("\n" + "=" * 72)
    print("PHASE 4 — LANCEDB INGESTION")
    print("=" * 72)

    if RECREATE_TABLE:
        print("\nRecreating LanceDB table...")
        db_resource.recreate_table(schema=SCHEMA, table_name=LANCEDB_TABLE)
    else:
        db_resource.ensure_table(schema=SCHEMA, table_name=LANCEDB_TABLE)

    batch: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0
    source_row_index = 0
    start_time = time.time()

    for row in dataset:
        source_row_index += 1

        record_id = row.get("record_id")
        start = row.get("start")
        end = row.get("end")
        vector = normalize_embedding(row.get("embedding"))

        if record_id is None or start is None or end is None or not vector:
            skipped += 1
            continue

        if len(vector) != VECTOR_SIZE:
            skipped += 1
            continue

        batch.append({
            "source_row_index": source_row_index,
            "record_id": str(record_id),
            "start": start,
            "end": end,
            "string_length": row.get("string_lengths"),
            "embedding_norm": row.get("embedding_norm"),
            "vector": vector,
        })

        if len(batch) >= INGEST_BATCH_SIZE:
            pa_table = pa.Table.from_pylist(batch, schema=SCHEMA)
            db_resource.upsert_points(pa_table)
            
            processed += len(batch)
            batch = []
            
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(f"Submitted: {processed:,} | Skipped: {skipped:,} | Rate: {rate:,.1f} rec/s")

    if batch:
        pa_table = pa.Table.from_pylist(batch, schema=SCHEMA)
        db_resource.upsert_points(pa_table)
        processed += len(batch)

    # ----------------------------------------------------------------------
    # INDEX BUILD
    # ----------------------------------------------------------------------
    print("\nBuilding vector index (this may take a moment)...")
    table = db_resource.get_db().open_table(LANCEDB_TABLE)
    table.create_index(metric="cosine", vector_column_name="vector")
    
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0

    actual_count = db_resource.count()

    print("\n" + "-" * 72)
    print("INGESTION COMPLETE")
    print(f"Records valid    : {processed:,}")
    print(f"Table count      : {actual_count:,}")
    print(f"Elapsed          : {elapsed:,.1f} sec")
    
    if actual_count != processed:
        print("\nERROR: Table count does not match valid ingested observations.")
        sys.exit(1)

    return processed, skipped

# ============================================================================
# Analysis
# ============================================================================

def analyze_embeddings(db_resource: LanceDBResource, dataset) -> tuple[int, int, int]:
    print("\n" + "=" * 72)
    print("PHASE 4 — LANCEDB ANALYSIS")
    print("=" * 72)

    analyzer = Phase4EmbeddingAnalyzer(db_resource)
    
    processed = 0
    skipped = 0
    knn_rows = 0
    source_row_index = 0
    start_time = time.time()

    def build_batches():
        nonlocal skipped, source_row_index
        batch: List[Dict[str, Any]] = []

        for row in dataset:
            source_row_index += 1
            record_id = row.get("record_id")
            start = row.get("start")
            end = row.get("end")
            vector = normalize_embedding(row.get("embedding"))

            if record_id is None or start is None or end is None or not vector or len(vector) != VECTOR_SIZE:
                skipped += 1
                continue

            batch.append({
                "source_row_index": source_row_index,
                "record_id": str(record_id),
                "start": start,
                "end": end,
                "string_length": row.get("string_lengths"),
                "embedding_norm": row.get("embedding_norm"),
                "vector": vector,
            })

            if len(batch) >= ANALYSIS_BATCH_SIZE:
                yield batch
                batch = []

        if batch:
            yield batch

    def write_batch_results(batch, densities, knn_results, metrics_writer, knn_writer):
        for item in batch:
            idx = item["source_row_index"]
            metrics_writer.writerow({
                "source_row_index": idx,
                "record_id": item["record_id"],
                "start": item["start"],
                "end": item["end"],
                "string_length": item["string_length"],
                "embedding_norm": item["embedding_norm"],
                "local_density": densities.get(idx, 0),
            })
        knn_writer.writerows(knn_results)

    with (
        open(METRICS_OUTPUT, "w", newline="", encoding="utf-8") as metrics_file,
        open(KNN_OUTPUT, "w", newline="", encoding="utf-8") as knn_file,
    ):
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=[
            "source_row_index", "record_id", "start", "end", 
            "string_length", "embedding_norm", "local_density"
        ])
        knn_writer = csv.DictWriter(knn_file, fieldnames=[
            "query_row_index", "query_record_id", "query_start", "query_end",
            "neighbor_rank", "neighbor_row_index", "neighbor_record_id", 
            "neighbor_start", "neighbor_end", "cosine_similarity", "cosine_distance"
        ])

        metrics_writer.writeheader()
        knn_writer.writeheader()

        with ThreadPoolExecutor(max_workers=ANALYSIS_PREFETCH) as executor:
            pending = []
            batch_iter = build_batches()

            def submit_next():
                try:
                    b = next(batch_iter)
                    future = executor.submit(analyzer.compute_density_and_knn_batch, b)
                    pending.append((b, future))
                    return True
                except StopIteration:
                    return False

            for _ in range(ANALYSIS_PREFETCH):
                if not submit_next():
                    break

            while pending:
                batch, future = pending.pop(0)
                densities, knn_results = future.result()
                
                submit_next()

                write_batch_results(batch, densities, knn_results, metrics_writer, knn_writer)
                
                processed += len(batch)
                knn_rows += len(knn_results)
                
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                
                print(f"Analyzed: {processed:,} | KNN rows: {knn_rows:,} | Rate: {rate:,.1f} rec/s")

    return processed, skipped, knn_rows

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = LanceDBConfig(uri=LANCEDB_URI, table_name=LANCEDB_TABLE, distance="cosine")
    dataset = open_local_dataset()

    with LanceDBResource(config) as db:
        ingest_embeddings(db, dataset)
        analyze_embeddings(db, dataset)

if __name__ == "__main__":
    main()