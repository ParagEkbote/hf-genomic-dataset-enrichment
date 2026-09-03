from __future__ import annotations

import csv
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from qdrant_client import models

from carbon_enrichment.resources.qdrant import (
    QdrantConfig,
    QdrantResource,
)


# ============================================================================
# Configuration
# ============================================================================

HF_DATASET = "AINovice2005/carbon-embeddings"
HF_SPLIT = "train"

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "carbon_embeddings"

# Confirmed embedding dimensionality.
VECTOR_SIZE = 3072


# ============================================================================
# Ingestion
# ============================================================================

INGEST_BATCH_SIZE = 512


# ============================================================================
# Analysis
# ============================================================================

ANALYSIS_BATCH_SIZE = 256

# Number of nearest OTHER observations.
K = 20

# Fixed-radius cosine similarity threshold.
DENSITY_THRESHOLD = 0.85

# One extra result because Qdrant normally returns the query point itself.
KNN_LIMIT = K + 1

# Maximum number of results requested for fixed-radius density.
DENSITY_LIMIT = 10_000


# ============================================================================
# Output
# ============================================================================

OUTPUT_DIR = Path(
    "results/case_study/phase4"
)

METRICS_OUTPUT = OUTPUT_DIR / "qdrant_metrics.csv"

KNN_OUTPUT = OUTPUT_DIR / "knn_neighbors.csv"


# ============================================================================
# Collection behavior
# ============================================================================

# IMPORTANT:
#
# The previous collection used:
#
#     (record_id, start, end)
#
# as its point identity.
#
# That is NOT sufficient because duplicate composite keys exist and some
# duplicate composite-key rows contain different embeddings.
#
# Therefore the collection MUST be recreated for this corrected run.
RECREATE_COLLECTION = True


# ============================================================================
# Helpers
# ============================================================================

def normalize_embedding(
    value: Any,
) -> List[float]:
    """
    Convert an HF embedding into a plain Python float list.
    """

    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [float(x) for x in value]


def normalize_search_response(
    response: Any,
) -> List[Any]:
    """Normalize Qdrant search_batch responses across return shapes."""
    if response is None:
        return []
    
    if hasattr(response, "points") and response.points is not None:
        return response.points
        
    # Handle legacy BatchResult objects
    if hasattr(response, "result") and response.result is not None:
        return response.result

    if isinstance(response, list):
        return response
    if isinstance(response, tuple):
        for part in response:
            if isinstance(part, (list, tuple)):
                if not part:
                    return []
                if hasattr(part[0], "id"):
                    return list(part)
        raise RuntimeError(
            "Unexpected Qdrant batch search response shape: "
            f"{type(response).__name__}"
        )
    if hasattr(response, "id"):
        return [response]
    raise RuntimeError(
        "Unexpected Qdrant batch search response type: "
        f"{type(response).__name__}"
    )


def qdrant_point_id(
    record_id: str,
    start: Any,
    end: Any,
    source_row_index: int,
) -> str:
    """
    Generate a deterministic UUID5 for the individual source observation.

    IMPORTANT:
    The dataset does not provide a globally unique row identifier.

    Therefore the observation identity is:

        (record_id, start, end, source_row_index)

    source_row_index preserves duplicate rows rather than allowing Qdrant
    upserts to overwrite them.
    """

    observation_key = (
        f"{record_id}\x1f"
        f"{start}\x1f"
        f"{end}\x1f"
        f"{source_row_index}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            observation_key,
        )
    )


def open_hf_stream():
    """
    Open the Hugging Face dataset in streaming mode.
    """

    return load_dataset(
        HF_DATASET,
        split=HF_SPLIT,
        streaming=True,
    )


# ============================================================================
# Phase 4 analyzer
# ============================================================================

class Phase4EmbeddingAnalyzer:
    """
    Phase 4 embedding analysis layer.

    Qdrant must contain the complete embedding population before
    analysis starts.
    """

    def __init__(
        self,
        qdrant: QdrantResource,
    ) -> None:
        self.qdrant = qdrant

    # ----------------------------------------------------------------------
    # Fixed-radius density
    # ----------------------------------------------------------------------

    def compute_fixed_radius_density_batch(
        self,
        query_batch: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Count neighbors with cosine similarity >= DENSITY_THRESHOLD.

        The query point itself is excluded using its unique Qdrant UUID.

        Returns:

            Qdrant point UUID -> local density
        """

        requests = [
            models.QueryRequest(
                query=item["vector"],
                limit=DENSITY_LIMIT,
                score_threshold=DENSITY_THRESHOLD,
                with_payload=False,
            )
            for item in query_batch
        ]

        responses = self.qdrant.search_batch(
            requests,
        )

        densities: Dict[str, int] = {}

        for item, response in zip(
            query_batch,
            responses,
        ):
            record_id = str(
                item["record_id"]
            )

            start = item["start"]
            end = item["end"]
            source_row_index = item["source_row_index"]

            own_point_id = qdrant_point_id(
                record_id,
                start,
                end,
                source_row_index,
            )

            response = normalize_search_response(response)

            count = 0

            for hit in response:

                # Exclude the exact query observation.
                if str(hit.id) == own_point_id:
                    continue

                count += 1

            densities[own_point_id] = count

        return densities

    # ----------------------------------------------------------------------
    # KNN
    # ----------------------------------------------------------------------

    def extract_knn_batch(
        self,
        query_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve K nearest OTHER observations for each query.

        Payload includes source_row_index so duplicate observations remain
        individually identifiable.
        """

        requests = [
            models.QueryRequest(
                query=item["vector"],
                limit=KNN_LIMIT,

                with_payload=[
                    "source_row_index",
                    "record_id",
                    "start",
                    "end",
                ],
            )
            for item in query_batch
        ]

        responses = self.qdrant.search_batch(
            requests,
        )

        results: List[Dict[str, Any]] = []

        for item, response in zip(
            query_batch,
            responses,
        ):
            query_source_row_index = item[
                "source_row_index"
            ]

            query_record_id = str(
                item["record_id"]
            )

            query_start = item["start"]
            query_end = item["end"]

            own_point_id = qdrant_point_id(
                query_record_id,
                query_start,
                query_end,
                query_source_row_index,
            )

            response = normalize_search_response(response)

            rank = 1

            for hit in response:

                # ----------------------------------------------------------
                # Exclude self.
                # ----------------------------------------------------------

                if str(hit.id) == own_point_id:
                    continue

                # ----------------------------------------------------------
                # Validate payload.
                # ----------------------------------------------------------

                if not hit.payload:
                    continue

                neighbor_source_row_index = hit.payload.get(
                    "source_row_index"
                )

                neighbor_record_id = hit.payload.get(
                    "record_id"
                )

                neighbor_start = hit.payload.get(
                    "start"
                )

                neighbor_end = hit.payload.get(
                    "end"
                )

                if (
                    neighbor_source_row_index is None
                    or neighbor_record_id is None
                    or neighbor_start is None
                    or neighbor_end is None
                ):
                    continue

                neighbor_record_id = str(
                    neighbor_record_id
                )

                # ----------------------------------------------------------
                # Write KNN result.
                # ----------------------------------------------------------

                results.append(
                    {
                        "query_row_index": query_source_row_index,
                        "query_record_id": query_record_id,
                        "query_start": query_start,
                        "query_end": query_end,

                        "neighbor_rank": rank,

                        "neighbor_row_index": (
                            neighbor_source_row_index
                        ),
                        "neighbor_record_id": (
                            neighbor_record_id
                        ),
                        "neighbor_start": neighbor_start,
                        "neighbor_end": neighbor_end,

                        "cosine_similarity": hit.score,
                        "cosine_distance": 1.0 - hit.score,
                    }
                )

                rank += 1

                if rank > K:
                    break

        return results


# ============================================================================
# Ingestion
# ============================================================================

def ingest_embeddings(
    qdrant: QdrantResource,
) -> tuple[int, int]:
    """
    Stream the complete HF dataset into Qdrant.

    Every source observation receives a unique source_row_index.

    Qdrant identity:

        (record_id, start, end, source_row_index)

    Returns:

        processed_count, skipped_count
    """

    print()
    print("=" * 72)
    print("PHASE 4 — QDRANT INGESTION")
    print("=" * 72)
    print(f"Dataset          : {HF_DATASET}")
    print(f"Split            : {HF_SPLIT}")
    print(f"Collection       : {QDRANT_COLLECTION}")
    print(f"Vector size      : {VECTOR_SIZE}")
    print(f"Batch size       : {INGEST_BATCH_SIZE}")
    print("Point identity   : (record_id, start, end, source_row_index)")
    print("=" * 72)

    # ----------------------------------------------------------------------
    # Collection creation.
    # ----------------------------------------------------------------------

    if RECREATE_COLLECTION:

        print(
            "\nRecreating Qdrant collection..."
        )

        qdrant.recreate_collection(
            vector_size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        )

        print(
            f"Created '{QDRANT_COLLECTION}'."
        )

    else:

        qdrant.ensure_collection(
            vector_size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        )

    # ----------------------------------------------------------------------
    # Open HF stream.
    # ----------------------------------------------------------------------

    print(
        "\nOpening Hugging Face dataset "
        "in streaming mode..."
    )

    try:

        dataset = open_hf_stream()

    except Exception as exc:

        print(
            "\nERROR: Could not open Hugging Face dataset."
        )

        print(exc)

        sys.exit(1)

    print("HF stream opened.")

    batch: List[models.PointStruct] = []

    processed = 0
    skipped = 0

    # This index represents the original observation position in the
    # streamed HF dataset. It increments for EVERY source row, including
    # rows that are later skipped.
    source_row_index = 0

    start_time = time.time()

    # ----------------------------------------------------------------------
    # Stream dataset.
    # ----------------------------------------------------------------------

    for row in dataset:

        source_row_index += 1

        record_id = row.get("record_id")
        start = row.get("start")
        end = row.get("end")

        string_length = row.get(
            "string_lengths"
        )

        embedding_norm = row.get(
            "embedding_norm"
        )

        vector = normalize_embedding(
            row.get("embedding")
        )

        # --------------------------------------------------------------
        # Validate record ID.
        # --------------------------------------------------------------

        if record_id is None:

            skipped += 1
            continue

        record_id = str(record_id)

        # --------------------------------------------------------------
        # Validate positional identity.
        # --------------------------------------------------------------

        if start is None or end is None:

            skipped += 1
            continue

        # --------------------------------------------------------------
        # Validate embedding.
        # --------------------------------------------------------------

        if not vector:

            skipped += 1
            continue

        if len(vector) != VECTOR_SIZE:

            print(
                f"WARNING: skipping row "
                f"{source_row_index:,} "
                f"{record_id} [{start}, {end}]: "
                f"expected {VECTOR_SIZE} dimensions, "
                f"got {len(vector)}"
            )

            skipped += 1
            continue

        # --------------------------------------------------------------
        # Generate unique deterministic Qdrant ID.
        # --------------------------------------------------------------

        point_id = qdrant_point_id(
            record_id,
            start,
            end,
            source_row_index,
        )

        # --------------------------------------------------------------
        # Construct Qdrant point.
        # --------------------------------------------------------------

        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "source_row_index": source_row_index,
                "record_id": record_id,
                "start": start,
                "end": end,
                "string_length": string_length,
                "embedding_norm": embedding_norm,
            },
        )

        batch.append(point)

        # --------------------------------------------------------------
        # Flush batch.
        # --------------------------------------------------------------

        if len(batch) < INGEST_BATCH_SIZE:
            continue

        qdrant.upsert_points(
            batch,
            wait=True,
        )

        processed += len(batch)

        batch.clear()

        elapsed = time.time() - start_time

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0.0
        )

        print(
            f"Ingested: {processed:,} | "
            f"Source rows: {source_row_index:,} | "
            f"Skipped: {skipped:,} | "
            f"Rate: {rate:,.1f} records/s"
        )

    # ----------------------------------------------------------------------
    # Flush final batch.
    # ----------------------------------------------------------------------

    if batch:

        qdrant.upsert_points(
            batch,
            wait=True,
        )

        processed += len(batch)

        batch.clear()

    elapsed = time.time() - start_time

    rate = (
        processed / elapsed
        if elapsed > 0
        else 0.0
    )

    qdrant_count = qdrant.count()

    # ----------------------------------------------------------------------
    # Report.
    # ----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("INGESTION COMPLETE")
    print("-" * 72)
    print(f"HF source rows     : {source_row_index:,}")
    print(f"HF records valid   : {processed:,}")
    print(f"HF records skipped : {skipped:,}")
    print(f"Qdrant point count : {qdrant_count:,}")
    print(f"Elapsed            : {elapsed:,.1f} sec")
    print(f"Rate               : {rate:,.1f} records/s")
    print("-" * 72)

    # ----------------------------------------------------------------------
    # Critical validation.
    # ----------------------------------------------------------------------

    if qdrant_count != processed:

        print()
        print(
            "ERROR: Qdrant point count does not match "
            "the number of valid ingested observations."
        )

        print(
            f"Expected: {processed:,}"
        )

        print(
            f"Actual:   {qdrant_count:,}"
        )

        sys.exit(1)

    print()
    print(
        f"VALIDATED: {qdrant_count:,} Qdrant points "
        f"== {processed:,} valid HF observations."
    )

    return processed, skipped


# ============================================================================
# Analysis
# ============================================================================

def analyze_embeddings(
    qdrant: QdrantResource,
) -> tuple[int, int, int]:
    """
    Second HF streaming pass.

    Calculates density and KNN against the complete Qdrant collection.

    source_row_index is reconstructed from the same deterministic stream
    order used during ingestion.
    """

    print()
    print("=" * 72)
    print("PHASE 4 — QDRANT ANALYSIS")
    print("=" * 72)
    print(f"Collection        : {QDRANT_COLLECTION}")
    print(f"Analysis batch    : {ANALYSIS_BATCH_SIZE}")
    print(f"K                 : {K}")
    print(f"Density threshold : {DENSITY_THRESHOLD}")
    print(f"Density limit     : {DENSITY_LIMIT:,}")
    print(f"Output            : {OUTPUT_DIR}")
    print("=" * 72)

    dataset = open_hf_stream()

    analyzer = Phase4EmbeddingAnalyzer(
        qdrant
    )

    batch: List[Dict[str, Any]] = []

    processed = 0
    skipped = 0
    knn_rows = 0

    source_row_index = 0

    start_time = time.time()

    # ----------------------------------------------------------------------
    # Open output CSVs.
    # ----------------------------------------------------------------------

    with (
        open(
            METRICS_OUTPUT,
            "w",
            newline="",
            encoding="utf-8",
        ) as metrics_file,

        open(
            KNN_OUTPUT,
            "w",
            newline="",
            encoding="utf-8",
        ) as knn_file,
    ):

        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "source_row_index",
                "record_id",
                "start",
                "end",
                "string_length",
                "embedding_norm",
                "local_density",
            ],
        )

        knn_writer = csv.DictWriter(
            knn_file,
            fieldnames=[
                "query_row_index",
                "query_record_id",
                "query_start",
                "query_end",
                "neighbor_rank",
                "neighbor_row_index",
                "neighbor_record_id",
                "neighbor_start",
                "neighbor_end",
                "cosine_similarity",
                "cosine_distance",
            ],
        )

        metrics_writer.writeheader()
        knn_writer.writeheader()

        # --------------------------------------------------------------
        # Second HF streaming pass.
        # --------------------------------------------------------------

        for row in dataset:

            source_row_index += 1

            record_id = row.get("record_id")
            start = row.get("start")
            end = row.get("end")

            string_length = row.get(
                "string_lengths"
            )

            embedding_norm = row.get(
                "embedding_norm"
            )

            vector = normalize_embedding(
                row.get("embedding")
            )

            # ----------------------------------------------------------
            # Validation.
            # ----------------------------------------------------------

            if record_id is None:

                skipped += 1
                continue

            record_id = str(record_id)

            if start is None or end is None:

                skipped += 1
                continue

            if not vector:

                skipped += 1
                continue

            if len(vector) != VECTOR_SIZE:

                skipped += 1
                continue

            # ----------------------------------------------------------
            # Add to analysis batch.
            # ----------------------------------------------------------

            batch.append(
                {
                    "source_row_index": source_row_index,
                    "record_id": record_id,
                    "start": start,
                    "end": end,
                    "string_length": string_length,
                    "embedding_norm": embedding_norm,
                    "vector": vector,
                }
            )

            if len(batch) < ANALYSIS_BATCH_SIZE:
                continue

            # ----------------------------------------------------------
            # Density.
            # ----------------------------------------------------------

            densities = (
                analyzer.compute_fixed_radius_density_batch(
                    batch
                )
            )

            # ----------------------------------------------------------
            # KNN.
            # ----------------------------------------------------------

            knn_results = (
                analyzer.extract_knn_batch(
                    batch
                )
            )

            # ----------------------------------------------------------
            # Write metrics.
            # ----------------------------------------------------------

            for item in batch:

                record_id = str(
                    item["record_id"]
                )

                point_id = qdrant_point_id(
                    record_id,
                    item["start"],
                    item["end"],
                    item["source_row_index"],
                )

                metrics_writer.writerow(
                    {
                        "source_row_index": item[
                            "source_row_index"
                        ],
                        "record_id": record_id,
                        "start": item["start"],
                        "end": item["end"],
                        "string_length": item[
                            "string_length"
                        ],
                        "embedding_norm": item[
                            "embedding_norm"
                        ],
                        "local_density": densities.get(
                            point_id,
                            0,
                        ),
                    }
                )

            # ----------------------------------------------------------
            # Write KNN.
            # ----------------------------------------------------------

            knn_writer.writerows(
                knn_results
            )

            metrics_file.flush()
            knn_file.flush()

            processed += len(batch)
            knn_rows += len(knn_results)

            elapsed = time.time() - start_time

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"Analyzed: {processed:,} | "
                f"Source rows: {source_row_index:,} | "
                f"Skipped: {skipped:,} | "
                f"KNN rows: {knn_rows:,} | "
                f"Rate: {rate:,.1f} records/s"
            )

            batch.clear()

        # --------------------------------------------------------------
        # Final partial batch.
        # --------------------------------------------------------------

        if batch:

            densities = (
                analyzer.compute_fixed_radius_density_batch(
                    batch
                )
            )

            knn_results = (
                analyzer.extract_knn_batch(
                    batch
                )
            )

            for item in batch:

                record_id = str(
                    item["record_id"]
                )

                point_id = qdrant_point_id(
                    record_id,
                    item["start"],
                    item["end"],
                    item["source_row_index"],
                )

                metrics_writer.writerow(
                    {
                        "source_row_index": item[
                            "source_row_index"
                        ],
                        "record_id": record_id,
                        "start": item["start"],
                        "end": item["end"],
                        "string_length": item[
                            "string_length"
                        ],
                        "embedding_norm": item[
                            "embedding_norm"
                        ],
                        "local_density": densities.get(
                            point_id,
                            0,
                        ),
                    }
                )

            knn_writer.writerows(
                knn_results
            )

            metrics_file.flush()
            knn_file.flush()

            processed += len(batch)
            knn_rows += len(knn_results)

            batch.clear()

    elapsed = time.time() - start_time

    rate = (
        processed / elapsed
        if elapsed > 0
        else 0.0
    )

    print()
    print("-" * 72)
    print("ANALYSIS COMPLETE")
    print("-" * 72)
    print(f"Source rows processed : {source_row_index:,}")
    print(f"Records analyzed      : {processed:,}")
    print(f"Records skipped       : {skipped:,}")
    print(f"KNN rows              : {knn_rows:,}")
    print(f"Elapsed               : {elapsed:,.1f} sec")
    print(f"Rate                  : {rate:,.1f} records/s")
    print()
    print(f"Metrics CSV           : {METRICS_OUTPUT}")
    print(f"KNN CSV               : {KNN_OUTPUT}")
    print("-" * 72)

    return processed, skipped, knn_rows


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = QdrantConfig(
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        vector_size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
        timeout=500.0,
        prefer_grpc=False,

        # Five total attempts.
        max_retries=5,

        # Exponential retry delays:
        # 2s, 4s, 8s, 16s
        retry_base_delay=2.0,
    )

    print()
    print("#" * 72)
    print("# PHASE 4 — COMPLETE PIPELINE")
    print("#" * 72)
    print("#")
    print(
        "# Observation identity:"
    )
    print(
        "# (record_id, start, end, source_row_index)"
    )
    print("#")
    print("# Duplicate source observations are preserved.")
    print("#")
    print("# 1. Recreate Qdrant collection")
    print("# 2. Stream HF dataset")
    print("# 3. Assign source_row_index to every row")
    print("# 4. Ingest all valid embeddings")
    print("# 5. Validate Qdrant population")
    print("# 6. Stream HF dataset again")
    print("# 7. Calculate fixed-radius density")
    print("# 8. Calculate KNN")
    print("# 9. Write CSV outputs")
    print("#")
    print("#" * 72)

    with QdrantResource(config) as qdrant:

        # ==============================================================
        # PASS 1 — INGESTION
        # ==============================================================

        ingest_embeddings(
            qdrant
        )

        # ==============================================================
        # PASS 2 — ANALYSIS
        # ==============================================================

        analyze_embeddings(
            qdrant
        )

    print()
    print("=" * 72)
    print("PHASE 4 PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Metrics : {METRICS_OUTPUT}")
    print(f"KNN     : {KNN_OUTPUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()