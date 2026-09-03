from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for a local Qdrant resource."""

    url: str = "http://localhost:6333"
    api_key: str | None = None

    collection_name: str = "carbon_embeddings"

    vector_size: int | None = None
    distance: models.Distance = models.Distance.COSINE

    timeout: float | None = 500.0
    prefer_grpc: bool = False

    # ------------------------------------------------------------------
    # Retry configuration
    # ------------------------------------------------------------------

    max_retries: int = 50
    retry_base_delay: float = 2.0


class QdrantResource:
    """
    Thin resource wrapper around a Qdrant client.

    Provides collection management, storage, retrieval,
    grouping, and faceting primitives.

    Transient Qdrant/network failures are retried automatically.

    Biological interpretation belongs in the derived
    analysis layer.
    """

    def __init__(self, config: QdrantConfig | None = None) -> None:
        self.config = config or QdrantConfig()
        self._client: QdrantClient | None = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )

        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> QdrantResource:
        self.get_client()
        return self

    def __exit__(
        self,
        _exc_type,
        _exc_value,
        _traceback,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _with_retry(self, operation, operation_name: str):
        """
        Execute a Qdrant operation with exponential-backoff retries.

        max_retries = 5 means up to 5 total attempts.

        Delays between attempts:
            2s
            4s
            8s
            16s
        """

        max_attempts = max(1, self.config.max_retries)
        base_delay = max(0.0, self.config.retry_base_delay)

        for attempt in range(1, max_attempts + 1):
            try:
                return operation()

            except Exception as exc:
                if attempt >= max_attempts:
                    print(
                        f"\nERROR: {operation_name} failed after "
                        f"{max_attempts} attempts."
                    )
                    print(f"Last error: {exc}")
                    raise

                delay = base_delay * (2 ** (attempt - 1))

                print(
                    f"\nWARNING: {operation_name} failed "
                    f"(attempt {attempt}/{max_attempts})."
                )
                print(f"Error: {exc}")
                print(
                    f"Retrying in {delay:.1f} seconds..."
                )

                time.sleep(delay)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(
        self,
        collection_name: str | None = None,
    ) -> bool:
        """Return whether a collection exists."""

        name = collection_name or self.config.collection_name

        return self._with_retry(
            lambda: self.get_client().collection_exists(
                collection_name=name,
            ),
            f"collection_exists('{name}')",
        )

    def ensure_collection(
        self,
        *,
        vector_size: int | None = None,
        distance: models.Distance | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Create the collection if it does not already exist."""

        name = collection_name or self.config.collection_name
        size = vector_size or self.config.vector_size
        metric = distance or self.config.distance

        if size is None:
            raise ValueError(
                "vector_size must be supplied either through "
                "QdrantConfig or ensure_collection()."
            )

        if self.collection_exists(name):
            return

        self._with_retry(
            lambda: self.get_client().create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=size,
                    distance=metric,
                ),
            ),
            f"create_collection('{name}')",
        )

    def recreate_collection(
        self,
        *,
        vector_size: int | None = None,
        distance: models.Distance | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Delete and recreate a collection.

        Intended for reproducible full-dataset ingestion runs.
        """

        name = collection_name or self.config.collection_name
        size = vector_size or self.config.vector_size
        metric = distance or self.config.distance

        if size is None:
            raise ValueError(
                "vector_size must be supplied either through "
                "QdrantConfig or recreate_collection()."
            )

        client = self.get_client()

        exists = self._with_retry(
            lambda: client.collection_exists(
                collection_name=name,
            ),
            f"collection_exists('{name}')",
        )

        if exists:
            self._with_retry(
                lambda: client.delete_collection(
                    collection_name=name,
                ),
                f"delete_collection('{name}')",
            )

        self._with_retry(
            lambda: client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=size,
                    distance=metric,
                ),
            ),
            f"create_collection('{name}')",
        )

    def count(
        self,
        *,
        collection_name: str | None = None,
    ) -> int:
        """Return the number of points in a collection."""

        name = collection_name or self.config.collection_name

        result = self._with_retry(
            lambda: self.get_client().count(
                collection_name=name,
                exact=True,
            ),
            f"count('{name}')",
        )

        return int(result.count)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def upsert_points(
        self,
        points: Sequence[models.PointStruct],
        *,
        collection_name: str | None = None,
        wait: bool = True,
    ) -> None:
        """
        Insert or update points in Qdrant.

        The complete batch is retried if the Qdrant request fails.
        """

        name = collection_name or self.config.collection_name
        point_list = list(points)

        self._with_retry(
            lambda: self.get_client().upsert(
                collection_name=name,
                points=point_list,
                wait=wait,
            ),
            f"upsert({len(point_list)} points)",
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
        with_payload: bool | Sequence[str] = True,
    ):
        """Search for nearest vectors."""

        name = collection_name or self.config.collection_name

        result = self._with_retry(
            lambda: self.get_client().query_points(
                collection_name=name,
                query=list(vector),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
            ),
            f"query_points('{name}')",
        )

        return result.points

    def search_batch(
        self,
        requests: Sequence[models.QueryRequest],
        *,
        collection_name: str | None = None,
    ):
        """
        Execute multiple search queries in a single network call.

        The complete batch request is retried on failure.
        """

        name = collection_name or self.config.collection_name
        request_list = list(requests)

        return self._with_retry(
            lambda: self.get_client().query_batch_points(
                collection_name=name,
                requests=request_list,
            ),
            f"query_batch_points({len(request_list)} queries)",
        )

    def search_groups(
        self,
        vector: Sequence[float],
        group_by: str,
        *,
        limit: int = 10,
        group_size: int = 1,
        collection_name: str | None = None,
        with_payload: bool | Sequence[str] = True,
    ):
        """Search for nearest vectors grouped by a payload key."""

        name = collection_name or self.config.collection_name

        result = self._with_retry(
            lambda: self.get_client().query_points_groups(
                collection_name=name,
                query=list(vector),
                group_by=group_by,
                limit=limit,
                group_size=group_size,
                with_payload=with_payload,
            ),
            f"query_points_groups('{name}')",
        )

        return result.groups

    # ------------------------------------------------------------------
    # Faceting
    # ------------------------------------------------------------------

    def facet(
        self,
        key: str,
        *,
        query_filter: models.Filter | None = None,
        limit: int = 100,
        collection_name: str | None = None,
    ):
        """Count value frequencies for a categorical payload key."""

        name = collection_name or self.config.collection_name

        return self._with_retry(
            lambda: self.get_client().facet(
                collection_name=name,
                key=key,
                query_filter=query_filter,
                limit=limit,
            ),
            f"facet('{name}', key='{key}')",
        )