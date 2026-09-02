from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for a local Qdrant resource."""

    url: str = "http://localhost:6333"
    api_key: str | None = None

    collection_name: str = "carbon_embeddings"

    vector_size: int | None = None
    distance: models.Distance = models.Distance.COSINE

    # Qdrant client HTTP timeout, in seconds.
    timeout: float | None = 300.0

    prefer_grpc: bool = False


class QdrantResource:
    """
    Thin resource wrapper around a Qdrant client.

    The resource provides storage and retrieval primitives only.
    Biological interpretation and derived metrics belong in the
    derived analysis layer.
    """

    def __init__(
        self,
        config: QdrantConfig | None = None,
    ) -> None:
        self.config = config or QdrantConfig()
        self._client: QdrantClient | None = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def get_client(self) -> QdrantClient:
        """
        Create and return the configured Qdrant client.

        The client is initialized lazily and reused for the lifetime
        of this resource instance.
        """
        if self._client is None:
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )

        return self._client

    def close(self) -> None:
        """Close the Qdrant client."""
        if self._client is None:
            return

        self._client.close()
        self._client = None

    def __enter__(self) -> QdrantResource:
        self.get_client()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: Any,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(
        self,
        collection_name: str | None = None,
    ) -> bool:
        """Return whether a collection exists."""

        client = self.get_client()

        name = collection_name or self.config.collection_name

        return client.collection_exists(
            collection_name=name,
        )

    def ensure_collection(
        self,
        *,
        vector_size: int | None = None,
        distance: models.Distance | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Create the collection if it does not already exist.

        Existing collections are left unchanged.
        """

        client = self.get_client()

        name = collection_name or self.config.collection_name

        size = vector_size or self.config.vector_size

        if size is None:
            raise ValueError(
                "vector_size must be supplied either through "
                "QdrantConfig or ensure_collection()."
            )

        metric = distance or self.config.distance

        if self.collection_exists(name):
            return

        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=size,
                distance=metric,
            ),
        )

    def get_collection_info(
        self,
        collection_name: str | None = None,
    ):
        """
        Return Qdrant collection information.
        """

        client = self.get_client()

        name = collection_name or self.config.collection_name

        return client.get_collection(
            collection_name=name,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert_points(
        self,
        points: Sequence[models.PointStruct],
        *,
        collection_name: str | None = None,
        wait: bool = True,
    ):
        """
        Upsert an explicit batch of points into the collection.
        """

        client = self.get_client()

        name = collection_name or self.config.collection_name

        return client.upsert(
            collection_name=name,
            points=list(points),
            wait=wait,
        )

    def upload_points(
        self,
        points: Iterable[models.PointStruct],
        *,
        collection_name: str | None = None,
        batch_size: int = 1_000,
        parallel: int = 1,
        max_retries: int = 3,
        wait: bool = True,
    ) -> None:
        """
        Upload an iterable of points using Qdrant's bulk upload
        facilities.

        The iterable may be generated lazily, so the complete
        embedding corpus does not need to reside in memory.
        """

        client = self.get_client()

        name = collection_name or self.config.collection_name

        client.upload_points(
            collection_name=name,
            points=points,
            batch_size=batch_size,
            parallel=parallel,
            max_retries=max_retries,
            wait=wait,
        )

    # ------------------------------------------------------------------
    # Collection inspection
    # ------------------------------------------------------------------

    def count_points(
        self,
        *,
        collection_name: str | None = None,
        exact: bool = True,
    ) -> int:
        """
        Return the number of points in a collection.
        """

        client = self.get_client()

        name = collection_name or self.config.collection_name

        result = client.count(
            collection_name=name,
            exact=exact,
        )

        return result.count

    # ------------------------------------------------------------------
    # Retrieval
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
        with_vectors: bool = False,
    ):
        """
        Search for nearest vectors.

        Returns raw Qdrant query results. Biological interpretation
        belongs in the derived analysis layer.
        """

        client = self.get_client()

        name = collection_name or self.config.collection_name

        result = client.query_points(
            collection_name=name,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        return result.points

    def retrieve(
        self,
        point_ids: Sequence[int | str],
        *,
        collection_name: str | None = None,
        with_payload: bool | Sequence[str] = True,
        with_vectors: bool = False,
    ):
        """
        Retrieve specific points by ID.
        """

        client = self.get_client()

        name = collection_name or self.config.collection_name

        return client.retrieve(
            collection_name=name,
            ids=list(point_ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )