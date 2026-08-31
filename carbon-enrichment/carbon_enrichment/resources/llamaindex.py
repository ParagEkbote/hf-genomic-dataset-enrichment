"""
LlamaIndex retrieval resource.

This module provides the LlamaIndex integration layer over the
project's existing retrieval resources.

Responsibilities
----------------
- Connect LlamaIndex to an existing Qdrant collection.
- Construct a LlamaIndex vector-store/index interface.
- Normalize retrieved nodes into RetrievalResult objects.
- Provide a stable retrieval contract for downstream evaluation.

This module does not:
- generate embeddings,
- ingest the embedding corpus,
- perform biological analysis,
- own Qdrant storage,
- define Phase 4 case-study logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from carbon_enrichment.resources.resource_logging import get_logger, timed_operation
from carbon_enrichment.resources.qdrant import QdrantResource


logger = get_logger("llamaindex")


@dataclass(frozen=True)
class LlamaIndexConfig:
    """Configuration for the LlamaIndex resource."""

    top_k: int = 10


@dataclass(frozen=True)
class RetrievalResult:
    """
    Normalized retrieval result.

    This is the resource-layer contract shared by vector,
    lexical, and eventually hybrid retrieval.
    """

    record_id: str
    rank: int
    score: float | None
    metadata: dict[str, Any]
    retrieval_method: str


class LlamaIndexResource:
    """
    Thin LlamaIndex orchestration layer.

    Qdrant remains the underlying vector resource.
    """

    def __init__(
        self,
        qdrant: QdrantResource,
        config: LlamaIndexConfig | None = None,
    ) -> None:
        self.qdrant = qdrant
        self.config = config or LlamaIndexConfig()

        self._vector_store: QdrantVectorStore | None = None
        self._storage_context: StorageContext | None = None
        self._index: VectorStoreIndex | None = None
        self._retriever: Any | None = None

    # ------------------------------------------------------------------
    # Qdrant / LlamaIndex integration
    # ------------------------------------------------------------------

    def get_vector_store(
        self,
        *,
        collection_name: str | None = None,
    ) -> QdrantVectorStore:
        """
        Construct the LlamaIndex QdrantVectorStore over the existing
        Qdrant client/collection.
        """
        if self._vector_store is not None:
            return self._vector_store

        client = self.qdrant.get_client()

        name = (
            collection_name
            or self.qdrant.config.collection_name
        )

        with timed_operation(
            logger,
            "llamaindex",
            "initialize_qdrant_vector_store",
            collection=name,
        ):
            self._vector_store = QdrantVectorStore(
                client=client,
                collection_name=name,
            )

        return self._vector_store

    def get_storage_context(self) -> StorageContext:
        """
        Construct a LlamaIndex StorageContext around the existing
        Qdrant vector store.
        """
        if self._storage_context is not None:
            return self._storage_context

        vector_store = self.get_vector_store()

        with timed_operation(
            logger,
            "llamaindex",
            "initialize_storage_context",
        ):
            self._storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )

        return self._storage_context

    def get_index(self) -> VectorStoreIndex:
        """
        Construct a LlamaIndex VectorStoreIndex over the existing
        Qdrant vector store.

        No new embedding generation is performed here.
        """
        if self._index is not None:
            return self._index

        storage_context = self.get_storage_context()

        with timed_operation(
            logger,
            "llamaindex",
            "initialize_vector_index",
        ):
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=storage_context.vector_store,
            )

        return self._index

    def get_retriever(
        self,
        *,
        top_k: int | None = None,
    ) -> Any:
        """
        Construct the LlamaIndex vector retriever.
        """
        if self._retriever is not None:
            return self._retriever

        index = self.get_index()

        effective_top_k = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        with timed_operation(
            logger,
            "llamaindex",
            "initialize_vector_retriever",
            top_k=effective_top_k,
        ):
            self._retriever = index.as_retriever(
                similarity_top_k=effective_top_k,
            )

        return self._retriever

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Execute vector retrieval through LlamaIndex and normalize
        the returned nodes.
        """
        retriever = self.get_retriever(top_k=top_k)

        effective_top_k = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        with timed_operation(
            logger,
            "llamaindex",
            "vector_retrieval",
            count=1,
            unit="query",
            top_k=effective_top_k,
        ) as timing:
            nodes = retriever.retrieve(query)

        results = self.normalize_results(
            nodes,
            retrieval_method="qdrant",
        )

        timing.metadata["source_nodes"] = len(results)

        return results

    # ------------------------------------------------------------------
    # Result normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_results(
        nodes: Sequence[NodeWithScore],
        *,
        retrieval_method: str,
    ) -> list[RetrievalResult]:
        """
        Normalize LlamaIndex NodeWithScore objects.

        The normalization contract deliberately preserves:
        - record ID
        - rank
        - score
        - metadata
        - retrieval method
        """
        results: list[RetrievalResult] = []

        for rank, node_with_score in enumerate(
            nodes,
            start=1,
        ):
            node = node_with_score.node

            metadata = dict(
                getattr(node, "metadata", {}) or {}
            )

            record_id = (
                metadata.get("record_id")
                or metadata.get("id")
                or node.node_id
            )

            results.append(
                RetrievalResult(
                    record_id=str(record_id),
                    rank=rank,
                    score=(
                        float(node_with_score.score)
                        if node_with_score.score is not None
                        else None
                    ),
                    metadata=metadata,
                    retrieval_method=retrieval_method,
                )
            )

        return results