"""
Elasticsearch resource for local lexical retrieval.

Responsibilities
----------------
- Initialize an Elasticsearch client.
- Create/configure an index.
- Bulk ingest searchable documents.
- Retrieve documents.
- Execute lexical/filtered searches.
- Inspect index statistics.

This module intentionally contains no Phase 4 analytical logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from elasticsearch import Elasticsearch, helpers

from carbon_enrichment.resources.resource_logging import get_logger, timed_operation


logger = get_logger("elasticsearch")


@dataclass(frozen=True)
class ElasticsearchConfig:
    """Configuration for a local Elasticsearch resource."""

    url: str = "http://localhost:9200"

    api_key: str | None = None

    index_name: str = "carbon_records"

    request_timeout: float = 30.0

    bulk_chunk_size: int = 1_000

    max_retries: int = 3

    retry_on_timeout: bool = True


class ElasticsearchResource:
    """
    Thin resource wrapper around an Elasticsearch client.

    The resource manages index and retrieval primitives only.
    """

    def __init__(
        self,
        config: ElasticsearchConfig | None = None,
    ) -> None:
        self.config = config or ElasticsearchConfig()
        self._client: Elasticsearch | None = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def get_client(self) -> Elasticsearch:
        """
        Create and return the configured Elasticsearch client.

        The client is initialized lazily and reused for the lifetime
        of this resource instance.
        """
        if self._client is not None:
            return self._client

        with timed_operation(
            logger,
            "elasticsearch",
            "client_initialize",
        ):
            client_kwargs: dict[str, Any] = {
                "request_timeout": self.config.request_timeout,
                "retry_on_timeout": self.config.retry_on_timeout,
                "max_retries": self.config.max_retries,
            }

            if self.config.api_key is not None:
                client_kwargs["api_key"] = self.config.api_key

            self._client = Elasticsearch(
                self.config.url,
                **client_kwargs,
            )

        return self._client

    def close(self) -> None:
        """Close the Elasticsearch client."""
        if self._client is None:
            return

        with timed_operation(
            logger,
            "elasticsearch",
            "client_close",
        ):
            self._client.close()
            self._client = None

    def __enter__(self) -> ElasticsearchResource:
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
    # Connection / cluster
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check whether Elasticsearch is reachable."""
        client = self.get_client()

        with timed_operation(
            logger,
            "elasticsearch",
            "ping",
        ):
            return bool(client.ping())

    def info(self):
        """Return Elasticsearch cluster/server information."""
        client = self.get_client()

        with timed_operation(
            logger,
            "elasticsearch",
            "cluster_info",
        ):
            return client.info()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def index_exists(
        self,
        index_name: str | None = None,
    ) -> bool:
        """Return whether an Elasticsearch index exists."""
        client = self.get_client()

        name = index_name or self.config.index_name

        return bool(
            client.indices.exists(
                index=name,
            )
        )

    def ensure_index(
        self,
        *,
        mappings: Mapping[str, Any],
        settings: Mapping[str, Any] | None = None,
        index_name: str | None = None,
    ) -> None:
        """
        Create the configured index if it does not already exist.

        Existing indices are left unchanged.
        """
        client = self.get_client()

        name = index_name or self.config.index_name

        if self.index_exists(name):
            logger.info(
                "index_exists | index=%s",
                name,
            )
            return

        body: dict[str, Any] = {
            "mappings": dict(mappings),
        }

        if settings is not None:
            body["settings"] = dict(settings)

        with timed_operation(
            logger,
            "elasticsearch",
            "index_create",
            index=name,
        ):
            client.indices.create(
                index=name,
                **body,
            )

        logger.info(
            "index_created | index=%s",
            name,
        )

    def delete_index(
        self,
        *,
        index_name: str | None = None,
        ignore_missing: bool = True,
    ) -> None:
        """Delete an index."""
        client = self.get_client()

        name = index_name or self.config.index_name

        if ignore_missing and not self.index_exists(name):
            logger.info(
                "index_delete_skipped | index=%s | reason=missing",
                name,
            )
            return

        with timed_operation(
            logger,
            "elasticsearch",
            "index_delete",
            index=name,
        ):
            client.indices.delete(
                index=name,
            )

    def get_mapping(
        self,
        *,
        index_name: str | None = None,
    ):
        """Return the mapping for an index."""
        client = self.get_client()

        name = index_name or self.config.index_name

        with timed_operation(
            logger,
            "elasticsearch",
            "get_mapping",
            index=name,
        ):
            return client.indices.get_mapping(
                index=name,
            )

    # ------------------------------------------------------------------
    # Bulk ingestion
    # ------------------------------------------------------------------

    def bulk_index(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        index_name: str | None = None,
        chunk_size: int | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """
        Bulk-index documents using the official Elasticsearch helpers.

        ``documents`` may be a generator, allowing large local datasets
        to be streamed without materializing the complete corpus.
        """
        client = self.get_client()

        name = index_name or self.config.index_name

        effective_chunk_size = (
            chunk_size
            if chunk_size is not None
            else self.config.bulk_chunk_size
        )

        effective_retries = (
            max_retries
            if max_retries is not None
            else self.config.max_retries
        )

        actions = self._prepare_bulk_actions(
            documents,
            index_name=name,
        )

        with timed_operation(
            logger,
            "elasticsearch",
            "bulk_index",
            unit="documents",
            index=name,
            chunk_size=effective_chunk_size,
            max_retries=effective_retries,
        ):
            success_count, errors = helpers.bulk(
                client,
                actions,
                chunk_size=effective_chunk_size,
                max_retries=effective_retries,
                raise_on_error=False,
            )

        error_count = len(errors)

        logger.info(
            "bulk_index_complete | index=%s | "
            "documents=%s | errors=%s",
            name,
            success_count,
            error_count,
        )

        return {
            "success_count": success_count,
            "error_count": error_count,
            "errors": errors,
        }

    def _prepare_bulk_actions(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        index_name: str,
    ) -> Iterable[dict[str, Any]]:
        """
        Normalize user documents into Elasticsearch bulk actions.

        The iterable remains lazy.
        """
        for document in documents:
            action = dict(document)

            action.setdefault(
                "_index",
                index_name,
            )

            yield action

    # ------------------------------------------------------------------
    # Single-document operations
    # ------------------------------------------------------------------

    def get_document(
        self,
        document_id: str | int,
        *,
        index_name: str | None = None,
    ):
        """Retrieve a document by ID."""
        client = self.get_client()

        name = index_name or self.config.index_name

        with timed_operation(
            logger,
            "elasticsearch",
            "get_document",
            index=name,
        ):
            return client.get(
                index=name,
                id=document_id,
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: Mapping[str, Any],
        *,
        size: int = 10,
        index_name: str | None = None,
        source: bool | list[str] | tuple[str, ...] | None = True,
        post_filter: Mapping[str, Any] | None = None,
        aggregations: Mapping[str, Any] | None = None,
    ):
        """
        Execute an Elasticsearch search.

        ``query`` should contain the Elasticsearch query DSL.
        The resource does not interpret the query semantically.
        """
        client = self.get_client()

        name = index_name or self.config.index_name

        kwargs: dict[str, Any] = {
            "index": name,
            "query": dict(query),
            "size": size,
        }

        if source is not None:
            kwargs["_source"] = source

        if post_filter is not None:
            kwargs["post_filter"] = dict(post_filter)

        if aggregations is not None:
            kwargs["aggs"] = dict(aggregations)

        with timed_operation(
            logger,
            "elasticsearch",
            "search",
            count=1,
            unit="query",
            index=name,
            top_k=size,
        ):
            response = client.search(**kwargs)

        return response

    # ------------------------------------------------------------------
    # Index statistics
    # ------------------------------------------------------------------

    def count_documents(
        self,
        *,
        index_name: str | None = None,
    ) -> int:
        """Return the document count for an index."""
        client = self.get_client()

        name = index_name or self.config.index_name

        with timed_operation(
            logger,
            "elasticsearch",
            "count_documents",
            index=name,
        ):
            response = client.count(
                index=name,
            )

        return int(response["count"])

    def get_index_stats(
        self,
        *,
        index_name: str | None = None,
    ):
        """
        Return Elasticsearch index statistics.

        This includes document and index-level accounting exposed
        by Elasticsearch.
        """
        client = self.get_client()

        name = index_name or self.config.index_name

        with timed_operation(
            logger,
            "elasticsearch",
            "index_stats",
            index=name,
        ):
            return client.indices.stats(
                index=name,
            )

    def refresh(
        self,
        *,
        index_name: str | None = None,
    ) -> None:
        """
        Refresh the index so recently indexed documents become
        available for search.
        """
        client = self.get_client()

        name = index_name or self.config.index_name

        with timed_operation(
            logger,
            "elasticsearch",
            "index_refresh",
            index=name,
        ):
            client.indices.refresh(
                index=name,
            )