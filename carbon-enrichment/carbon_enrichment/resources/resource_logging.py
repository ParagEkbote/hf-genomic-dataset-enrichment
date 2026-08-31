"""
Shared structured logging and timing utilities for the resource layer.

This module provides:
- consistent resource loggers
- timed operation context managers
- data-throughput measurements
- query-latency measurements

Resource-specific behavior belongs in the individual resource modules.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


@dataclass
class OperationTiming:
    """Timing information produced by a resource operation."""

    resource: str
    operation: str
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def throughput(self) -> float | None:
        """
        Calculate throughput when an appropriate count is available.

        Supported count fields:
        rows, vectors, documents, tokens, items
        """
        count = None

        for field_name in (
            "rows",
            "vectors",
            "documents",
            "tokens",
            "items",
        ):
            value = self.metadata.get(field_name)
            if isinstance(value, (int, float)) and value >= 0:
                count = value
                break

        if count is None or self.elapsed_seconds <= 0:
            return None

        return count / self.elapsed_seconds


def get_logger(
    resource: str,
    *,
    level: int = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """
    Return a consistently configured logger for a resource.

    Parameters
    ----------
    resource:
        Logical resource name, e.g. ``duckdb`` or ``qdrant``.
    level:
        Python logging level.
    """
    logger = logging.getLogger(f"carbon.resources.{resource}")

    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(DEFAULT_LOG_FORMAT)
        )
        logger.addHandler(handler)

    logger.propagate = False

    return logger


def _format_metadata(metadata: dict[str, Any]) -> str:
    """Format operation metadata for stable human-readable logging."""
    if not metadata:
        return ""

    return " | ".join(
        f"{key}={value}"
        for key, value in metadata.items()
    )


@contextmanager
def timed_operation(
    logger: logging.Logger,
    resource: str,
    operation: str,
    **metadata: Any,
) -> Generator[OperationTiming]:
    """
    Time a resource operation and emit structured start/completion logs.

    Example
    -------
    with timed_operation(
        logger,
        "duckdb",
        "taxonomy_statistics",
        rows=32_410_000,
    ) as timing:
        result = db.execute(query)

    The completion log includes elapsed time and throughput.
    """
    logger.info(
        "%s | %s",
        operation,
        _format_metadata(metadata),
    )

    start = time.perf_counter()

    timing = OperationTiming(
        resource=resource,
        operation=operation,
        elapsed_seconds=0.0,
        metadata=dict(metadata),
    )

    try:
        yield timing
    except Exception:
        timing.elapsed_seconds = time.perf_counter() - start

        logger.exception(
            "%s failed | elapsed=%.3fs | %s",
            operation,
            timing.elapsed_seconds,
            _format_metadata(metadata),
        )

        raise

    timing.elapsed_seconds = time.perf_counter() - start

    throughput = timing.throughput

    completion_metadata = dict(metadata)
    completion_metadata["elapsed"] = (
        f"{timing.elapsed_seconds:.3f}s"
    )

    if throughput is not None:
        completion_metadata["throughput"] = (
            f"{throughput:,.2f}/s"
        )

    logger.info(
        "%s complete | %s",
        operation,
        _format_metadata(completion_metadata),
    )


def log_query_result(
    logger: logging.Logger,
    *,
    resource: str,
    operation: str,
    elapsed_seconds: float,
    rows_returned: int | None = None,
    **metadata: Any,
) -> None:
    """
    Log a completed query with latency and optional result size.

    This is intended for query operations where the timing context
    and result cardinality are naturally known after execution.
    """
    fields: dict[str, Any] = {
        **metadata,
        "elapsed": f"{elapsed_seconds:.3f}s",
    }

    if rows_returned is not None:
        fields["rows_returned"] = rows_returned

    logger.info(
        "%s | %s | %s",
        resource,
        operation,
        _format_metadata(fields),
    )