from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    Return a consistently configured application logger.

    Logger configuration is centralized here so individual resources
    do not create their own handlers, formatters, or logging policies.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(_LOG_FORMAT)
        )
        logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger


# ----------------------------------------------------------------------
# Short-operation timing
# ----------------------------------------------------------------------

@dataclass
class OperationTiming:
    """
    Runtime metadata for a single timed operation.

    Additional metadata can be attached while the operation is running
    and will be included in the completion event.
    """

    resource: str
    operation: str
    start_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time


@contextmanager
def timed_operation(
    logger: logging.Logger,
    resource: str,
    operation: str,
    **metadata: Any,
) -> Iterator[OperationTiming]:
    """
    Time a short-lived resource operation.

    This is intended for operations such as:

        DuckDB connection initialization
        DuckDB view registration
        DuckDB analytical query
        Elasticsearch index creation
        Qdrant collection creation

    Long-running row/batch workloads should use ProgressTracker instead.
    """
    timing = OperationTiming(
        resource=resource,
        operation=operation,
        start_time=time.perf_counter(),
        metadata=dict(metadata),
    )

    logger.info(
        "%s | %s | START | %s",
        resource,
        operation,
        _format_metadata(timing.metadata),
    )

    try:
        yield timing

    except Exception as exc:
        elapsed = timing.elapsed_seconds

        logger.exception(
            "%s | %s | FAIL | elapsed=%.3fs | error=%s | %s",
            resource,
            operation,
            elapsed,
            type(exc).__name__,
            _format_metadata(timing.metadata),
        )

        raise

    else:
        elapsed = timing.elapsed_seconds

        logger.info(
            "%s | %s | COMPLETE | elapsed=%.3fs | %s",
            resource,
            operation,
            elapsed,
            _format_metadata(timing.metadata),
        )


# ----------------------------------------------------------------------
# Long-running progress tracking
# ----------------------------------------------------------------------

@dataclass
class ProgressTracker:
    """
    Reusable progress tracker for long-running row/batch operations.

    Responsibilities:
      - progress logging
      - elapsed time
      - percentage
      - throughput
      - ETA
      - completion/failure events

    It deliberately does NOT wrap the operation in timed_operation().
    This class already owns the timing lifecycle for the long-running
    workload.
    """

    logger: logging.Logger
    resource: str
    operation: str
    total: int
    unit: str = "rows"
    log_every: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    _started: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _completed: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _current: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _start_time: float | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_logged_batch: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def start(self) -> None:
        """Start the progress timer and emit the initial event."""
        if self._started:
            raise RuntimeError(
                "ProgressTracker has already been started."
            )

        if self.total < 0:
            raise ValueError("total must be >= 0")

        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")

        self._started = True
        self._start_time = time.perf_counter()

        self.logger.info(
            "%s | %s | START | total=%d %s | %s",
            self.resource,
            self.operation,
            self.total,
            self.unit,
            _format_metadata(self.metadata),
        )

    def update(
        self,
        amount: int,
        *,
        batch: int | None = None,
        force: bool = False,
        **metadata: Any,
    ) -> None:
        """
        Advance progress and optionally emit a progress event.

        `amount` is the number of newly processed units, not the
        cumulative count.
        """
        if not self._started:
            raise RuntimeError(
                "ProgressTracker.start() must be called before update()."
            )

        if self._completed:
            raise RuntimeError(
                "Cannot update a completed ProgressTracker."
            )

        if amount < 0:
            raise ValueError("amount must be >= 0")

        self._current += amount

        if self.total > 0:
            self._current = min(
                self._current,
                self.total,
            )

        should_log = force

        if batch is not None:
            if batch - self._last_logged_batch >= self.log_every:
                should_log = True
        elif self._current >= self.total:
            should_log = True

        if not should_log:
            return

        if batch is not None:
            self._last_logged_batch = batch

        self._log_progress(
            batch=batch,
            metadata=metadata,
        )

    def complete(self, **metadata: Any) -> None:
        """Mark the operation complete and emit the final summary."""
        if not self._started:
            raise RuntimeError(
                "ProgressTracker.start() must be called before complete()."
            )

        if self._completed:
            return

        self._current = self.total
        self._completed = True

        elapsed = self.elapsed_seconds
        throughput = self._throughput(elapsed)

        combined_metadata = {
            **self.metadata,
            **metadata,
        }

        self.logger.info(
            "%s | %s | COMPLETE | "
            "processed=%d/%d %s | "
            "elapsed=%.3fs | "
            "throughput=%.2f %s/s | %s",
            self.resource,
            self.operation,
            self._current,
            self.total,
            self.unit,
            elapsed,
            throughput,
            self.unit,
            _format_metadata(combined_metadata),
        )

    def fail(self, **metadata: Any) -> None:
        """Emit a failure summary for the current progress state."""
        if not self._started:
            return

        elapsed = self.elapsed_seconds
        throughput = self._throughput(elapsed)

        combined_metadata = {
            **self.metadata,
            **metadata,
        }

        self.logger.error(
            "%s | %s | FAIL | "
            "processed=%d/%d %s | "
            "elapsed=%.3fs | "
            "throughput=%.2f %s/s | %s",
            self.resource,
            self.operation,
            self._current,
            self.total,
            self.unit,
            elapsed,
            throughput,
            self.unit,
            _format_metadata(combined_metadata),
        )

    @property
    def current(self) -> int:
        """Return the number of processed units."""
        return self._current

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed wall-clock seconds."""
        if self._start_time is None:
            return 0.0

        return time.perf_counter() - self._start_time

    @property
    def percentage(self) -> float:
        """Return completion percentage."""
        if self.total <= 0:
            return 100.0

        return min(
            100.0,
            self._current / self.total * 100.0,
        )

    @property
    def throughput(self) -> float:
        """Return current units-per-second throughput."""
        return self._throughput(self.elapsed_seconds)

    @property
    def eta_seconds(self) -> float | None:
        """Return estimated seconds remaining."""
        if self._current <= 0:
            return None

        if self._current >= self.total:
            return 0.0

        rate = self.throughput

        if rate <= 0:
            return None

        remaining = self.total - self._current

        return remaining / rate

    def _log_progress(
        self,
        *,
        batch: int | None,
        metadata: dict[str, Any],
    ) -> None:
        elapsed = self.elapsed_seconds
        throughput = self._throughput(elapsed)
        eta = self.eta_seconds

        eta_text = (
            f"{eta:.1f}s"
            if eta is not None
            else "unknown"
        )

        combined_metadata = {
            **self.metadata,
            **metadata,
        }

        batch_text = (
            f" | batch={batch}"
            if batch is not None
            else ""
        )

        self.logger.info(
            "%s | %s | PROGRESS | "
            "processed=%d/%d %s | "
            "percent=%.2f%% | "
            "elapsed=%.3fs | "
            "throughput=%.2f %s/s | "
            "eta=%s%s | %s",
            self.resource,
            self.operation,
            self._current,
            self.total,
            self.unit,
            self.percentage,
            elapsed,
            throughput,
            self.unit,
            eta_text,
            batch_text,
            _format_metadata(combined_metadata),
        )

    @staticmethod
    def _throughput(elapsed: float, current: int | None = None) -> float:
        if elapsed <= 0:
            return 0.0

        if current is None:
            return 0.0

        return current / elapsed

    def _throughput(self, elapsed: float) -> float:
        if elapsed <= 0:
            return 0.0

        return self._current / elapsed


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------

def _format_metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""

    return " | ".join(
        f"{key}={value}"
        for key, value in metadata.items()
    )