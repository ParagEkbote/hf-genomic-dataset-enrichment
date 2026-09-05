from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pyarrow as pa


# ----------------------------------------------------------------------
# Local execution defaults
# ----------------------------------------------------------------------

DEFAULT_THREADS = max(
    1,
    (os.cpu_count() or 1) - 1,
)

DEFAULT_BINARY = "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment/carbon-enrichment/clickhouse"

DEFAULT_TEMP_DIRECTORY = Path(
    "data/tmp/clickhouse"
)

@dataclass(frozen=True)
class ClickHouseConfig:
    """Configuration for a local `clickhouse local` resource."""
    binary_path: str = DEFAULT_BINARY
    threads: int | None = DEFAULT_THREADS
    temp_directory: str | Path | None = DEFAULT_TEMP_DIRECTORY
    join_algorithm: str = "grace_hash"
    max_http_get_redirects: int = 10
    hf_token: str | None = None


class ClickHouseResource:
    """
    Thin resource wrapper around `clickhouse local`.

    Responsibilities:
      - binary discovery / execution configuration
      - registration of local Parquet datasets
      - registration of Hugging Face Parquet datasets without downloading them
      - HF shard discovery via the Hugging Face dataset API
      - analytical query execution, returning Arrow tables/batches (either
        fully materialized or streamed as record batches)

    Analytical logic belongs in consuming modules such as `distributions.py`
    or `case_study.py`.
    """

    def __init__(
        self,
        config: ClickHouseConfig | None = None,
    ) -> None:
        self.config = config or ClickHouseConfig()

        self._binary: str | None = None

        # name -> source expression
        self._registered_datasets: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Binary lifecycle
    # ------------------------------------------------------------------

    def get_binary(self) -> str:
        """Resolve and cache the clickhouse binary path, initializing once."""
        if self._binary is not None:
            return self._binary

        resolved = shutil.which(self.config.binary_path)

        if resolved is None:
            raise RuntimeError(
                "clickhouse binary not found on PATH. Install with:\n"
                "  curl https://clickhouse.com/ | sh\n"
                "then re-run (this resource uses `clickhouse local`)."
            )

        self._binary = resolved

        if self.config.temp_directory is not None:
            Path(
                self.config.temp_directory
            ).expanduser().mkdir(
                parents=True,
                exist_ok=True,
            )

        return self._binary

    def close(self) -> None:
        """No persistent connection is held; kept for interface parity."""
        self._binary = None

    def __enter__(self) -> ClickHouseResource:
        self.get_binary()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        path: str | Path,
    ) -> None:
        """
        Register a local Parquet dataset.

        `path` may be a single Parquet file or a directory containing
        Parquet files.
        """
        self._validate_relation_name(name)

        dataset_path = Path(path).expanduser()

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Parquet dataset does not exist: {dataset_path}"
            )

        glob_expr = (
            str(dataset_path / "*.parquet")
            if dataset_path.is_dir()
            else str(dataset_path)
        )

        self._registered_datasets[name] = (
            f"file('{self._escape_sql_string(glob_expr)}', Parquet)"
        )

    def register_enriched_glob(
        self,
        name: str,
        directory: str | Path,
        pattern: str = "batch_*.parquet",
    ) -> None:
        """
        Register a directory of enriched batch Parquet files as one source.
        """
        self._validate_relation_name(name)

        directory = Path(directory).expanduser()

        if not directory.exists():
            raise FileNotFoundError(
                f"Enriched output directory does not exist: {directory}"
            )

        glob_expr = str(directory / pattern)

        self._registered_datasets[name] = (
            f"file('{self._escape_sql_string(glob_expr)}', Parquet)"
        )

    def get_hf_token(self) -> str | None:
        """
        Resolve the optional Hugging Face token.

        Precedence:
          1. ClickHouseConfig.hf_token
          2. HF_TOKEN environment variable

        Public datasets work without a token.
        """
        return self.config.hf_token or os.environ.get("HF_TOKEN")

    def _build_hf_url_source(
        self,
        url: str,
        *,
        token: str | None = None,
    ) -> str:
        """
        Build a ClickHouse url() source for a Hugging Face Parquet dataset.

        When token is absent, the source is identical to the public-dataset
        path. When token is present, it is passed as an Authorization header
        rather than embedded in the URL.
        """
        escaped_url = self._escape_sql_string(url)

        if not token:
            return f"url('{escaped_url}', Parquet)"

        escaped_token = self._escape_sql_string(token)

        return (
            f"url('{escaped_url}', Parquet, 'auto', "
            f"headers('Authorization'='Bearer {escaped_token}'))"
        )

    def register_hf_dataset(
        self,
        name: str,
        url_glob: str,
        *,
        revision: str = "main",
        pattern: str = "*.parquet",
    ) -> None:
        """
        Register a Hugging Face dataset as a remote ClickHouse `url()` source.

        The dataset is NOT downloaded by Python. For wildcard URLs, the HF
        dataset tree API is queried once to discover the Parquet shards, and
        the resulting shard list is compressed into a ClickHouse brace
        expression where possible.

        NOTE: every query against this source reads the matched remote
        Parquet shards over HTTP. This resource deliberately does not stage
        or persist a local dataset cache.

        Example:
            register_hf_dataset(
                "cpu",
                "https://huggingface.co/datasets/AINovice2005/"
                "carbon-pilot-corpus-dedup/resolve/main/*.parquet",
            )
        """
        self._validate_relation_name(name)

        resolved_glob = self._resolve_hf_glob(
            url_glob,
            revision=revision,
            pattern=pattern,
        )

        token = self.get_hf_token()

        source = self._build_hf_url_source(
            resolved_glob,
            token=token,
        )

        self._registered_datasets[name] = source

    def register_hf_shards(
        self,
        name: str,
        shard_urls: list[str],
    ) -> None:
        """
        Register an explicit list of Hugging Face Parquet shard URLs.

        This is useful when shard discovery has already happened elsewhere.
        """
        self._validate_relation_name(name)

        if not shard_urls:
            raise ValueError("shard_urls cannot be empty")

        normalized = [
            url for url in shard_urls
            if urlparse(url).scheme in {"http", "https"}
        ]

        if len(normalized) != len(shard_urls):
            raise ValueError(
                "All Hugging Face shard URLs must use http:// or https://"
            )

        source = self._build_hf_url_source(
            self._hf_urls_to_clickhouse_glob(normalized),
            token=self.get_hf_token(),
        )

        self._registered_datasets[name] = source

    def source_expr(self, name: str) -> str:
        """Return the registered ClickHouse table-function source expression."""
        if name not in self._registered_datasets:
            raise KeyError(f"No dataset registered under name {name!r}")

        return self._registered_datasets[name]

    # ------------------------------------------------------------------
    # Hugging Face shard discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover_hf_shards(
        cls,
        url_glob: str,
        *,
        revision: str = "main",
        pattern: str = "*.parquet",
        timeout: int = 30,
    ) -> list[str]:
        """
        Resolve a Hugging Face dataset URL/glob into concrete Parquet URLs.

        This mirrors the shard discovery used by build_cohort_pipeline.py:
        the HF dataset tree API is queried only for metadata; the actual
        Parquet data remains remote and is read by ClickHouse.
        """
        if not cls._is_hf_dataset_url(url_glob):
            raise ValueError(
                "Hugging Face URL must have the form "
                "https://huggingface.co/datasets/<org>/<repo>/..."
            )

        parsed = urlparse(url_glob)

        # If there is no wildcard, treat it as an explicit Parquet object.
        if not any(ch in parsed.path for ch in "*?{"):
            return [url_glob]

        match = re.match(
            r"^/datasets/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$",
            parsed.path,
        )
        if not match:
            raise ValueError(
                "Hugging Face wildcard expansion requires a URL of the form "
                "https://huggingface.co/datasets/<org>/<repo>/resolve/"
                "<revision>/..."
            )

        repo = match.group(1)
        url_revision = match.group(2)
        path_pattern = match.group(3)

        # The explicit revision in the URL wins unless the caller supplied
        # a different revision for a URL that uses the default main branch.
        if revision != "main" and url_revision == "main":
            url_revision = revision

        api_url = (
            f"https://huggingface.co/api/datasets/{repo}/tree/"
            f"{url_revision}?recursive=true&limit=1000"
        )

        req = Request(
            api_url,
            headers={"User-Agent": "carbon-enrichment-clickhouse/1.0"},
        )

        try:
            with urlopen(req, timeout=timeout) as response:
                payload = json.load(response)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query Hugging Face dataset tree for "
                f"{repo}@{url_revision}: {exc}"
            ) from exc

        shard_paths: list[str] = []

        for entry in payload:
            path = str(entry.get("path", ""))

            if not path.endswith(".parquet"):
                continue

            if fnmatch.fnmatch(path, path_pattern):
                shard_paths.append(path)

        if not shard_paths:
            raise RuntimeError(
                f"No Parquet shards matched {url_glob} in Hugging Face "
                f"dataset {repo}@{url_revision}."
            )

        shard_paths.sort()

        return [
            f"https://huggingface.co/datasets/{repo}/resolve/"
            f"{url_revision}/{path}"
            for path in shard_paths
        ]

    @classmethod
    def _resolve_hf_glob(
        cls,
        url_glob: str,
        *,
        revision: str = "main",
        pattern: str = "*.parquet",
    ) -> str:
        urls = cls.discover_hf_shards(
            url_glob,
            revision=revision,
            pattern=pattern,
        )

        return cls._hf_urls_to_clickhouse_glob(urls)

    @staticmethod
    def _hf_urls_to_clickhouse_glob(urls: list[str]) -> str:
        """
        Compress shard-N.parquet URLs into ClickHouse brace syntax.

        For non-contiguous or non-standard shard names, emit an explicit
        brace list. This keeps the behavior deterministic and avoids
        relying on an HTTP wildcard that HF itself may not support.
        """
        if not urls:
            raise ValueError("Cannot build ClickHouse source from no URLs")

        if len(urls) == 1:
            return urls[0]

        return "{" + ",".join(urls) + "}"

    @staticmethod
    def _is_hf_dataset_url(url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.scheme in {"http", "https"}
            and parsed.netloc == "huggingface.co"
            and parsed.path.startswith("/datasets/")
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _base_args(
        self,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        args = [
            self.get_binary(),
            "local",
        ]

        if self.config.threads is not None:
            args += ["--max_threads", str(self.config.threads)]

        args += [
            "--join_algorithm",
            self.config.join_algorithm,
            "--joined_subquery_requires_alias",
            "0",
            "--max_http_get_redirects",
            str(self.config.max_http_get_redirects),
            "--allow_experimental_url_wildcard_from_index_pages",
            "1",
            "--progress",
        ]

        if extra_args:
            args += extra_args

        return args

    @staticmethod
    def _substitute_parameters(
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None,
    ) -> str:
        """
        Minimal positional `?` substitution.

        This mirrors the previous DuckDB-style resource interface used by
        consuming modules. Values are SQL-quoted as strings.
        """
        if not parameters:
            return sql

        remaining = list(parameters)
        out: list[str] = []
        i = 0

        while i < len(sql):
            if sql[i] == "?" and remaining:
                value = remaining.pop(0)
                escaped = str(value).replace("'", "''")
                out.append(f"'{escaped}'")
            else:
                out.append(sql[i])
            i += 1

        if remaining:
            raise ValueError(
                f"{len(remaining)} query parameters were not consumed"
            )

        return "".join(out)

    def query_arrow(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None = None,
        extra_args: list[str] | None = None,
    ) -> pa.Table:
        """
        Execute SQL and return the full result as an Arrow table.

        This fully buffers the result: once inside the `clickhouse local`
        child process (building the Arrow-format output), once in the
        pipe buffer captured by `subprocess.run`, and once more as the
        materialized `pa.Table`. Fine for small/aggregate results
        (summaries, top-N, correlations); for a large row-level result,
        prefer `stream_arrow_batches` and write batches out incrementally
        instead of calling this.
        """
        final_sql = self._substitute_parameters(sql, parameters)

        cmd = self._base_args(extra_args) + [
            "--query",
            final_sql,
            "--format",
            "Arrow",
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"clickhouse local query failed with exit code "
                f"{result.returncode}"
            )

        table = pa.ipc.open_file(
            pa.BufferReader(result.stdout)
        ).read_all()

        return table

    def stream_arrow_batches(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None = None,
        extra_args: list[str] | None = None,
    ):
        """
        Execute SQL and stream Arrow record batches without materializing the
        whole result in this process's memory at once.

        Uses ClickHouse's ArrowStream output format, piped directly into
        PyArrow's streaming reader. Prefer this over `query_arrow` for any
        query whose result could be large (row-level analytical output,
        not aggregate summaries) — peak Python-side memory is bounded by
        one batch rather than the full result set.
        """
        final_sql = self._substitute_parameters(sql, parameters)

        cmd = self._base_args(extra_args) + [
            "--query",
            final_sql,
            "--format",
            "ArrowStream",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
        )

        if process.stdout is None:
            process.kill()
            raise RuntimeError(
                "Failed to open ClickHouse stdout pipe."
            )

        try:
            reader = pa.ipc.open_stream(process.stdout)

            for batch in reader:
                yield batch

        finally:
            process.stdout.close()

            returncode = process.wait()

            if returncode != 0:
                raise RuntimeError(
                    f"clickhouse local streaming query failed with exit code "
                    f"{returncode}"
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def registered_datasets(self) -> dict[str, str]:
        """Return a copy of currently registered dataset source expressions."""
        return dict(self._registered_datasets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _validate_relation_name(name: str) -> None:
        if not name:
            raise ValueError("Dataset name cannot be empty.")

        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid dataset name: {name!r}")