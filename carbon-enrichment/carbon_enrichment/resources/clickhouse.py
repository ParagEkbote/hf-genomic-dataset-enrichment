from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pyarrow as pa

DEFAULT_THREADS = max(1, (os.cpu_count() or 1) - 1)
DEFAULT_BINARY = "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment/carbon-enrichment/clickhouse"
DEFAULT_TEMP_DIRECTORY = Path("data/tmp/clickhouse")


@dataclass(frozen=True)
class ClickHouseConfig:
    binary_path: str = DEFAULT_BINARY
    threads: int | None = DEFAULT_THREADS
    temp_directory: str | Path | None = DEFAULT_TEMP_DIRECTORY
    join_algorithm: str = "grace_hash"
    max_http_get_redirects: int = 10
    hf_token: str | None = None
    # Parallelism for remote url()/file() reads.
    max_download_threads: int = DEFAULT_THREADS
    max_parsing_threads: int = DEFAULT_THREADS
    max_download_buffer_size: int = 10 * 1024 * 1024  # 10 MiB


class ClickHouseResource:
    def __init__(self, config: ClickHouseConfig | None = None) -> None:
        self.config = config or ClickHouseConfig()
        self._binary: str | None = None
        self._registered_datasets: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Binary lifecycle
    # ------------------------------------------------------------------

    def get_binary(self) -> str:
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
            Path(self.config.temp_directory).expanduser().mkdir(parents=True, exist_ok=True)
        return self._binary

    def close(self) -> None:
        self._binary = None

    def __enter__(self) -> "ClickHouseResource":
        self.get_binary()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    # ------------------------------------------------------------------
    # Dataset registration (unchanged)
    # ------------------------------------------------------------------

    def register_dataset(self, name: str, path: str | Path) -> None:
        self._validate_relation_name(name)
        dataset_path = Path(path).expanduser()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Parquet dataset does not exist: {dataset_path}")
        glob_expr = str(dataset_path / "*.parquet") if dataset_path.is_dir() else str(dataset_path)
        self._registered_datasets[name] = f"file('{self._escape_sql_string(glob_expr)}', Parquet)"

    def register_enriched_glob(self, name: str, directory: str | Path, pattern: str = "batch_*.parquet") -> None:
        self._validate_relation_name(name)
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise FileNotFoundError(f"Enriched output directory does not exist: {directory}")
        glob_expr = str(directory / pattern)
        self._registered_datasets[name] = f"file('{self._escape_sql_string(glob_expr)}', Parquet)"

    def get_hf_token(self) -> str | None:
        return self.config.hf_token or os.environ.get("HF_TOKEN")

    def _build_hf_url_source(self, url: str, *, token: str | None = None) -> str:
        escaped_url = self._escape_sql_string(url)
        if not token:
            return f"url('{escaped_url}', Parquet)"
        escaped_token = self._escape_sql_string(token)
        return f"url('{escaped_url}', Parquet, 'auto', headers('Authorization'='Bearer {escaped_token}'))"

    def register_hf_dataset(self, name: str, url_glob: str, *, revision: str = "main", pattern: str = "*.parquet") -> None:
        self._validate_relation_name(name)
        resolved_glob = self._resolve_hf_glob(url_glob, revision=revision, pattern=pattern)
        source = self._build_hf_url_source(resolved_glob, token=self.get_hf_token())
        self._registered_datasets[name] = source

    def register_hf_shards(self, name: str, shard_urls: list[str]) -> None:
        self._validate_relation_name(name)
        if not shard_urls:
            raise ValueError("shard_urls cannot be empty")
        normalized = [url for url in shard_urls if urlparse(url).scheme in {"http", "https"}]
        if len(normalized) != len(shard_urls):
            raise ValueError("All Hugging Face shard URLs must use http:// or https://")
        source = self._build_hf_url_source(self._hf_urls_to_clickhouse_glob(normalized), token=self.get_hf_token())
        self._registered_datasets[name] = source

    def source_expr(self, name: str) -> str:
        if name not in self._registered_datasets:
            raise KeyError(f"No dataset registered under name {name!r}")
        return self._registered_datasets[name]

    # ------------------------------------------------------------------
    # Hugging Face shard discovery (unchanged)
    # ------------------------------------------------------------------

    @classmethod
    def discover_hf_shards(cls, url_glob: str, *, revision: str = "main", pattern: str = "*.parquet", timeout: int = 30) -> list[str]:
        if not cls._is_hf_dataset_url(url_glob):
            raise ValueError("Hugging Face URL must have the form https://huggingface.co/datasets/<org>/<repo>/...")
        parsed = urlparse(url_glob)
        if not any(ch in parsed.path for ch in "*?{"):
            return [url_glob]
        match = re.match(r"^/datasets/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$", parsed.path)
        if not match:
            raise ValueError(
                "Hugging Face wildcard expansion requires a URL of the form "
                "https://huggingface.co/datasets/<org>/<repo>/resolve/<revision>/..."
            )
        repo = match.group(1)
        url_revision = match.group(2)
        path_pattern = match.group(3)
        if revision != "main" and url_revision == "main":
            url_revision = revision
        api_url = f"https://huggingface.co/api/datasets/{repo}/tree/{url_revision}?recursive=true&limit=1000"
        req = Request(api_url, headers={"User-Agent": "carbon-enrichment-clickhouse/1.0"})
        try:
            with urlopen(req, timeout=timeout) as response:
                payload = json.load(response)
        except Exception as exc:
            raise RuntimeError(f"Failed to query Hugging Face dataset tree for {repo}@{url_revision}: {exc}") from exc
        shard_paths: list[str] = []
        for entry in payload:
            path = str(entry.get("path", ""))
            if not path.endswith(".parquet"):
                continue
            if fnmatch.fnmatch(path, path_pattern):
                shard_paths.append(path)
        if not shard_paths:
            raise RuntimeError(f"No Parquet shards matched {url_glob} in Hugging Face dataset {repo}@{url_revision}.")
        shard_paths.sort()
        return [f"https://huggingface.co/datasets/{repo}/resolve/{url_revision}/{path}" for path in shard_paths]

    @classmethod
    def _resolve_hf_glob(cls, url_glob: str, *, revision: str = "main", pattern: str = "*.parquet") -> str:
        urls = cls.discover_hf_shards(url_glob, revision=revision, pattern=pattern)
        return cls._hf_urls_to_clickhouse_glob(urls)

    @staticmethod
    def _hf_urls_to_clickhouse_glob(urls: list[str]) -> str:
        if not urls:
            raise ValueError("Cannot build ClickHouse source from no URLs")
        if len(urls) == 1:
            return urls[0]
        return "{" + ",".join(urls) + "}"

    @staticmethod
    def _is_hf_dataset_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc == "huggingface.co" and parsed.path.startswith("/datasets/")

    # ------------------------------------------------------------------
    # Query / execute
    # ------------------------------------------------------------------

    def _base_args(self, extra_args: list[str] | None = None) -> list[str]:
        args = [self.get_binary(), "local"]
        if self.config.threads is not None:
            args += ["--max_threads", str(self.config.threads)]
        args += [
            "--join_algorithm", self.config.join_algorithm,
            "--joined_subquery_requires_alias", "0",
            "--max_http_get_redirects", str(self.config.max_http_get_redirects),
            "--allow_experimental_url_wildcard_from_index_pages", "1",
            "--max_download_threads", str(self.config.max_download_threads),
            "--max_parsing_threads", str(self.config.max_parsing_threads),
            "--max_download_buffer_size", str(self.config.max_download_buffer_size),
            "--input_format_parquet_use_native_reader", "1",
            "--remote_filesystem_read_method", "threadpool",
            
            # --- PERFORMANCE FLAGS FOR PARALLEL READ/WRITE ---
            "--max_insert_threads", str(self.config.threads or 8),
            "--remote_filesystem_read_prefetch", "1",
            
            "--progress",
        ]
        if extra_args:
            args += extra_args
        return args

    @staticmethod
    def _substitute_parameters(sql: str, parameters: list[Any] | tuple[Any, ...] | None) -> str:
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
            raise ValueError(f"{len(remaining)} query parameters were not consumed")
        return "".join(out)

    def execute_sql(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Run a statement with no result set: INSERT INTO FUNCTION ...
        SELECT ..., CREATE, TRUNCATE, etc. Does NOT append FORMAT — that
        was being tacked onto INSERT statements by query_arrow and is the
        main reason ingestion throughput collapsed to ~1k rows/sec."""
        final_sql = self._substitute_parameters(sql, parameters)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", encoding="utf-8", delete=False) as f:
            f.write(final_sql)
            query_file = f.name
        try:
            cmd = self._base_args(extra_args) + ["--queries-file", query_file]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None)
            if result.returncode != 0:
                raise RuntimeError(f"clickhouse local statement failed with exit code {result.returncode}")
        finally:
            Path(query_file).unlink(missing_ok=True)

    def execute_sql_with_progress(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] | None = None,
        extra_args: list[str] | None = None,
        progress_callback: Any | None = None,
        output_callback: Any | None = None,
    ) -> None:
        """Execute a statement while forwarding ClickHouse progress output.

        ``progress_callback`` receives the cumulative processed row count.
        ``output_callback`` receives non-progress stderr messages.

        ClickHouse commonly refreshes progress using carriage returns rather
        than newline characters, so stderr is consumed in a background
        thread to keep notebook progress updates responsive.
        """
        import threading

        final_sql = self._substitute_parameters(sql, parameters)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sql",
            encoding="utf-8",
            delete=False,
        ) as f:
            f.write(final_sql)
            query_file = f.name

        progress_pattern = re.compile(r"(?P<rows>[0-9][0-9 ]*)\s+rows")
        ansi_pattern = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        last_rows = 0
        stderr_messages: list[str] = []
        stderr_buffer = ""

        try:
            cmd = self._base_args(extra_args) + ["--queries-file", query_file]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            if process.stderr is None:
                process.kill()
                raise RuntimeError("Failed to open ClickHouse stderr pipe.")

            def consume_stderr() -> None:
                nonlocal last_rows, stderr_buffer

                while True:
                    chunk = process.stderr.read(4096)
                    if not chunk:
                        break

                    stderr_buffer += chunk.decode("utf-8", errors="replace")
                    parts = re.split(r"[\r\n]", stderr_buffer)
                    stderr_buffer = parts.pop() or ""

                    for raw_message in parts:
                        message = ansi_pattern.sub("", raw_message).strip()
                        if not message:
                            continue

                        match = progress_pattern.search(message)
                        if match:
                            rows = int(match.group("rows").replace(" ", ""))
                            if rows > last_rows:
                                last_rows = rows
                                if progress_callback is not None:
                                    progress_callback(rows)
                        elif output_callback is not None:
                            stderr_messages.append(message)
                            output_callback(message)

                trailing = ansi_pattern.sub("", stderr_buffer).strip()
                if trailing and output_callback is not None:
                    output_callback(trailing)

            reader = threading.Thread(
                target=consume_stderr,
                name="clickhouse-progress-reader",
                daemon=True,
            )
            reader.start()

            returncode = process.wait()
            reader.join()

            if returncode != 0:
                diagnostics = "\n".join(stderr_messages[-20:])
                detail = f"\n{diagnostics}" if diagnostics else ""
                raise RuntimeError(
                    "clickhouse local statement failed with "
                    f"exit code {returncode}.{detail}"
                )

            if progress_callback is not None and last_rows:
                progress_callback(last_rows)

        finally:
            Path(query_file).unlink(missing_ok=True)

    def query_arrow(self, sql: str, parameters: list[Any] | tuple[Any, ...] | None = None, extra_args: list[str] | None = None) -> pa.Table:
        """For statements that return rows only. Do not use this for
        INSERT/DDL — use execute_sql instead."""
        final_sql = self._substitute_parameters(sql, parameters)
        final_sql = f"""
        {final_sql}
        FORMAT Arrow
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", encoding="utf-8", delete=False) as f:
            f.write(final_sql)
            query_file = f.name
        try:
            cmd = self._base_args(extra_args) + ["--queries-file", query_file]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None)
            if result.returncode != 0:
                raise RuntimeError(f"clickhouse local query failed with exit code {result.returncode}")
            return pa.ipc.open_file(pa.BufferReader(result.stdout)).read_all()
        finally:
            Path(query_file).unlink(missing_ok=True)

    def stream_arrow_batches(self, sql: str, parameters: list[Any] | tuple[Any, ...] | None = None, extra_args: list[str] | None = None):
        final_sql = self._substitute_parameters(sql, parameters)
        cmd = self._base_args(extra_args) + ["--query", final_sql, "--format", "ArrowStream"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
        if process.stdout is None:
            process.kill()
            raise RuntimeError("Failed to open ClickHouse stdout pipe.")
        try:
            reader = pa.ipc.open_stream(process.stdout)
            for batch in reader:
                yield batch
        finally:
            process.stdout.close()
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"clickhouse local streaming query failed with exit code {returncode}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def registered_datasets(self) -> dict[str, str]:
        return dict(self._registered_datasets)

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _validate_relation_name(name: str) -> None:
        if not name:
            raise ValueError("Dataset name cannot be empty.")
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid dataset name: {name!r}")