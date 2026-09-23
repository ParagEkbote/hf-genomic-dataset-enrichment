"""
Taxonomy-targeted ingestion and CPU-only biological characterization of
genomic cohorts, built on ClickHouseResource / `clickhouse local`.

Source dataset:
    AINovice2005/carbon-cpu-enriched-sequences  (Hugging Face, Parquet)

Workflow:
    remote Parquet shards (no download)
        -> registered as a ClickHouse url() source
        -> DESCRIBE'd once to build a column-alias / normalization map
        -> taxonomy predicate pushed down as SQL (match()), not Python
        -> matching rows written to a local Parquet cache with
           `INSERT INTO FUNCTION file(...)` (single pass, no server)
        -> every analysis cell queries the local cache, so charts never
           re-hit the network.

Marimo note: only the last top-level expression of a cell is displayed, so
every figure cell assigns its output to a cell-local `_out` variable inside
the if/else and ends with a bare `_out`.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # Adjust this import path if clickhouse_resource.py lives elsewhere
    # relative to this notebook.
    from carbon_enrichment.resources.clickhouse import ClickHouseConfig, ClickHouseResource

    return (
        ClickHouseConfig,
        ClickHouseResource,
        Path,
        go,
        mo,
        np,
        pd,
        px,
        re,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # CPU-feature characterization of a taxonomy-targeted cohort

    This notebook pulls one taxonomic group out of the CPU-enriched genomic
    dataset on Hugging Face, caches it locally, and profiles its measurable
    sequence properties: GC content, interval length, GC skew, coding state,
    and record-level heterogeneity.

    **How it works**

    1. The remote Parquet shards are read in place through `clickhouse local`
       (nothing is downloaded up front and no ClickHouse server is needed).
    2. The taxonomy filter runs as SQL inside that read, so only matching
       intervals are written to a local Parquet cache.
    3. Every figure below queries the local cache, which keeps reruns fast and
       offline.

    **How to use it**

    Set the source and target in *1 · Configure*. If a cache already exists
    for the chosen name, it is reused automatically. Otherwise, click
    **Run targeted ingestion**.

    *Terminology:* an **interval** is one annotated region (one row); a
    **record** is one sequence accession (`record_id`) that can hold many
    intervals.
    """)
    return


@app.cell
def _(mo):
    dataset_repo = mo.ui.text(
        value="AINovice2005/carbon-cpu-enriched-sequences",
        label="Hugging Face dataset (`org/name`)",
        full_width=True,
    )
    dataset_revision = mo.ui.text(
        value="main",
        label="Revision (branch, tag or commit)",
        full_width=True,
    )
    shard_pattern = mo.ui.text(
        value="*.parquet",
        label="Parquet shard pattern",
        full_width=True,
    )
    target_lineage = mo.ui.text(
        value="Fungi;Pleosporales;Pleosporaceae;Alternaria",
        label="Target taxon or lineage (`;`-separated, broad → specific)",
        full_width=True,
    )
    taxonomy_mode = mo.ui.dropdown(
        options={
            "Exact taxon token": "exact",
            "Ordered lineage": "lineage",
            "Taxon token and descendants": "descendants",
        },
        value="Ordered lineage",
        label="Taxonomy matching mode",
    )
    cache_name = mo.ui.text(
        value="targeted_cpu_enriched",
        label="Cache file name (without `.parquet`)",
        full_width=True,
    )
    reuse_cache = mo.ui.switch(
        value=True,
        label="Reuse existing cache if present (turn off to rebuild)",
    )
    run_ingestion = mo.ui.run_button(
        label="Run targeted ingestion",
        kind="success",
        tooltip="Stream matching intervals from Hugging Face into the local cache",
    )

    mo.vstack(
        [
            mo.md("## 1 · Configure"),
            mo.md("**Source dataset**"),
            dataset_repo,
            dataset_revision,
            shard_pattern,
            mo.md("**Target cohort**"),
            target_lineage,
            taxonomy_mode,
            mo.accordion(
                {
                    "What do the matching modes do?": mo.md(
                        """
                        Matching is case-insensitive and works on whole
                        lineage tokens, so `Alternaria` will not match
                        `Alternariaster`.

                        - **Exact taxon token**: keeps intervals whose lineage
                          contains the *last* entry above, at any rank.
                        - **Ordered lineage**: keeps intervals whose lineage
                          contains *every* entry, in the order given. Other
                          ranks may sit between them.
                        - **Taxon token and descendants**: currently applies
                          the same filter as *Exact taxon token*. Lineages are
                          stored broad → specific, so descendants of the taxon
                          contain its token and are matched too.
                        """
                    )
                }
            ),
            mo.md("**Local cache**"),
            cache_name,
            reuse_cache,
            run_ingestion,
        ]
    )
    return (
        cache_name,
        dataset_repo,
        dataset_revision,
        reuse_cache,
        run_ingestion,
        shard_pattern,
        target_lineage,
        taxonomy_mode,
    )


@app.cell
def _(ClickHouseConfig, ClickHouseResource):
    # Aggressively parallelize HTTP downloads and parsing, overriding the
    # default notebook CPU core count.
    config = ClickHouseConfig(
        threads=16,
        max_download_threads=32,
        max_parsing_threads=16,
        max_download_buffer_size=50 * 1024 * 1024,  # 50 MiB
    )
    resource = ClickHouseResource(config)
    resource.get_binary()  # fail fast if `clickhouse` isn't on PATH
    return (resource,)


@app.cell
def _(Path, cache_name, mo, re, reuse_cache):
    cache_dir = Path(
        "/teamspace/studios/this_studio/"
        "hf-genomic-dataset-enrichment/carbon-enrichment/"
        "carbon_enrichment/results/notebook/data/cache/clickhouse"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    _stem = re.sub(r"[^A-Za-z0-9_.-]", "_", cache_name.value.strip())
    if _stem.lower().endswith(".parquet"):
        _stem = _stem[: -len(".parquet")]
    cache_path = cache_dir / f"{_stem or 'targeted_cpu_enriched'}.parquet"
    reuse_existing_cache = reuse_cache.value

    if cache_path.is_file():
        _status = f"found ({cache_path.stat().st_size / 1e6:,.1f} MB)"
    else:
        _status = "not created yet"

    mo.md(
        f"""
        **Cache file:** `{cache_path}`

        **Status:** {_status} &nbsp;·&nbsp; **Reuse if present:** `{reuse_existing_cache}`
        """
    )
    return cache_path, reuse_existing_cache


@app.cell
def _(mo, re, target_lineage, taxonomy_mode):
    def build_taxonomy_pattern(lineage: str, mode: str) -> str:
        """Express the selected matching mode as a single RE2 pattern so it
        can be pushed into ClickHouse's `match()`."""
        taxa = [part.strip() for part in lineage.split(";") if part.strip()]
        if mode in {"exact", "descendants"}:
            return r"(?i)(^|;)" + re.escape(taxa[-1]) + r"(;|$)"
        pattern = r"(?i)(^|;)" + re.escape(taxa[0])
        for taxon in taxa[1:]:
            pattern += r"(?:;[^;]+)*;" + re.escape(taxon)
        return pattern + r"(?:;|$)"

    def sql_quote(value: str) -> str:
        return value.replace("'", "''")

    mo.stop(
        not [p for p in target_lineage.value.split(";") if p.strip()],
        mo.callout(
            mo.md("Enter at least one taxon or lineage component above."),
            kind="warn",
        ),
    )

    selected_taxonomy_pattern = build_taxonomy_pattern(
        target_lineage.value, taxonomy_mode.value
    )

    mo.accordion(
        {
            "Generated taxonomy filter (RE2 pattern passed to ClickHouse `match()`)": mo.md(
                f"`{selected_taxonomy_pattern}`"
            )
        }
    )
    return selected_taxonomy_pattern, sql_quote


@app.cell
def _(
    cache_path,
    dataset_repo,
    dataset_revision,
    mo,
    resource,
    reuse_existing_cache,
    run_ingestion,
    selected_taxonomy_pattern,
    shard_pattern,
    sql_quote,
    target_lineage,
    taxonomy_mode,
    time,
):
    def _ingest():
        hf_glob = (
            f"https://huggingface.co/datasets/{dataset_repo.value.strip()}"
            f"/resolve/{dataset_revision.value.strip()}/{shard_pattern.value.strip()}"
        )

        resource.register_hf_dataset(
            "hf_source",
            hf_glob,
            revision=dataset_revision.value.strip(),
            pattern=shard_pattern.value.strip(),
        )
        hf_source_expr = resource.source_expr("hf_source")

        # ---- Schema discovery ----
        columns_table = resource.query_arrow(f"DESCRIBE {hf_source_expr}")
        available_columns = set(columns_table.column("name").to_pylist())

        ALIASES = {
            "record_id": ["record_id", "id", "sequence_id"],
            "start": ["start", "begin"],
            "end": ["end", "stop"],
            "sequence_length": ["sequence_length", "length", "seq_length"],
            "gc_content": ["gc_content", "gc_fraction", "gc"],
            "gc_skew": ["gc_skew", "gc_skewness"],
            "strand": ["strand"],
            "shannon_entropy": ["shannon_entropy", "entropy"],
            "gene_length": ["gene_length"],
            "is_coding_region": ["is_coding_region", "is_coding", "coding"],
            "taxonomy": ["taxonomy", "taxonomy_lineage", "taxon"],
            "taxonomy_depth": ["taxonomy_depth"],
            "taxonomy_domain": ["taxonomy_domain"],
            "qc_flag": ["qc_flag"],
        }

        def resolve(target):
            for candidate in ALIASES[target]:
                if candidate in available_columns:
                    return candidate
            return None

        resolved = {t: resolve(t) for t in ALIASES}

        required = ["record_id", "start", "end", "sequence_length", "gc_content", "taxonomy"]
        missing = [f for f in required if resolved[f] is None]
        if missing:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        f"**Missing required fields:** `{missing}`\n\n"
                        f"Available columns: `{sorted(available_columns)}`"
                    ),
                    kind="danger",
                ),
            )

        def select_expr(target, cast=None):
            source_col = resolved[target]
            expr = f"`{source_col}`" if source_col else "NULL"
            if cast:
                expr = f"CAST({expr} AS {cast})"
            return f"{expr} AS {target}"

        # ---- SQL projection ----
        gc_col = resolved["gc_content"]
        gc_expr = (
            f"(CASE WHEN `{gc_col}` > 1 THEN `{gc_col}` / 100.0 "
            f"ELSE `{gc_col}` END) AS gc_content"
        )

        select_list = [
            select_expr("record_id", "String"),
            select_expr("start", "UInt64"),
            select_expr("end", "UInt64"),
            select_expr("sequence_length", "UInt64"),
            gc_expr,
            select_expr("gc_skew", "Nullable(Float32)"),
            (
                f"toString(coalesce(`{resolved['strand']}`, '')) AS strand"
                if resolved["strand"] else "'' AS strand"
            ),
            select_expr("shannon_entropy", "Nullable(Float32)"),
            select_expr("gene_length", "Nullable(UInt64)"),
            (
                f"CAST(`{resolved['is_coding_region']}` AS Nullable(UInt8)) AS is_coding_region"
                if resolved["is_coding_region"] else "NULL AS is_coding_region"
            ),
            f"trim(toString(`{resolved['taxonomy']}`)) AS taxonomy",
            select_expr("taxonomy_depth", "Nullable(UInt16)"),
            (
                f"toString(coalesce(`{resolved['taxonomy_domain']}`, '')) AS taxonomy_domain"
                if resolved["taxonomy_domain"] else "'' AS taxonomy_domain"
            ),
            (
                f"toString(coalesce(`{resolved['qc_flag']}`, '')) AS qc_flag"
                if resolved["qc_flag"] else "'' AS qc_flag"
            ),
            "generateUUIDv4() AS ingestion_id",
            f"'{sql_quote(target_lineage.value.strip())}' AS target_lineage",
            f"'{sql_quote(taxonomy_mode.value)}' AS match_mode",
            "now() AS ingested_at",
        ]

        filter_sql = (
            f"match(trim(toString(`{resolved['taxonomy']}`)), "
            f"'{sql_quote(selected_taxonomy_pattern)}')"
        )

        ingest_sql = f"""
        INSERT INTO FUNCTION file('{sql_quote(str(cache_path))}', Parquet)
        SELECT {", ".join(select_list)}
        FROM {hf_source_expr}
        WHERE {filter_sql}
        """

        # ---- Progress (no preliminary COUNT; avoids a second remote scan) ----
        progress_state = {"last_rows": 0, "last_message": ""}
        started = time.perf_counter()

        def update_progress(processed_rows: int) -> None:
            inc = processed_rows - progress_state["last_rows"]
            if inc > 0:
                progress.update(increment=inc)
                progress_state["last_rows"] = processed_rows

        def output_callback(message: str) -> None:
            progress_state["last_message"] = message

        # A rebuild replaces the old file; ClickHouse will not append to or
        # overwrite an existing file() target by default.
        cache_path.unlink(missing_ok=True)

        with mo.status.progress_bar(
            title="Targeted ingestion",
            subtitle="Reading remote Parquet shards and writing the local cache",
            show_eta=False,
            show_rate=True,
        ) as progress:
            resource.execute_sql_with_progress(
                ingest_sql,
                progress_callback=update_progress,
                output_callback=output_callback,
            )

        elapsed = time.perf_counter() - started
        rows_processed = progress_state["last_rows"]
        rate = rows_processed / elapsed if elapsed > 0 else 0.0

        # ---- Register cache + verify ----
        resource.register_dataset("cohort", cache_path)
        expr = resource.source_expr("cohort")

        summary_table = resource.query_arrow(
            f"""
            SELECT count() AS rows_inserted, uniqExact(record_id) AS unique_records
            FROM {expr}
            """
        )
        rows_inserted = summary_table.column("rows_inserted")[0].as_py()
        unique = summary_table.column("unique_records")[0].as_py()

        ui = mo.vstack(
            [
                mo.md("## 2 · Ingestion result"),
                mo.ui.table([{"intervals_cached": rows_inserted, "unique_records": unique}]),
                mo.md(
                    f"""
                    **Target:** `{target_lineage.value}` &nbsp;·&nbsp; **Mode:** `{taxonomy_mode.value}`

                    **Source rows scanned (reported):** `{rows_processed:,}` &nbsp;·&nbsp;
                    **Elapsed:** `{elapsed:.1f} s` &nbsp;·&nbsp;
                    **Rate:** `{rate:,.0f} rows/s`

                    Matching intervals were streamed from the remote Parquet
                    shards straight into `{cache_path}`. The full dataset was
                    never materialized in Python.
                    """
                ),
            ]
        )
        return expr, ui

    # ---------------------------------------------------------
    # Existing-cache fast path vs. remote ingestion
    # ---------------------------------------------------------
    if reuse_existing_cache and cache_path.is_file():
        resource.register_dataset("cohort", cache_path)
        cache_expr = resource.source_expr("cohort")

        cached_summary = resource.query_arrow(
            f"""
            SELECT count() AS intervals_cached, uniqExact(record_id) AS unique_records
            FROM {cache_expr}
            """
        )

        # Which target was this cache built for? (older caches may lack these columns)
        try:
            _meta = resource.query_arrow(
                f"""
                SELECT any(target_lineage) AS target_lineage,
                       any(match_mode) AS match_mode,
                       max(ingested_at) AS ingested_at
                FROM {cache_expr}
                """
            )
            cache_target = _meta.column("target_lineage")[0].as_py()
            cache_mode = _meta.column("match_mode")[0].as_py()
            cache_time = _meta.column("ingested_at")[0].as_py()
        except Exception:
            cache_target, cache_mode, cache_time = None, None, None

        _items = [
            mo.md("## 2 · Existing cache reused"),
            mo.ui.table(
                [
                    {
                        "intervals_cached": cached_summary.column("intervals_cached")[0].as_py(),
                        "unique_records": cached_summary.column("unique_records")[0].as_py(),
                    }
                ]
            ),
            mo.md(
                f"""
                The local Parquet cache was reused; the remote dataset was
                not scanned.

                **Cache:** `{cache_path}`
                """
            ),
        ]
        if cache_target is not None:
            _items.append(
                mo.md(
                    f"**Cache built for:** `{cache_target}` (mode `{cache_mode}`, "
                    f"ingested {cache_time:%Y-%m-%d %H:%M})"
                )
            )
            if (
                cache_target != target_lineage.value.strip()
                or cache_mode != taxonomy_mode.value
            ):
                _items.append(
                    mo.callout(
                        mo.md(
                            "The cache was built for a **different target** than the one "
                            "configured above. Every figure below describes the cache, not "
                            "the current target. Turn off *Reuse existing cache* or change "
                            "the cache file name, then run ingestion."
                        ),
                        kind="warn",
                    )
                )
        view = mo.vstack(_items)
    else:
        _reason = (
            "Cache reuse is turned off, so the cache will be rebuilt."
            if reuse_existing_cache is False
            else "No cache file exists at the path above yet."
        )
        mo.stop(
            not run_ingestion.value,
            mo.callout(
                mo.md(
                    f"{_reason} Click **Run targeted ingestion** to stream the "
                    "matching intervals from Hugging Face."
                ),
                kind="warn",
            ),
        )
        cache_expr, view = _ingest()

    view
    return (cache_expr,)


@app.cell
def _(cache_expr, mo, resource):
    counts = resource.query_arrow(
        f"""
        SELECT
            count() AS intervals,
            uniqExact(record_id) AS unique_records,
            avg(sequence_length) AS mean_sequence_length,
            median(sequence_length) AS median_sequence_length,
            avg(gc_content) AS mean_gc_content,
            median(gc_content) AS median_gc_content
        FROM {cache_expr}
        """
    ).to_pandas()

    mo.vstack(
        [
            mo.md(
                """
                ## 3 · Cohort summary

                Size and central tendency of the cached cohort. Compare mean
                and median: a large gap means a skewed distribution
                (typically sequence length).
                """
            ),
            mo.ui.table(counts),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Distributions

    One variable at a time. Each histogram has a box plot on top showing the
    median, quartiles and outliers.
    """)
    return


@app.cell
def _(cache_expr, mo, px, resource):
    data_1 = resource.query_arrow(
        f"SELECT gc_content FROM {cache_expr} WHERE isFinite(gc_content)"
    ).to_pandas()

    if data_1.empty:
        _out = mo.md("### CPU-1 · GC content\n\nNo valid `gc_content` values are available.")
    else:
        fig_1 = px.histogram(
            data_1,
            x="gc_content",
            nbins=40,
            marginal="box",
            title="CPU-1 — GC content distribution",
            labels={"gc_content": "GC content (fraction)"},
            template="plotly_white",
        )
        _out = mo.vstack(
            [
                mo.md(
                    """
                    ### CPU-1 · GC content
                    Fraction of G and C bases per interval (0 to 1). Very low
                    or very high values can indicate short or low-complexity
                    intervals; see CPU-3 and CPU-7 for how length affects it.
                    """
                ),
                mo.ui.plotly(fig_1),
            ]
        )

    _out
    return


@app.cell
def _(cache_expr, mo, np, px, resource):
    data_2 = resource.query_arrow(
        f"SELECT sequence_length FROM {cache_expr} WHERE sequence_length > 0"
    ).to_pandas()

    if data_2.empty:
        _out = mo.md("### CPU-2 · Interval length\n\nNo valid `sequence_length` values are available.")
    else:
        data_2["log10_sequence_length"] = np.log10(data_2["sequence_length"])
        fig_2 = px.histogram(
            data_2,
            x="log10_sequence_length",
            nbins=40,
            marginal="box",
            title="CPU-2 — Sequence length distribution",
            labels={"log10_sequence_length": "log10(sequence length)"},
            template="plotly_white",
        )
        _out = mo.vstack(
            [
                mo.md(
                    """
                    ### CPU-2 · Interval length
                    Length in bases on a log10 scale (2 = 100 bp, 3 = 1 kb,
                    4 = 10 kb), because lengths span orders of magnitude.
                    """
                ),
                mo.ui.plotly(fig_2),
            ]
        )

    _out
    return


@app.cell
def _(cache_expr, mo, px, resource):
    data_3 = resource.query_arrow(
        f"""
        SELECT sequence_length, gc_content, is_coding_region
        FROM {cache_expr}
        WHERE sequence_length > 0 AND isFinite(gc_content)
        ORDER BY sipHash64(record_id, start, end)
        LIMIT 100000
        """
    ).to_pandas()

    if data_3.empty:
        _out = mo.md("### CPU-3 · GC content vs length\n\nNo data available.")
    else:
        color_column = "is_coding_region" if data_3["is_coding_region"].notna().any() else None
        fig_3 = px.scatter(
            data_3,
            x="sequence_length",
            y="gc_content",
            color=color_column,
            log_x=True,
            opacity=0.65,
            title="CPU-3 — GC content versus sequence length",
            labels={
                "sequence_length": "Sequence length (log scale)",
                "gc_content": "GC content (fraction)",
            },
            template="plotly_white",
        )
        _out = mo.vstack(
            [
                mo.md(
                    """
                    ### CPU-3 · GC content vs length
                    Each point is an interval; color marks coding status
                    (1 = coding, 0 = noncoding) when available. The sample is
                    capped at 100,000 intervals, chosen server-side with
                    `sipHash64`, so it is reproducible across reruns.
                    """
                ),
                mo.ui.plotly(fig_3),
            ]
        )

    _out
    return


@app.cell
def _(cache_expr, mo, px, resource):
    data_4 = resource.query_arrow(
        f"""
        SELECT gc_skew, strand
        FROM {cache_expr}
        WHERE gc_skew IS NOT NULL AND isFinite(gc_skew)
        """
    ).to_pandas()

    if data_4.empty:
        _out = mo.md("### CPU-4 · GC skew\n\nNo valid `gc_skew` values are available.")
    else:
        # strand is '' when missing; treat blanks as absent
        color_column_4 = "strand" if (data_4["strand"].astype(str) != "").any() else None
        fig_4 = px.histogram(
            data_4,
            x="gc_skew",
            color=color_column_4,
            nbins=40,
            marginal="box",
            barmode="overlay",
            opacity=0.70,
            title="CPU-4 — GC skew distribution",
            labels={"gc_skew": "GC skew"},
            template="plotly_white",
        )
        _out = mo.vstack(
            [
                mo.md(
                    """
                    ### CPU-4 · GC skew by strand
                    GC skew is (G − C) / (G + C): positive means G-rich,
                    negative means C-rich. The strands are overlaid. If skew
                    was computed on the forward reference sequence, the two
                    strands are expected to be offset in opposite directions,
                    so check how the enrichment step handles minus-strand
                    intervals before interpreting the difference.
                    """
                ),
                mo.ui.plotly(fig_4),
            ]
        )

    _out
    return


@app.cell
def _(cache_expr, mo, px, resource):
    data_5 = resource.query_arrow(
        f"""
        SELECT
            if(is_coding_region IS NULL, 'unknown', if(is_coding_region = 1, 'coding', 'noncoding')) AS coding_status,
            count() AS intervals
        FROM {cache_expr}
        GROUP BY coding_status
        ORDER BY coding_status
        """
    ).to_pandas()

    if data_5.empty:
        _out = mo.md("### CPU-5 · Coding composition\n\nNo data available.")
    else:
        data_5["percentage"] = data_5["intervals"] / data_5["intervals"].sum() * 100
        fig_5 = px.bar(
            data_5,
            x="coding_status",
            y="percentage",
            text="percentage",
            title="CPU-5 — Coding versus noncoding composition",
            labels={"coding_status": "Coding status", "percentage": "Intervals (%)"},
            template="plotly_white",
        )
        fig_5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _out = mo.vstack(
            [
                mo.md(
                    """
                    ### CPU-5 · Coding composition
                    Share of intervals by coding status. `unknown` means the
                    dataset carries no coding annotation for that interval.
                    """
                ),
                mo.ui.plotly(fig_5),
                mo.ui.table(data_5),
            ]
        )

    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Record-level profile
    """)
    return


@app.cell
def _(cache_expr, go, mo, np, pd, resource):
    profile = resource.query_arrow(
        f"""
        SELECT
            record_id,
            count() AS interval_count,
            avg(gc_content) AS gc_content,
            avg(sequence_length) AS sequence_length,
            avgOrNull(gc_skew) AS gc_skew,
            avgOrNull(toFloat64(is_coding_region)) AS coding_proportion
        FROM {cache_expr}
        GROUP BY record_id
        ORDER BY record_id
        LIMIT 100
        """
    ).to_pandas()

    feature_columns = [
        column
        for column in ["gc_content", "sequence_length", "gc_skew", "coding_proportion", "interval_count"]
        if column in profile.columns
    ]

    if len(profile) < 2 or not feature_columns:
        _out = mo.md("### CPU-6 · Record profile\n\nInsufficient record-level data for a heatmap.")
    else:
        matrix = profile.set_index("record_id")[feature_columns]
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        matrix = matrix.fillna(matrix.median(numeric_only=True))
        standardized = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
        standardized = standardized.replace([np.inf, -np.inf], 0).fillna(0)

        fig_6 = go.Figure(
            data=go.Heatmap(
                z=standardized.T.values,
                x=[str(index) for index in standardized.index],
                y=list(standardized.columns),
                colorbar={"title": "z-score"},
                hoverongaps=False,
            )
        )
        fig_6.update_layout(
            title="CPU-6 — Record-level biological profile heatmap",
            xaxis_title="Record ID (first 100 records)",
            yaxis_title="Standardized feature",
            template="plotly_white",
            height=550,
        )
        _out = mo.vstack(
            [
                mo.md(
                    f"""
                    ### CPU-6 · Record-level profile heatmap
                    Interval values are averaged per record, then each
                    feature is z-scored across the displayed records (0 =
                    average record). **Read with care:**

                    - Only the first **{len(standardized):,}** records in
                      alphabetical ID order are shown, so this is a slice, not
                      a random sample.
                    - Records with few intervals have noisy averages (see
                      CPU-8).
                    - One extreme record can compress the color scale and make
                      other rows look flat.
                    """
                ),
                mo.ui.plotly(fig_6),
            ]
        )

    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Cross-feature views

    The plots above look at one feature at a time. These three test whether
    apparent patterns are real or artifacts:

    - **CPU-7:** are the long GC-skew tails just short, noisy intervals?
    - **CPU-8:** are extreme record-level averages just records with few intervals?
    - **CPU-9:** do features shift between groups of accessions, which would point to assembly or batch effects rather than biology?
    """)
    return


@app.cell
def _(cache_expr, mo, np, px, resource):
    # Sampled rows for the density view.
    data_7 = resource.query_arrow(
        f"""
        SELECT
            if(strand = '', 'unknown', replaceRegexpAll(strand, '[<>]', '')) AS strand_label,
            sequence_length,
            gc_skew
        FROM {cache_expr}
        WHERE sequence_length > 0
          AND gc_skew IS NOT NULL AND isFinite(gc_skew)
        ORDER BY sipHash64(record_id, start, end)
        LIMIT 100000
        """
    ).to_pandas()

    # Full-cohort aggregate: mean |skew| per log10-length bin (0.25 wide) and strand.
    bins_7 = resource.query_arrow(
        f"""
        SELECT
            if(strand = '', 'unknown', replaceRegexpAll(strand, '[<>]', '')) AS strand_label,
            floor(log10(sequence_length) * 4) / 4 AS log10_bin_start,
            count() AS intervals,
            avg(abs(gc_skew)) AS mean_abs_gc_skew
        FROM {cache_expr}
        WHERE sequence_length > 0
          AND gc_skew IS NOT NULL AND isFinite(gc_skew)
        GROUP BY strand_label, log10_bin_start
        HAVING intervals >= 30
        ORDER BY strand_label, log10_bin_start
        """
    ).to_pandas()

    if data_7.empty or bins_7.empty:
        _out = mo.md("### CPU-7 · GC skew vs length\n\nNot enough valid `gc_skew` data.")
    else:
        data_7["log10_sequence_length"] = np.log10(data_7["sequence_length"])
        fig_7a = px.density_heatmap(
            data_7,
            x="log10_sequence_length",
            y="gc_skew",
            facet_col="strand_label",
            nbinsx=40,
            nbinsy=40,
            color_continuous_scale="Viridis",
            labels={
                "log10_sequence_length": "log10(sequence length)",
                "gc_skew": "GC skew",
                "strand_label": "strand",
            },
            title="CPU-7a — GC skew versus interval length (density, by strand)",
            template="plotly_white",
        )
        fig_7a.for_each_annotation(lambda a: a.update(text="strand " + a.text.split("=")[-1]))

        bins_7["log10_bin_center"] = bins_7["log10_bin_start"] + 0.125
        fig_7b = px.line(
            bins_7,
            x="log10_bin_center",
            y="mean_abs_gc_skew",
            color="strand_label",
            markers=True,
            hover_data=["intervals"],
            labels={
                "log10_bin_center": "log10(sequence length), bin center",
                "mean_abs_gc_skew": "Mean |GC skew|",
                "strand_label": "strand",
            },
            title="CPU-7b — Mean absolute GC skew by length bin",
            template="plotly_white",
        )

        _out = mo.vstack(
            [
                mo.md(
                    """
                    ### CPU-7 · GC skew vs interval length
                    **7a** is a density view of a 100,000-interval sample:
                    brighter cells hold more intervals. **7b** uses *all*
                    intervals, averaging |GC skew| in length bins of 0.25
                    log10 units (bins with fewer than 30 intervals are
                    dropped).

                    **What to look for:** if mean |skew| falls as length
                    increases and the wide tails come from the shortest bins,
                    the extremes are mostly small-sample noise. If it stays
                    flat, the skew has a real length-independent component.
                    """
                ),
                fig_7a,
                fig_7b,
            ]
        )

    _out
    return


@app.cell
def _(cache_expr, mo, np, px, resource):
    data_8 = resource.query_arrow(
        f"""
        SELECT
            record_id,
            substring(record_id, 1, 7) AS accession_group,
            count() AS interval_count,
            sum(sequence_length) AS total_interval_length,
            avg(gc_content) AS mean_gc_content,
            avgIf(gc_skew, ifNull(isFinite(gc_skew), 0)) AS mean_gc_skew
        FROM {cache_expr}
        WHERE isFinite(gc_content)
        GROUP BY record_id
        """
    ).to_pandas()

    cohort_8 = resource.query_arrow(
        f"""
        SELECT
            avg(gc_content) AS mean_gc_content,
            avgIf(gc_skew, ifNull(isFinite(gc_skew), 0)) AS mean_gc_skew
        FROM {cache_expr}
        WHERE isFinite(gc_content)
        """
    ).to_pandas()

    if len(data_8) < 2:
        _out = mo.md("### CPU-8 · Record-level averages vs interval count\n\nNot enough records.")
    else:
        _top = data_8["accession_group"].value_counts().head(8).index
        data_8["accession_group_display"] = np.where(
            data_8["accession_group"].isin(_top), data_8["accession_group"], "other"
        )

        _panels = []
        for _col, _label, _title in [
            ("mean_gc_skew", "Mean GC skew", "CPU-8a — Mean GC skew per record"),
            ("mean_gc_content", "Mean GC content (fraction)", "CPU-8b — Mean GC content per record"),
        ]:
            _fig = px.scatter(
                data_8.dropna(subset=[_col]),
                x="interval_count",
                y=_col,
                size="total_interval_length",
                size_max=18,
                color="accession_group_display",
                log_x=True,
                opacity=0.7,
                hover_name="record_id",
                hover_data=["interval_count", "total_interval_length"],
                labels={
                    "interval_count": "Intervals per record (log scale)",
                    _col: _label,
                    "accession_group_display": "accession group",
                },
                title=_title,
                template="plotly_white",
                height=480,
            )
            _ref = cohort_8[_col].iloc[0]
            if _ref is not None and np.isfinite(_ref):
                _fig.add_hline(
                    y=float(_ref),
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="cohort mean (all intervals)",
                    annotation_position="top left",
                )
            _panels.append(_fig)

        _out = mo.vstack(
            [
                mo.md(
                    f"""
                    ### CPU-8 · Record-level averages vs number of intervals
                    One point per record (**{len(data_8):,}** records). Point
                    size is the record's total interval length; color is the
                    accession group (first 7 characters of `record_id`, top 8
                    groups, the rest shown as *other*); the dashed line is the
                    cohort mean over all intervals.

                    **What to look for:** small-sample noise produces a
                    *funnel*: wide scatter on the left (few intervals) that
                    narrows as interval count grows. Extreme averages that
                    persist at high interval counts are more likely to be
                    real. This is the check for the outlier columns in CPU-6.
                    """
                ),
                mo.hstack(_panels, widths="equal"),
            ]
        )

    _out
    return


@app.cell
def _(cache_expr, mo, np, px, resource):
    groups_9 = resource.query_arrow(
        f"""
        SELECT
            substring(record_id, 1, 7) AS accession_group,
            count() AS intervals,
            uniqExact(record_id) AS records,
            avg(gc_content) AS mean_gc_content,
            avgIf(gc_skew, ifNull(isFinite(gc_skew), 0)) AS mean_gc_skew,
            avg(sequence_length) AS mean_sequence_length
        FROM {cache_expr}
        WHERE isFinite(gc_content)
        GROUP BY accession_group
        ORDER BY intervals DESC
        LIMIT 10
        """
    ).to_pandas()

    if groups_9.empty:
        _out = mo.md("### CPU-9 · Features by accession group\n\nNo data available.")
    else:
        _order = groups_9["accession_group"].tolist()
        _in_list = ", ".join("'" + g.replace("'", "''") + "'" for g in _order)

        sample_9 = resource.query_arrow(
            f"""
            SELECT
                substring(record_id, 1, 7) AS accession_group,
                gc_content,
                gc_skew,
                sequence_length
            FROM {cache_expr}
            WHERE isFinite(gc_content)
              AND sequence_length > 0
              AND substring(record_id, 1, 7) IN ({_in_list})
            ORDER BY sipHash64(record_id, start, end)
            LIMIT 5000 BY accession_group
            """
        ).to_pandas()

        sample_9["log10_sequence_length"] = np.log10(sample_9["sequence_length"])
        _names = {
            "gc_content": "GC content",
            "gc_skew": "GC skew",
            "log10_sequence_length": "log10(length)",
        }
        long_9 = (
            sample_9.melt(
                id_vars="accession_group",
                value_vars=list(_names),
                var_name="feature",
                value_name="value",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["value"])
        )
        long_9["feature"] = long_9["feature"].map(_names)

        fig_9 = px.box(
            long_9,
            x="accession_group",
            y="value",
            color="accession_group",
            facet_row="feature",
            category_orders={
                "accession_group": _order,
                "feature": list(_names.values()),
            },
            points=False,
            labels={"accession_group": "Accession group", "value": ""},
            title="CPU-9 — Feature distributions by accession group",
            template="plotly_white",
            height=820,
        )
        fig_9.update_yaxes(matches=None)
        fig_9.update_layout(showlegend=False)
        fig_9.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        _out = mo.vstack(
            [
                mo.md(
                    """
                    ### CPU-9 · Features by accession group
                    Records are grouped by the first 7 characters of
                    `record_id` (for example `NC_0877` or `NW_0173`). This is
                    a proxy for accession batches: records deposited together
                    often come from the same assembly or project, but the
                    prefix is not an assembly ID. The 10 largest groups are
                    shown, each subsampled to at most 5,000 intervals for the
                    box plots. The table below uses **all** intervals.

                    **What to look for:** groups whose medians or spreads
                    differ clearly. That points to assembly, annotation or
                    batch effects that could be mistaken for biology in the
                    cohort-wide plots.
                    """
                ),
                fig_9,
                mo.ui.table(groups_9),
            ]
        )

    _out
    return


@app.cell
def _(cache_expr, mo, resource):
    quality_report = resource.query_arrow(
        f"""
        SELECT
            'sequence_length' AS feature,
            countIf(sequence_length IS NULL) AS null_count,
            toFloat64(min(sequence_length)) AS minimum,
            toFloat64(max(sequence_length)) AS maximum
        FROM {cache_expr}
        UNION ALL
        SELECT
            'gc_content' AS feature,
            countIf(gc_content IS NULL) AS null_count,
            toFloat64(min(gc_content)) AS minimum,
            toFloat64(max(gc_content)) AS maximum
        FROM {cache_expr}
        """
    ).to_pandas()

    mo.vstack(
        [
            mo.md(
                """
                ## 7 · Data-quality checks

                Null counts and value ranges for the two required numeric
                features. `gc_content` should lie within 0 to 1 (values above
                1 are rescaled from percent at ingestion) and
                `sequence_length` should be positive.
                """
            ),
            mo.ui.table(quality_report),
        ]
    )
    return


@app.cell
def _(mo, target_lineage):
    mo.md(f"""
    ## Interpretation boundary

    These figures describe measurable properties of the CPU-enriched
    cohort **{target_lineage.value.strip()}**: nucleotide composition,
    interval length, GC asymmetry (when available), coding-state
    composition, and record-level heterogeneity.

    They do not establish that Carbon learned a biological concept.
    GPU-derived likelihoods, embeddings, and nearest-neighbor relationships
    should be introduced only through a validated shared CPU–GPU cohort.
    """)
    return


if __name__ == "__main__":
    app.run()
