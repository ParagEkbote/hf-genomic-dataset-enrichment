import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import hashlib
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    return Path, hashlib, mo, np, pd, tqdm


@app.cell
def _(mo):
    mo.md(r"""
    # Mapping Biological Diversity into Embedding Space

    **Carbon embeddings — interactive representation analysis**

    This notebook treats the embedding as a **model-generated representation of genomic
    sequence**, not as a vector whose individual dimensions have direct biological meaning.

    The central question is:

    > **Does the model's representation space organize genomic sequences in ways that
    > correspond to biological sequence properties — including taxonomy (e.g. genus)?**

    We use streaming access to the Hugging Face dataset and materialize only the
    bounded sample required for interactive analysis. Progress through the stream is
    reported with `tqdm`.

    `AINovice2005/carbon-embeddings` itself carries no taxonomy — only
    `record_id`, `start`, `end`, `embedding`, `embedding_norm`. Any genus/taxon view
    below depends on joining this against a CPU-enrichment artifact that carries
    taxonomy, keyed on `(record_id, start, end)`. That join source is configurable
    in Section 3.
    """)
    return


@app.cell
def _(mo):
    dataset_id = "AINovice2005/carbon-embeddings"
    split = "train"

    # Known row count from the dataset card — used only to give tqdm a total so the
    # bar shows real progress/ETA instead of an indeterminate spinner. If the dataset
    # is regenerated with a different row count, update this value.
    known_total_rows = 251_427

    sample_size = mo.ui.slider(
        start=1000,
        stop=20000,
        value=8000,
        step=1000,
        label="Embedding sample size",
    )

    seed = mo.ui.number(
        start=0,
        stop=999999,
        value=42,
        step=1,
        label="Sampling seed",
    )

    mo.vstack(
        [
            mo.md("## 1. Analysis controls"),
            mo.hstack([sample_size, seed]),
            mo.md(
                f"""
                **Dataset:** `{dataset_id}`  
                **Split:** `{split}`  
                **Access mode:** Hugging Face streaming
                """
            ),
        ]
    )
    return dataset_id, known_total_rows, sample_size, seed, split


@app.cell
def _(dataset_id, split):
    from datasets import load_dataset

    stream = load_dataset(
        dataset_id,
        split=split,
        streaming=True,
    )
    return (stream,)


@app.cell
def _(hashlib, known_total_rows, np, pd, sample_size, seed, stream, tqdm):
    def stable_score(record_id, start, end, seed_value):
        key = f"{seed_value}|{record_id}|{start}|{end}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)

    # A deterministic bounded sample. The stream is scanned, but the full embedding
    # matrix is never materialized. tqdm reports scan progress against the dataset's
    # known row count, plus how many rows currently sit in the reservoir.
    scored = []
    seen = 0
    k = int(sample_size.value)
    seed_value = int(seed.value)

    progress = tqdm(
        stream,
        total=known_total_rows,
        desc="Scanning carbon-embeddings stream",
        unit="rows",
        dynamic_ncols=True,
    )

    for row in progress:
        seen += 1
        score = stable_score(
            row["record_id"],
            row["start"],
            row["end"],
            seed_value,
        )

        if len(scored) < k:
            scored.append((score, row))
            if len(scored) == k:
                scored.sort(key=lambda x: x[0], reverse=True)
        elif score < scored[0][0]:
            scored[0] = (score, row)
            scored.sort(key=lambda x: x[0], reverse=True)

        if seen % 2000 == 0:
            progress.set_postfix(reservoir=len(scored), refresh=False)

    progress.close()

    rows = [item[1] for item in sorted(scored, key=lambda x: x[0])]

    records = pd.DataFrame(
        {
            "record_id": [r["record_id"] for r in rows],
            "start": [r["start"] for r in rows],
            "end": [r["end"] for r in rows],
            "embedding": [r["embedding"] for r in rows],
            "embedding_norm": [r["embedding_norm"] for r in rows],
        }
    )

    matrix = np.asarray(records["embedding"].tolist(), dtype=np.float32)

    scan_summary = {
        "rows_seen": seen,
        "rows_sampled": len(records),
        "embedding_dim": matrix.shape[1] if matrix.ndim == 2 else 0,
    }
    return records, row, scan_summary


@app.cell
def _(mo, records, scan_summary):
    mo.vstack(
        [
            mo.md("## 2. Representation overview"),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{scan_summary['rows_seen']:,}",
                        label="Rows observed in stream",
                    ),
                    mo.stat(
                        value=f"{scan_summary['rows_sampled']:,}",
                        label="Rows materialized",
                    ),
                    mo.stat(
                        value=str(scan_summary["embedding_dim"]),
                        label="Embedding dimensions",
                    ),
                    mo.stat(
                        value=f"{records['embedding_norm'].median():.2f}",
                        label="Median embedding norm",
                    ),
                ],
                widths="equal",
            ),
        ]
    )
    return


@app.cell
def _(mo, pd, records):
    norm_summary = pd.DataFrame(
        {
            "statistic": [
                "minimum",
                "5th percentile",
                "median",
                "95th percentile",
                "maximum",
                "mean",
                "standard deviation",
            ],
            "embedding_norm": [
                records["embedding_norm"].min(),
                records["embedding_norm"].quantile(0.05),
                records["embedding_norm"].median(),
                records["embedding_norm"].quantile(0.95),
                records["embedding_norm"].max(),
                records["embedding_norm"].mean(),
                records["embedding_norm"].std(),
            ],
        }
    )

    mo.vstack(
        [
            mo.md(
                """
                ### Embedding magnitude

                `embedding_norm` is treated here as a **representation diagnostic**.
                It is not interpreted as a measure of biological quality, importance,
                or evolutionary significance.
                """
            ),
            mo.ui.table(norm_summary.round(3)),
        ]
    )
    return


@app.cell
def _(mo, records):
    import plotly.express as px

    fig = px.histogram(
        records,
        x="embedding_norm",
        nbins=60,
        title="Distribution of embedding norms",
        labels={"embedding_norm": "L2 embedding norm"},
    )
    fig.update_layout(height=430)

    mo.ui.plotly(fig)
    return


@app.cell
def _(np, records):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(
        np.asarray(records["embedding"].tolist(), dtype=np.float32)
    )

    projection = records[
        ["record_id", "start", "end", "embedding_norm"]
    ].copy()
    projection["PC1"] = coordinates[:, 0]
    projection["PC2"] = coordinates[:, 1]

    explained = pca.explained_variance_ratio_
    return explained, projection


@app.cell
def _(explained, mo, projection):
    import plotly.express as px

    fig = px.scatter(
        projection,
        x="PC1",
        y="PC2",
        color="embedding_norm",
        hover_data=["record_id", "start", "end"],
        color_continuous_scale="Viridis",
        title=(
            "PCA projection of Carbon sequence embeddings"
            f"<br><sup>PC1={explained[0]*100:.2f}% · "
            f"PC2={explained[1]*100:.2f}% variance explained</sup>"
        ),
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.65})
    fig.update_layout(height=650)

    mo.vstack(
        [
            mo.md(
                """
                ## 3. Embedding space

                PCA gives a reproducible first view of large-scale structure in the
                representation space. At this stage the colour encodes **embedding
                magnitude**, not taxonomy.
                """
            ),
            mo.ui.plotly(fig),
        ]
    )
    return


@app.cell
def _(mo):
    length_mode = mo.ui.dropdown(
        options={
            "Raw sequence length": "string_length",
            "Log10 sequence length": "log_length",
        },
        value="Raw sequence length",
        label="Colour variable",
    )
    return (length_mode,)


@app.cell
def _(length_mode, mo, np, projection):
    import plotly.express as px

    plot_data = projection.copy()

    if length_mode.value == "log_length":
        plot_data["log_length"] = np.log10(plot_data["string_length"].clip(lower=1))
        colour_column = "log_length"
        colour_label = "log10(sequence length)"
    else:
        colour_column = "string_length"
        colour_label = "sequence length"

    fig = px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color=colour_column,
        hover_data=["record_id", "string_length", "embedding_norm"],
        color_continuous_scale="Plasma",
        title="Embedding space coloured by sequence length",
        labels={colour_column: colour_label},
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.65})
    fig.update_layout(height=650)

    mo.vstack(
        [
            mo.md("### Does representation structure track sequence length?"),
            length_mode,
            mo.ui.plotly(fig),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Taxonomic / genus-level annotation

    `carbon-embeddings` has no taxonomy of its own. This section joins the
    materialized sample against a separate enrichment source — keyed on
    `record_id`, `start`, `end` — that does carry taxonomy (e.g. the CPU-enriched
    Carbon artifact / Phase 4.5 `record_metrics.csv` output). Point it at whichever
    source you're using; if the dataset or column name doesn't match, the notebook
    reports that clearly instead of failing silently.
    """)
    return


@app.cell
def _(mo):
    taxonomy_source_kind = mo.ui.dropdown(
        options=["Hugging Face dataset", "Local CSV/Parquet path"],
        value="Hugging Face dataset",
        label="Taxonomy source type",
    )

    taxonomy_source_id = mo.ui.text(
        value="AINovice2005/carbon-pilot-corpus-dedup",
        label="Taxonomy source (HF dataset id or local file path)",
        full_width=True,
    )

    taxonomy_genus_column = mo.ui.text(
        value="taxon_group",
        label="Column holding genus/taxon (comma-separated fallbacks allowed)",
        full_width=True,
    )

    load_taxonomy_button = mo.ui.run_button(label="Load & join taxonomy")

    mo.vstack(
        [
            mo.hstack([taxonomy_source_kind, taxonomy_source_id]),
            taxonomy_genus_column,
            load_taxonomy_button,
        ]
    )
    return (
        load_taxonomy_button,
        taxonomy_genus_column,
        taxonomy_source_id,
        taxonomy_source_kind,
    )


@app.cell
def _(
    Path,
    load_taxonomy_button,
    mo,
    pd,
    records,
    taxonomy_genus_column,
    taxonomy_source_id,
    taxonomy_source_kind,
    tqdm,
):
    taxonomy_error = None
    taxonomy_df = None
    genus_column_used = None

    mo.stop(not load_taxonomy_button.value, mo.md("_Click **Load & join taxonomy** to fetch taxonomy and enable genus-level plots below._"))

    candidate_columns = [
        c.strip() for c in taxonomy_genus_column.value.split(",") if c.strip()
    ]

    try:
        if taxonomy_source_kind.value == "Local CSV/Parquet path":
            path = Path(taxonomy_source_id.value)
            if not path.exists():
                raise FileNotFoundError(f"No such file: {path}")
            taxonomy_df = (
                pd.read_parquet(path)
                if path.suffix in (".parquet", ".pq")
                else pd.read_csv(path)
            )
        else:
            from datasets import load_dataset as _load_dataset

            wanted_ids = set(records["record_id"].unique().tolist())
            tax_stream = _load_dataset(
                taxonomy_source_id.value, split="train", streaming=True
            )

            hits = []
            for tax_row in tqdm(
                tax_stream,
                desc=f"Scanning {taxonomy_source_id.value} for matching record_ids",
                unit="rows",
                dynamic_ncols=True,
            ):
                if tax_row.get("record_id") in wanted_ids:
                    hits.append(tax_row)
                    if len(hits) >= len(wanted_ids):
                        break
            taxonomy_df = pd.DataFrame(hits)

        available = [c for c in candidate_columns if c in taxonomy_df.columns]
        if not available:
            raise KeyError(
                f"None of the candidate genus columns {candidate_columns} were found. "
                f"Available columns: {list(taxonomy_df.columns)}"
            )
        genus_column_used = available[0]

    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        taxonomy_error = str(exc)
    return genus_column_used, taxonomy_df, taxonomy_error


@app.cell
def _(genus_column_used, mo, records, taxonomy_df, taxonomy_error):
    enriched = None

    if taxonomy_error is not None:
        mo.callout(
            mo.md(
                f"""
                **Could not load/join taxonomy.**

                {taxonomy_error}

                The rest of the notebook still works on `embedding_norm` /
                `sequence length` views above — genus-coloured plots below will be
                skipped until this join succeeds.
                """
            ),
            kind="warn",
        )
    elif taxonomy_df is not None and genus_column_used is not None:
        join_cols = [c for c in ("record_id", "start", "end") if c in taxonomy_df.columns]
        if "record_id" not in join_cols:
            mo.callout(
                mo.md("Taxonomy source has no `record_id` column to join on."),
                kind="warn",
            )
        else:
            keep_cols = join_cols + [genus_column_used]
            enriched = records.merge(
                taxonomy_df[keep_cols].drop_duplicates(subset=join_cols),
                on=join_cols,
                how="left",
            ).rename(columns={genus_column_used: "genus"})

            matched = enriched["genus"].notna().sum()
            mo.callout(
                mo.md(
                    f"Joined taxonomy on `{', '.join(join_cols)}` using column "
                    f"`{genus_column_used}` → **{matched:,} / {len(enriched):,}** "
                    f"sampled rows matched a genus/taxon label."
                ),
                kind="success" if matched > 0 else "warn",
            )
    return (enriched,)


@app.cell
def _(enriched, mo):
    genus_filter = None

    if enriched is not None and enriched["genus"].notna().any():
        genus_counts = enriched["genus"].value_counts()
        top_genera = genus_counts.head(30).index.tolist()

        genus_filter = mo.ui.multiselect(
            options=top_genera,
            value=top_genera[: min(8, len(top_genera))],
            label="Genus / taxon to display (top 30 by sample count)",
        )

        mo.vstack(
            [
                mo.md("### 4a. Choose genera to plot"),
                mo.md(
                    f"{enriched['genus'].nunique():,} distinct genus/taxon labels "
                    f"found in the sample."
                ),
                genus_filter,
            ]
        )
    else:
        mo.md(
            "_No genus data joined yet — load a taxonomy source above to unlock "
            "genus-level plots._"
        )
    return (genus_filter,)


@app.cell
def _(enriched, genus_filter, mo):
    import plotly.express as px

    mo.stop(
        enriched is None or genus_filter is None or not genus_filter.value,
        mo.md(""),
    )

    genus_subset = enriched[enriched["genus"].isin(genus_filter.value)].copy()

    fig = px.scatter(
        genus_subset,
        x="PC1" if "PC1" in genus_subset.columns else "embedding_norm",
        y="PC2" if "PC2" in genus_subset.columns else "string_length",
        color="genus",
        hover_data=["record_id", "start", "end", "string_length", "embedding_norm"],
        title="Embedding space coloured by genus/taxon",
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.7})
    fig.update_layout(height=650)

    mo.vstack(
        [
            mo.md("### 4b. Embedding space by genus"),
            mo.md(
                "If `PC1`/`PC2` aren't present on the joined frame, re-run this cell "
                "after Section 3 has computed the PCA projection, or merge the "
                "projection dataframe into the taxonomy join above."
            ),
            mo.ui.plotly(fig),
        ]
    )
    return


@app.cell
def _(enriched, genus_filter, mo):
    import plotly.express as px

    mo.stop(
        enriched is None or genus_filter is None or not genus_filter.value,
        mo.md(""),
    )

    genus_subset = enriched[enriched["genus"].isin(genus_filter.value)].copy()

    fig = px.violin(
        genus_subset,
        x="genus",
        y="embedding_norm",
        box=True,
        points="outliers",
        title="Embedding norm distribution per genus/taxon",
    )
    fig.update_layout(height=500, xaxis_tickangle=-30)

    fig2 = px.violin(
        genus_subset,
        x="genus",
        y="string_length",
        box=True,
        points="outliers",
        title="Sequence length distribution per genus/taxon",
    )
    fig2.update_layout(height=500, xaxis_tickangle=-30)

    mo.vstack(
        [
            mo.md("### 4c. Do genera differ in representation magnitude or sequence length?"),
            mo.ui.plotly(fig),
            mo.ui.plotly(fig2),
        ]
    )
    return


@app.cell
def _(mo, np, pd, records):
    # Pairwise cosine similarity is computed only on a bounded subset.
    # This avoids constructing an O(N²) matrix for the complete dataset.
    pair_sample = min(1500, len(records))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(records), size=pair_sample, replace=False)

    x = np.asarray(records.iloc[indices]["embedding"].tolist(), dtype=np.float32)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_unit = x / np.maximum(x_norm, 1e-12)

    # Sample random pairs rather than materializing the full similarity matrix.
    n_pairs = min(10000, pair_sample * 10)
    i = rng.integers(0, pair_sample, size=n_pairs)
    j = rng.integers(0, pair_sample, size=n_pairs)

    similarities = np.sum(x_unit[i] * x_unit[j], axis=1)

    similarity_summary = pd.DataFrame(
        {
            "statistic": ["minimum", "5th percentile", "median", "95th percentile", "maximum"],
            "cosine_similarity": [
                similarities.min(),
                np.quantile(similarities, 0.05),
                np.median(similarities),
                np.quantile(similarities, 0.95),
                similarities.max(),
            ],
        }
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 5. Representation similarity

                Cosine similarity is a natural diagnostic for dense vector
                representations. Here it is estimated from random pairs in a bounded
                sample rather than by constructing the full pairwise matrix.
                """
            ),
            mo.ui.table(similarity_summary.round(4)),
        ]
    )
    return (similarities,)


@app.cell
def _(mo, similarities):
    import plotly.express as px

    fig = px.histogram(
        x=similarities,
        nbins=70,
        title="Random-pair cosine similarity",
        labels={"x": "Cosine similarity"},
    )
    fig.update_layout(height=430)

    mo.ui.plotly(fig)
    return


@app.cell
def _(enriched, mo):
    genus_neighbor_query = None

    if enriched is not None and enriched["genus"].notna().any():
        genus_neighbor_query = mo.ui.dropdown(
            options=sorted(enriched["genus"].dropna().unique().tolist()),
            label="Genus to inspect nearest-neighbour taxon mixing for",
        )
        mo.vstack(
            [
                mo.md(
                    """
                    ## 6. Does embedding similarity track genus?

                    For each sampled record belonging to the selected genus, find its
                    nearest neighbour (by cosine similarity) within the sample and
                    check whether that neighbour shares the same genus label. A high
                    same-genus neighbour rate is evidence the representation space
                    organizes sequences by taxonomy at genus depth; a rate near the
                    background genus frequency suggests it doesn't, at least not at
                    this resolution.
                    """
                ),
                genus_neighbor_query,
            ]
        )
    else:
        mo.md(
            "_Load taxonomy above to compute genus-level nearest-neighbour "
            "agreement._"
        )
    return (genus_neighbor_query,)


@app.cell
def _(enriched, genus_neighbor_query, mo, np, pd):
    mo.stop(
        enriched is None or genus_neighbor_query is None or not genus_neighbor_query.value,
        mo.md(""),
    )

    valid = enriched.dropna(subset=["genus"]).reset_index(drop=True)
    matrix = np.asarray(valid["embedding"].tolist(), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-12)

    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    nearest_idx = np.argmax(sims, axis=1)

    valid["nearest_genus"] = valid.loc[nearest_idx, "genus"].values
    valid["same_genus_neighbor"] = valid["genus"] == valid["nearest_genus"]

    target = genus_neighbor_query.value
    target_rows = valid[valid["genus"] == target]

    background_rate = (valid["genus"].value_counts(normalize=True)).get(target, 0.0)
    observed_rate = target_rows["same_genus_neighbor"].mean() if len(target_rows) else float("nan")

    summary = pd.DataFrame(
        {
            "metric": [
                "records with this genus in sample",
                "observed same-genus nearest-neighbour rate",
                "background frequency of this genus in sample",
            ],
            "value": [
                len(target_rows),
                f"{observed_rate:.3f}" if observed_rate == observed_rate else "n/a",
                f"{background_rate:.3f}",
            ],
        }
    )

    mo.vstack(
        [
            mo.md(f"### Genus `{target}` — nearest-neighbour taxon agreement"),
            mo.ui.table(summary),
            mo.md(
                "Observed rate well above the background frequency suggests the "
                "embedding space clusters this genus together; a rate near "
                "background suggests it doesn't separate cleanly from others at "
                "genus depth in this sample."
            ),
        ]
    )
    return


@app.cell
def _(mo, np, records):
    selected_index = mo.ui.slider(
        start=0,
        stop=max(0, len(records) - 1),
        value=0,
        step=1,
        label="Select sampled embedding",
    )

    row = records.iloc[int(selected_index.value)]
    vector = np.asarray(row["embedding"], dtype=np.float32)
    return row, selected_index, vector


@app.cell
def _(np, records, row, vector):
    # Retrieve nearest neighbours within the currently materialized sample.
    x = np.asarray(records["embedding"].tolist(), dtype=np.float32)
    x_norm = np.linalg.norm(x, axis=1)
    v_norm = np.linalg.norm(vector)

    similarities = (x @ vector) / np.maximum(x_norm * v_norm, 1e-12)
    similarities[int(row.name)] = -np.inf

    k = min(10, len(records) - 1)
    nearest = np.argpartition(-similarities, k)[:k]
    nearest = nearest[np.argsort(-similarities[nearest])]

    neighbors = records.iloc[nearest][
        ["record_id", "start", "end", "string_length", "embedding_norm"]
    ].copy()
    neighbors.insert(0, "rank", range(1, len(neighbors) + 1))
    neighbors["cosine_similarity"] = similarities[nearest]
    return neighbors, similarities


@app.cell
def _(mo, neighbors, row, selected_index):
    mo.vstack(
        [
            mo.md("## 7. Sequence-level nearest neighbours"),
            selected_index,
            mo.md(
                f"""
                **Selected record:** `{row["record_id"]}`  
                **Corpus span:** `{int(row["start"]):,}–{int(row["end"]):,}`  
                **Sequence/string length:** `{int(row["string_length"]):,}`  
                **Embedding norm:** `{float(row["embedding_norm"]):.3f}`
                """
            ),
            mo.ui.table(neighbors.round(4)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Biological interpretation

    The embedding space can be used to ask whether model-derived representations
    contain structure associated with biological sequence properties. With taxonomy
    joined in (Section 4), that question can be tested directly rather than only
    inferred from `embedding_norm` and sequence length.

    Further enrichment (via `record_id`, `start`, `end`) would add:

    - GC content
    - GC skew
    - sequence entropy
    - gene / sequence type
    - strand and topology

    The resulting analysis can then test questions such as:

    **Do nearby embeddings preferentially share taxonomy?** — partially answered in
    Section 6 above, at genus depth, for the genus you selected.

    **Does embedding similarity increase with biological similarity?**

    **At which taxonomic depth does representation-space organization become visible?**

    **Do particular regions of embedding space correspond to unusual sequence
    composition?**

    These are empirical questions. Embedding proximity should not be interpreted
    as proof of homology or evolutionary relatedness without independent validation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### Data provenance

    **Source:** `AINovice2005/carbon-embeddings`
    **Split:** `train`
    **Access:** Hugging Face streaming dataset
    **Materialization policy:** bounded deterministic sample only

    The published dataset contains 251,427 records and is approximately 1.24 GB.
    Its central fields are `record_id`, `string_lengths`, `start`, `end`,
    `embedding`, and `embedding_norm`.

    `start` and `end` refer to positions in the processed/tokenized Carbon corpus;
    they are **not genomic coordinates**.

    Genus/taxon fields shown in Section 4 onward come from a separately configured
    join source (default: `AINovice2005/carbon-pilot-corpus-dedup`), not from
    `carbon-embeddings` itself.
    """)
    return


if __name__ == "__main__":
    app.run()
