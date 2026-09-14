import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


# ============================================================================
# 01. Imports & configuration
# ============================================================================


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from ncbi_client import NCBIClient, NCBIError
    from taxonomy_index import cohort_for_rank, load_taxonomy_index

    return (
        NCBIClient,
        NCBIError,
        cohort_for_rank,
        load_taxonomy_index,
        mo,
        np,
        pd,
    )


@app.cell
def _():
    # Centralize paths/config here. Do not scatter literal paths through
    # the rest of the notebook.
    DATA_ROOT = "/teamspace/studios/this_studio/hf-genomic-dataset-enrichment"

    EMBEDDING_PATH = f"{DATA_ROOT}/embedding.parquet"
    CPU_PATH = f"{DATA_ROOT}/cpu_enrichment.parquet"
    LIKELIHOOD_PATH = f"{DATA_ROOT}/likelihood.parquet"
    TAXONOMY_INDEX_PATH = f"{DATA_ROOT}/taxonomy_index.parquet"

    COHORT_RANK_OPTIONS = {
        "All common-cohort records": None,
        "Same domain": "domain_name",
        "Same kingdom": "kingdom_name",
        "Same phylum": "phylum_name",
        "Same class": "class_name",
        "Same order": "order_name",
        "Same family": "family_name",
        "Same genus": "genus_name",
        "Same species": "species_name",
    }

    return (
        CPU_PATH,
        COHORT_RANK_OPTIONS,
        DATA_ROOT,
        EMBEDDING_PATH,
        LIKELIHOOD_PATH,
        TAXONOMY_INDEX_PATH,
    )


# ============================================================================
# 02. Data connections
# ============================================================================
#
# Load each dataset once. This notebook does not care how the data is
# stored underneath (parquet, DuckDB, ClickHouse) — it only needs
# record-keyed lookups. Replace the bodies of the get_*_record()
# helpers with your existing DuckDB/ClickHouse access if that's faster
# than filtering an in-memory DataFrame for your dataset sizes; the
# rest of the notebook only calls these functions, never the
# underlying tables directly.


@app.cell
def _(EMBEDDING_PATH, CPU_PATH, LIKELIHOOD_PATH, pd):
    # TODO: replace with the real loaders for your three datasets.
    # Kept as plain parquet reads here so the notebook runs standalone;
    # swap for your DuckDB/ClickHouse-backed access pattern as needed.
    embedding_df = pd.read_parquet(EMBEDDING_PATH)
    cpu_df = pd.read_parquet(CPU_PATH)
    likelihood_df = pd.read_parquet(LIKELIHOOD_PATH)

    return cpu_df, embedding_df, likelihood_df


@app.cell
def _(cpu_df, embedding_df, likelihood_df):
    # Common identity: don't assume every dataset covers every record.
    embedding_ids = set(embedding_df["record_id"])
    cpu_ids = set(cpu_df["record_id"])
    likelihood_ids = set(likelihood_df["record_id"])

    common_record_ids = sorted(embedding_ids & cpu_ids & likelihood_ids)

    return common_record_ids, cpu_ids, embedding_ids, likelihood_ids


@app.cell
def _(cpu_df, embedding_df, likelihood_df):
    def get_embedding_record(record_id: str):
        rows = embedding_df.loc[embedding_df["record_id"] == record_id]
        return rows.iloc[0] if not rows.empty else None

    def get_cpu_record(record_id: str):
        rows = cpu_df.loc[cpu_df["record_id"] == record_id]
        return rows.iloc[0] if not rows.empty else None

    def get_likelihood_record(record_id: str):
        rows = likelihood_df.loc[likelihood_df["record_id"] == record_id]
        return rows.iloc[0] if not rows.empty else None

    return get_cpu_record, get_embedding_record, get_likelihood_record


# ============================================================================
# 03. Record index
# ============================================================================


@app.cell
def _(common_record_ids, mo, pd):
    record_index = pd.DataFrame({"record_id": common_record_ids})

    mo.md(f"**Common cohort:** {len(common_record_ids):,} records")
    return (record_index,)


# ============================================================================
# 04. Taxonomy index (local, precomputed — see taxonomy_index.py)
# ============================================================================
#
# This is a pure local read. It must be built ahead of time via
# build_taxonomy_index() in taxonomy_index.py — the notebook never
# resolves the full cohort against NCBI live, only the one selected
# record (section 07 below).


@app.cell
def _(TAXONOMY_INDEX_PATH, load_taxonomy_index, mo):
    try:
        taxonomy_index_df = load_taxonomy_index(TAXONOMY_INDEX_PATH)
        taxonomy_index_error = None
    except FileNotFoundError as exc:
        taxonomy_index_df = None
        taxonomy_index_error = str(exc)

    taxonomy_index_warning = (
        mo.callout(
            f"Taxonomy index not found at {TAXONOMY_INDEX_PATH}. "
            "Run build_taxonomy_index() first — taxonomic cohort "
            "comparisons (section 14) will be unavailable until then.",
            kind="warn",
        )
        if taxonomy_index_error
        else mo.md("")
    )

    taxonomy_index_warning

    return taxonomy_index_df, taxonomy_index_error, taxonomy_index_warning


# ============================================================================
# 05. NCBI client
# ============================================================================
#
# Created once, independent of the selected record, so the client
# (and its in-memory cache) survives across re-selection. Do NOT put
# NCBIClient() in a cell that depends on selected_record_id, or Marimo
# will recreate it — and drop the cache — every time the user picks a
# new record.


@app.cell
def _(NCBIClient):
    ncbi = NCBIClient()
    return (ncbi,)


# ============================================================================
# 06. Specimen selector — the notebook's control center
# ============================================================================


@app.cell
def _(common_record_ids, mo):
    record_selector = mo.ui.dropdown(
        options=common_record_ids,
        value=common_record_ids[0] if common_record_ids else None,
        label="Record ID",
        searchable=True,
    )

    record_selector
    return (record_selector,)


@app.cell
def _(record_selector):
    selected_record_id = record_selector.value
    return (selected_record_id,)


# ============================================================================
# 07. Selected local records
# ============================================================================


@app.cell
def _(
    get_cpu_record,
    get_embedding_record,
    get_likelihood_record,
    selected_record_id,
):
    if selected_record_id is not None:
        selected_embedding = get_embedding_record(selected_record_id)
        selected_cpu = get_cpu_record(selected_record_id)
        selected_likelihood = get_likelihood_record(selected_record_id)
    else:
        selected_embedding = None
        selected_cpu = None
        selected_likelihood = None

    return selected_cpu, selected_embedding, selected_likelihood


# ============================================================================
# 08. NCBI enrichment
# ============================================================================
#
# NCBI resolution never blocks the rest of the explorer. A failure
# here is surfaced with its stage-specific message (the client already
# annotates which stage failed) and the local sections keep working.


@app.cell
def _(NCBIError, mo, ncbi, selected_record_id):
    specimen_ncbi = None
    ncbi_error = None

    if selected_record_id is not None:
        try:
            specimen_ncbi = ncbi.resolve_record(selected_record_id)
        except NCBIError as exc:
            ncbi_error = str(exc)

    ncbi_error_warning = (
        mo.callout(f"NCBI enrichment unavailable: {ncbi_error}", kind="warn")
        if ncbi_error
        else mo.md("")
    )

    ncbi_error_warning
    return ncbi_error, ncbi_error_warning, specimen_ncbi


# ============================================================================
# 09. Specimen overview
# ============================================================================


@app.cell
def _(mo, ncbi, specimen_ncbi):
    if specimen_ncbi is None:
        specimen_overview = mo.md("_NCBI context unavailable for this record._")
    else:
        lineage_str = (
            " → ".join(specimen_ncbi.lineage) if specimen_ncbi.lineage else "—"
        )

        # Image metadata only — bytes are fetched lazily in section 15.
        try:
            overview_image_metadata = ncbi.get_taxon_image_metadata(
                specimen_ncbi.tax_id
            )
        except Exception:
            overview_image_metadata = None

        overview_lines = [
            f"### {specimen_ncbi.scientific_name}",
            f"*{specimen_ncbi.common_name}*" if specimen_ncbi.common_name else "",
            "",
            f"| | |\n|---|---|\n"
            f"| TaxID | {specimen_ncbi.tax_id} |\n"
            f"| Rank | {specimen_ncbi.rank or '—'} |\n"
            f"| Record ID | {specimen_ncbi.record_id} |\n"
            f"| Assembly | {specimen_ncbi.assembly.accession} "
            f"({specimen_ncbi.assembly.assembly_name or '—'}, "
            f"{specimen_ncbi.assembly.assembly_level or '—'}) |",
            "",
            f"**Taxonomy:** {lineage_str}",
        ]

        if overview_image_metadata is not None and overview_image_metadata.attribution:
            overview_lines.append(
                f"\n_Image: {overview_image_metadata.attribution} · "
                f"{overview_image_metadata.license or 'license unknown'}_"
            )

        specimen_overview = mo.md("\n".join(overview_lines))

    specimen_overview
    return (specimen_overview,)


# ============================================================================
# 10. Sequence section
# ============================================================================


@app.cell
def _(mo, selected_embedding, specimen_ncbi):
    # Sequence length, GC%, etc. depend on fields your embedding/CPU
    # pipeline already computes per record — wire in the actual column
    # names from your dataset here.
    if selected_embedding is None:
        sequence_section = mo.md("_No local sequence data for this record._")
    else:
        sequence_section = mo.md(
            f"**Sequence accession:** "
            f"{specimen_ncbi.sequence.accession if specimen_ncbi else '—'}\n\n"
            "_TODO: pull sequence length / GC% / composition columns from "
            "your embedding or CPU-enrichment table and plot the selected "
            "record against the cohort distribution here._"
        )

    sequence_section
    return (sequence_section,)


# ============================================================================
# 11. Embedding section
# ============================================================================


@app.cell
def _(mo, selected_embedding):
    if selected_embedding is None:
        embedding_section = mo.md("_No embedding for this record._")
    else:
        embedding_section = mo.md(
            "_TODO: plot the embedding-space projection with the selected "
            "record highlighted, plus nearest-neighbor record ids. Once "
            "neighbors are found, look up their tax_id via the local "
            "taxonomy_index_df (not a live NCBI call) to show taxonomic "
            "context for each neighbor._"
        )

    embedding_section
    return (embedding_section,)


# ============================================================================
# 12. CPU enrichment section
# ============================================================================


@app.cell
def _(mo, selected_cpu):
    if selected_cpu is None:
        cpu_section = mo.md("_No CPU enrichment data for this record._")
    else:
        cpu_section = mo.md(
            "_TODO: show the selected record's CPU-enrichment metric(s) "
            "against the cohort distribution, with percentile/rank._"
        )

    cpu_section
    return (cpu_section,)


# ============================================================================
# 13. Likelihood section
# ============================================================================


@app.cell
def _(mo, selected_likelihood):
    if selected_likelihood is None:
        likelihood_section = mo.md("_No likelihood data for this record._")
    else:
        likelihood_section = mo.md(
            "_TODO: show the selected record's likelihood against the "
            "cohort distribution, with percentile/rank — same treatment "
            "as the CPU section above._"
        )

    likelihood_section
    return (likelihood_section,)


# ============================================================================
# 14. Cross-dataset profile
# ============================================================================


@app.cell
def _(mo, pd, selected_cpu, selected_embedding, selected_likelihood):
    # A compact joint summary rather than three disconnected analyses.
    # Fill in the real column names / percentile computations once the
    # sections above are wired to your actual data.
    profile_rows = []

    if selected_embedding is not None:
        profile_rows.append(
            {"Feature": "Embedding", "Selected": "—", "Cohort percentile": "—"}
        )
    if selected_cpu is not None:
        profile_rows.append(
            {"Feature": "CPU enrichment", "Selected": "—", "Cohort percentile": "—"}
        )
    if selected_likelihood is not None:
        profile_rows.append(
            {"Feature": "Likelihood", "Selected": "—", "Cohort percentile": "—"}
        )

    cross_dataset_profile = (
        mo.ui.table(pd.DataFrame(profile_rows))
        if profile_rows
        else mo.md("_No data available to build a cross-dataset profile._")
    )

    cross_dataset_profile
    return (cross_dataset_profile,)


# ============================================================================
# 15. Taxonomic cohort comparison
# ============================================================================
#
# Uses the local taxonomy_index_df exclusively — no NCBI calls here,
# regardless of cohort size, since this can run over thousands of
# records on every rank change.


@app.cell
def _(COHORT_RANK_OPTIONS, mo):
    cohort_rank_selector = mo.ui.dropdown(
        options=list(COHORT_RANK_OPTIONS.keys()),
        value="All common-cohort records",
        label="Compare against",
    )

    cohort_rank_selector
    return (cohort_rank_selector,)


@app.cell
def _(
    COHORT_RANK_OPTIONS,
    cohort_for_rank,
    cohort_rank_selector,
    mo,
    selected_record_id,
    taxonomy_index_df,
):
    if taxonomy_index_df is None or selected_record_id is None:
        taxonomic_cohort_section = mo.md(
            "_Taxonomic cohort comparison unavailable "
            "(missing taxonomy index or no record selected)._"
        )
        taxonomic_cohort_df = None
    else:
        rank_column = COHORT_RANK_OPTIONS[cohort_rank_selector.value]

        if rank_column is None:
            taxonomic_cohort_df = taxonomy_index_df
        else:
            try:
                taxonomic_cohort_df = cohort_for_rank(
                    taxonomy_index_df,
                    record_id=selected_record_id,
                    rank=rank_column,
                )
            except (KeyError, ValueError) as exc:
                taxonomic_cohort_df = None
                taxonomic_cohort_section = mo.md(f"_Could not build cohort: {exc}_")

        if taxonomic_cohort_df is not None:
            taxonomic_cohort_section = mo.md(
                f"**Cohort size:** {len(taxonomic_cohort_df):,} records "
                f"({cohort_rank_selector.value})\n\n"
                "_TODO: join this cohort's record_ids back into the "
                "embedding/CPU/likelihood tables to compute cohort-"
                "relative percentiles for the selected record, same "
                "pattern as sections 12–13 but scoped to this cohort._"
            )

    taxonomic_cohort_section
    return taxonomic_cohort_df, taxonomic_cohort_section


# ============================================================================
# 16. NCBI media / external links
# ============================================================================
#
# Lazy: image bytes are only fetched here, on demand, never as a side
# effect of selecting a record. get_taxon_image() is cached per tax_id
# in NCBIClient, so revisiting a same-species record won't re-download.


@app.cell
def _(mo):
    show_image_button = mo.ui.run_button(label="Load specimen image")
    show_image_button
    return (show_image_button,)


@app.cell
def _(mo, ncbi, show_image_button, specimen_ncbi):
    if not show_image_button.value or specimen_ncbi is None:
        media_section = mo.md("_Image not loaded._")
    else:
        try:
            image_bytes, image_metadata, links = ncbi.get_taxon_media(
                specimen_ncbi.tax_id,
                include_image=True,
            )

            link_items = "\n".join(
                f"- [{name}]({url})" for name, url in (links or {}).items()
            )

            media_section = mo.vstack(
                [
                    mo.image(image_bytes, width=240),
                    mo.md(
                        f"{image_metadata.attribution or 'Unknown attribution'} · "
                        f"{image_metadata.license or 'license unknown'}\n\n"
                        f"{link_items}"
                    ),
                ]
            )
        except Exception as exc:
            media_section = mo.md(f"_Could not load media: {exc}_")

    media_section
    return (media_section,)


if __name__ == "__main__":
    app.run()
