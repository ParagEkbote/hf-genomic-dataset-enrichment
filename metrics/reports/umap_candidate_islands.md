# Candidate island assessment: raw versus whitened UMAP

## Scope and conclusion

This report examines only `plot02_umap_raw_vs_whitened.html` and its generation code, run metadata, and directly supporting artifacts. No other visualization HTML was opened or analyzed. The HTML contains two 20,000-point traces with identical interval keys. The chart is reproducible as a source of point coordinates and interval IDs, but the coordinates are compactly encoded as Plotly typed arrays and require decoding the `bdata` values.

**Conclusion:** the plot alone does not support a defensible named candidate island. It shows a two-dimensional projection of a sampled cohort under two separately fitted transformations. The artifact set does not retain the 20,000-row projection parquet or original embedding matrix. I therefore select no candidate and do not claim biological separation. Confidence is high that the plot's contents and metadata are identified; confidence in any biological island interpretation is low.

## Methods and reconstruction

The generator is `carbon-enrichment/carbon_enrichment/results/visualization/phase4.py`. It samples metadata with `sample(n=20,000, random_state=42)`, keeps only sampled keys for which an embedding is found, preserves seeded sample order, and filters rows with non-finite embeddings. Run metadata reports 20,000 projection rows from 244,046 metadata rows and 251,427 streamed embedding rows, with 3,072 embedding dimensions. No evidence in the run metadata suggests a missing sampled vector; the HTML traces each contain 20,000 points.

For each panel independently, the code fits randomized PCA with 50 components and `random_state=42`. Raw scores are passed directly to a cosine-metric UMAP; whitened scores come from a second PCA fit with `whiten=True`, then a separately fitted cosine-metric UMAP. Both UMAPs use 2 dimensions, `n_neighbors=25`, `min_dist=0.1`, `random_state=42`, and `n_jobs=1`. Whitening rescales retained principal component scores; it does not whiten the original dimensions before PCA. Since PCA and UMAP are separately fitted in each branch, coordinates and absolute distances across panels are not directly aligned.

Each trace carries `customdata` with `(record_id, start, end)` and `text` with a kingdom label. The hover template displays taxonomy label and x/y coordinates; although the key is embedded, the template does not display it. Colors are numeric factorization codes for kingdom labels and no color scale is shown. The kingdom label is derived from the second component of an `Eukaryota;...` lineage when present, otherwise the first lineage component; missing values become `Unknown`. The chart therefore colors broad labels (the embedded text values are Fungi and Metazoa), not named taxonomic ranks at finer resolution.

Parsed trace metadata establishes 20,000 points per panel, 20,000 unique keys per panel, no duplicate keys within either panel, and exact equality of the two key sets. In the sample, label counts are Fungi 13,906 and Metazoa 6,094 in each panel. Exact x/y ranges are not stated here because the HTML stores coordinates as Plotly base64 typed arrays rather than ordinary numeric JSON arrays; they remain recoverable by decoding each trace's `x` and `y` `bdata` as little-endian float32 values (`dtype: f4`). This encoding is reliable, as are the string/integer interval keys in `customdata`.

Available covariates in the generation pipeline include taxonomy, kingdom, GC content, Shannon entropy, sequence length, gene type, coding status, likelihood (mean log probability and duplicate-copy audit fields), sequence hash, and full-common-cohort exact-sequence duplicate count. These are not all embedded in this plot: its point metadata are only kingdom text and interval key. Run metadata records 7,251 likelihood keys with copies, likelihood maximum spread 0.0093, and no missing likelihood keys; these facts describe the joining process, not an island. The directly relevant shared sampling variable is the seeded sample. The sample is not a set of guaranteed independent genomes or intervals, and its row multiplicities may reflect overlapping/repeated records or exact sequence identity.

The projection parquet and source embedding matrix are not present among the delivered phase 4 artifacts. Thus original-space neighborhood stability cannot be evaluated from the available local artifacts. No reproducible cluster labels or boundary definition accompanies the HTML.

## Candidate search and interpretation

A candidate would need a reproducible coordinate-space membership rule, per-panel membership, corresponding keys and taxonomic composition, followed by a test in the original embedding neighborhood graph. The current artifact provides no predetermined island boundary or cluster assignments. Visual inspection alone cannot yield reproducible membership, and ad hoc boundaries would risk encoding the desired result. I therefore report zero candidates rather than inventing one or up to three visually selected regions.

| Candidate | Observable geometry | Taxonomic composition | Raw/whitened correspondence | Main uncertainty | Priority |
|---|---|---|---|---|---|
| None selected | No region is assigned membership because the supplied plot has no reproducible boundary or clustering result. | Whole-sample labels are 69.5% Fungi and 30.5% Metazoa in both panels; no region-specific composition is established. | The same 20,000 interval keys appear in both panels, but this does not establish persistence of any island. | Independent nonlinear projections, sampling, density, PCA truncation, whitening, and broad-label imbalance can all shape apparent geometry. Original-space neighbors and covariate-adjusted region statistics are unavailable locally. | Defer island ranking; first define a testable region without visual boundary selection. |

The strongest supported observation is that both projections display the same sampled intervals and broad kingdom labels, while applying different PCA score scaling before independent UMAP fits. A visually isolated pattern, if perceived, could arise from the projection, local density, the finite sample, dominant embedding directions, whitening, or correlations with sequence properties and exact duplicates. UMAP proximity is not an evolutionary distance, and a separated 2D group would not by itself establish a distinct biological population.

## Selected follow-up

**Selected candidate:** none; no candidate region is sufficiently well defined from the authorized artifact and available supporting files.

**Membership definition and tracking keys:** none is asserted. The stable tracking key format is `(record_id, start, end)`, embedded in each trace's `customdata`; because the two trace key sets are identical, keys can be joined exactly across panels.

**Strongest supported observation:** both panels contain the same 20,000 interval keys and have the same broad-label sample counts, but arise from separately fitted raw-score and whitened-score PCA→UMAP pipelines.

**Most important competing explanation:** apparent separation can be introduced or magnified by the two-dimensional projection and preprocessing, including variance rescaling by whitening, without a corresponding stable neighborhood in the original embedding space.

**One discriminating test:** after a candidate has been defined by a reproducible projection-space rule on the decoded coordinates, recover the existing 3,072-dimensional embeddings for its interval keys and compare their cosine-neighbor enrichment/membership stability against matched nearby non-candidate intervals, with a permutation null preserving broad kingdom and sequence-length strata. This would test whether the apparent group corresponds to a coherent original-space neighborhood beyond broad-label and length composition. It requires the embedding data, which are not present in the current local artifact set.

No next visualization plot is selected yet: the necessary next question is whether a reproducibly defined candidate has supporting original-space neighborhood structure after accounting for broad taxonomy and sequence length. A plot should be chosen only after a candidate exists and that question's available data are confirmed.

## Limitations

This is a reconstruction and evidence audit, not a fresh embedding or significance analysis. No new plot was created and no GPU enrichment was rerun. Exact coordinate ranges can be recovered from the embedded float32 arrays but were not needed to make the negative candidate decision. Without the projection parquet/original embeddings and interval-level covariate joins, neither projected cluster memberships nor original-space neighborhood persistence can be established here. The conclusion is deliberately limited to this one HTML and its minimum supporting artifacts.
