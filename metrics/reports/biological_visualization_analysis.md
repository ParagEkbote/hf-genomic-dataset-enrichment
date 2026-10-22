# Biological visualization analysis

**Scope and method.** This is a read-only analysis of the existing HTML visualizations, embedded Plotly traces, CSV/JSON summaries, and the visualization-generation code. It makes no new plots and does not rerun enrichment. The primary common cohort is 244,046 unique `(record_id, start, end)` keys. It is not a set of guaranteed independent genomes: intervals may overlap, originate from repeated records, or be exact sequence duplicates. Embedding UMAP and PCA panels use a deterministic sample of 20,000 rows; the norm/length chart reports 244,046 source rows but displays 100,000; the kNN summaries operate on edge data and the distance chart shows bounded samples. The source code says taxonomy is an ordered semicolon-delimited lineage, not named ranks. Figures are exploratory.

## A. Executive summary

**Observed.** Existing kNN edges are strongly lineage-prefix-consistent at shallow to intermediate depths: plot 04 reports 100.0% agreement at depth 1, 92.7% at depth 3, 84.4% at depth 4, and 81.5% at depth 5 (2,495,502 valid edges for each). Agreement falls to 42.2% by depth 10, where only 1,803,031 edges have enough lineage depth to be tested. This is evidence that the embedding-derived neighborhood graph is organized by broad taxonomy labels in this cohort, not evidence of phylogenetic distance or held-out classification performance.

**Observed.** The representation has a severe anisotropy/outlier-channel signal. In `plot01_dimension_diagnostics.csv`, dimension 2984 has variance 86.015 versus median per-dimension variance 0.01886 (about 4,561×); dimension 3, the configured diagnostic channel, has mean −15.37 and variance 25.71. This makes raw-space geometry and whitening choices consequential. The raw-versus-whitened UMAPs use separate PCA transforms (50 components in the source defaults) followed by cosine UMAP, so visual differences cannot establish that one representation is biologically superior.

**Observed.** Composition and likelihood have modest rank association: the Spearman matrix reports mean log probability versus GC content ρ=0.228 and versus Shannon entropy ρ=−0.068 (244,046 joined records). GC is a plausible partial correlate; entropy alone has little monotonic association in this summary. Likelihood remains raw mean log probability, explicitly not Markov-corrected. RNA gene-type groups have higher median scores than CDS in the group summary, but group, sequence length, composition, and tokenization are confounded and no adjusted comparison is supplied.

**Observed.** Exact-sequence multiplicity in the common cohort is associated with substantially higher raw likelihood medians: −6.48 for 1-copy rows (n=237,044), −3.26 for 2-copy rows (n=1,034), and −2.56 for 3+-copy rows (n=5,968). The duplicate definition is SHA-256 identity of strand-normalized sequence, not homology or repeated genomic elements. This is a strong redundancy-related likelihood pattern, not a biological novelty signal. Plot 08 colors UMAP by these exact sequence-copy counts, but no machine-readable island membership or corrected density test exists.

**Candidate islands.** No particular island can be defensibly named or assigned to taxonomy from the current evidence. The 20,000-point UMAP traces and hover keys are embedded in the HTML, but the artifact set has no cluster labels, cluster statistics, or exported linkage of an island boundary to composition, taxonomy, likelihood, and duplicates. No quantitative raw-versus-whitened island persistence test is reported. Thus any island in the display remains exploratory and potentially projection-driven.

**Important negatives.** No composition-adjusted likelihood, formal cluster significance, duplicate-controlled taxonomic agreement, or composition-adjusted taxonomic association is shown. The high-level taxonomy sunburst is a descriptive treemap-like aggregation, not a test of clade separation. No conclusion about function, novelty, adaptation, or evolutionary distance is supported.

## B. Visualization inventory

| Visualization | Primary variables | Main observation | Biological relevance | Main limitation |
|---|---|---|---|---|
| `embedding_visualizations/plot01_outlier_channel.html` | Per-dimension mean/variance; absolute dimension 3 versus L2 norm | The diagnostic is designed to reveal dominant channels; CSV confirms an exceptionally high-variance channel, with the largest variance elsewhere at dimension 2984. | Representation anisotropy may shape similarity, PCA, and apparent islands. | It is a channel diagnostic, not a biological grouping analysis; plotted norm relation is sampled and does not identify outlier records or cause. |
| `embedding_visualizations/plot02_umap_raw_vs_whitened.html` | Two 2-D UMAPs of 20,000 intervals, colored by kingdom (or taxonomy fallback) | Side-by-side projections compare raw PCA scores to whitened PCA scores. | Could reveal broad cohort structure and whether a visible configuration changes after whitening. | Separate nonlinear projections; global distances and island persistence are not established. No cluster assignments. Defaults: 50 PCA components, cosine UMAP, 25 neighbors, min_dist 0.1, seed 42. |
| `embedding_visualizations/plot03_embedding_norm_vs_length.html` | Embedding L2 norm vs sequence length | Title gives Spearman ρ=0.056 for n=244,046; only 100,000 plotted. Two trace labels are Metazoa and Fungi. | Tests whether norm is mainly a length proxy. The association is weak in rank terms. | No adjustment for group or composition; displayed subsample; limited kingdom coverage in the shown traces. |
| `embedding_visualizations/plot04_knn_lineage_agreement.html` and `.csv` | Fraction of kNN edges sharing a full lineage prefix at each depth | Strong broad-prefix agreement with declining agreement at deeper prefixes. | Direct evidence of taxonomy-structured neighborhoods in the available graph. | Existing edges, not held-out predictions; lineage components are not named ranks; edge records can be correlated and taxon coverage uneven. |
| `embedding_visualizations/plot05_trinucleotide_vs_embedding_pca.html` | Separate PCA of 64 trinucleotide frequencies and Carbon embeddings, same projected intervals, colors by kingdom/taxonomy | Provides a side-by-side view of composition and learned representation. | Helps ask whether broad taxon groupings are also visible in 3-mer composition. | Independent PCA bases and scales; visual similarity is not a quantitative dependence test. Trace coordinates are embedded, but no explained variance, canonical correlation, or adjusted taxonomic test is provided. |
| `embedding_visualizations/plot06_umap_by_likelihood.html` | Raw UMAP coordinates colored continuously by mean log probability | Maps raw model score onto the 20,000-row UMAP sample. | Can reveal score gradients or score-enriched regions. | Score is raw, not Markov-corrected; visual color gradients do not establish separate populations. |
| `embedding_visualizations/plot07_umap_by_composition.html` | Same raw UMAP colored separately by GC content and Shannon entropy | Direct visual overlays of two CPU composition features. | Tests whether visible UMAP regions align with simple composition gradients. | Color overlays are exploratory; no cluster-level effect size or adjustment. |
| `embedding_visualizations/plot08_umap_by_duplication.html` | Raw UMAP colored by SHA-256 sequence frequency | Exact-sequence multiplicity is displayed for the projected sample, counts from the full common cohort. | Identifies whether repeated sequences coincide with dense projected regions. | No quantitative island-specific enrichment or local-density comparison; exact identity only. |
| `embedding_visualizations/plot09_intra_vs_inter_clade_distance.html` and `_counts.csv` | kNN cosine distance by shared lineage-prefix-depth bin | 1–2, 3–4, 5–6, 7–8, 9+ bins have 181,480; 279,149; 675,944; 365,044; 993,885 edges; up to 50,000 rows sampled per bin. | Tests whether more shared lineage labels correspond to smaller cosine distance. | Not exactly intra- versus inter-clade: bins are pairwise shared-prefix depths, and zero-shared has zero recorded edges. The CSV has counts, not medians or tests. Bounded sampled violin data are in HTML. |
| `likelihood_visualizations/01_likelihood_distribution.html` | Histograms of mean log probability and perplexity | n=244,046 deduplicated interval keys; perplexity is an inverse transform of score, not independent evidence. | Describes score range and potential tails. | Histogram alone cannot assign biological meaning or distinguish sequence composition effects. |
| `likelihood_visualizations/02a_gc_vs_likelihood.html` | 60×60 count density heatmap of GC vs mean log probability | Full joined cohort (n=244,046); Spearman ρ=0.228 in correlation CSV. | GC is associated with score to a modest degree. | Binned exploratory association; no conditional model, and GC may covary with taxonomy and gene class. |
| `likelihood_visualizations/02b_entropy_vs_likelihood.html` | 60×60 count density heatmap of Shannon entropy vs mean log probability | Full joined cohort; Spearman ρ=−0.068. | Little global monotonic relationship by this one summary. | Nonlinear or subgroup patterns could remain; no adjustment. |
| `likelihood_visualizations/03_likelihood_by_biological_group.html` and summary CSV | Raw mean log probability by gene_type and coding flag | Medians: CDS −6.489 (n=222,428), rRNA −1.817 (1,963), tRNA −3.185 (9,560), ncRNA −7.255 (7,026), pseudo −5.815 (2,505), misc_RNA −6.582 (564); coding true/false −6.489/−4.793. | Gene class is associated with raw model score in this cohort. | Unequal group sizes, different sequence lengths/composition, no uncertainty test or adjustment; score is not corrected. |
| `likelihood_visualizations/04_taxonomic_sunburst.html` | Taxonomy-prefix hierarchy sized by records and colored by median raw score | 2,990 nodes; displays hierarchical score summaries. | Locates descriptive score variation in labeled lineages. | Ordered lineage text, not guaranteed named ranks; many nested nodes and no multiplicity-adjusted comparisons. |
| `likelihood_visualizations/06_likelihood_vs_duplication.html` and `06_duplication_summary.csv` | Raw score by exact sequence-copy category | Higher copy-count categories have much higher median raw scores (values above). | Flags redundancy as a score confounder. | Exact hash equality only; category groups are unequal and repeated observations are not independent biological replicates. |
| `likelihood_visualizations/07_minimum_likelihood_position.html` | Normalized location of minimum token log probability, n=244,046 | Describes where the least likely token falls within a sequence. | Could motivate boundary/artifact checks. | No per-position covariates or group summary in the supplied artifacts; no anomaly identification can be made. |
| `likelihood_visualizations/08_summary_correlation_matrix.html` and `.csv` | Spearman correlations among score, composition, and length | Most useful global quantitative context: GC-score 0.228, entropy-score −0.068, length-score 0.074, GC-entropy 0.261. | Establishes global pairwise associations and weak score-length monotonic association. | Unadjusted correlations; does not distinguish within-taxon from between-taxon effects. |
| `sunburst_visualizations/01_taxonomy_embedding_dispersion.html` | Median cosine distance to deepest adequately sampled taxonomy-prefix centroid | 77 hierarchy nodes; groups below 20 are excluded per metadata. | Descriptive within-group dispersion by taxonomy prefix. | Uses normalized-vector cosine centroids; sample-size and lineage depth matter. No tabular node summary or uncertainty interval is supplied, and this is not a raw/whitened comparison. |
| `sunburst_visualizations/02_taxonomy_cpu_composition.html` | Taxonomy hierarchy with GC quantile and coding status, colored by median Shannon entropy | 400 nodes; GC bands are cohort-relative quantiles, with coding status as an added ring. | Shows taxonomy/composition cross-tab structure. | Hierarchical aggregation can conceal sparse subgroup sizes; it does not adjust embedding or likelihood results. |
| `sunburst_visualizations/03_cpu_vs_gpu_profiles.html` | Taxonomy × GC quantile × GPU-likelihood quantile; color median mean log probability | 367 nodes; CPU GC and GPU score are binned into corpus-relative quantiles. | Describes joint profiles, potentially useful for selecting strata. | It bins the same likelihood that colors the chart and is descriptive; it does not show GPU embedding information gain or statistical independence. |

## C. Candidate island and anomaly register

| Candidate | Location or group | Observed pattern | Supporting plots | Competing explanations | Confidence | Next validation |
|---|---|---|---|---|---|---|
| No named embedding island is established | Raw/whitened 20,000-row UMAP projections; no cluster identifier available | The HTML contains points and hover interval keys, but the delivered summaries do not define islands or quantify their membership. A specific island cannot be cross-referenced across all covariates without defining membership. | `plot02`, `plot05`–`plot08` | UMAP projection, color scale, sampling, anisotropy, composition, length, exact duplicates, and taxonomy imbalance could each produce apparent separation. | Low for any biological island claim | Recover interval-level coordinates and metadata, define clusters in original cosine space and projected space, then test covariate/taxonomy enrichment with matched or adjusted controls and a second sample/metric. |
| Broad taxonomy structure in neighborhoods | kNN graph across common-cohort interval keys | Prefix agreement is 1.000 at depth 1, 0.927 at depth 3, 0.816 at depth 5, falling to 0.422 at depth 10 (depth-specific eligible edge counts). | `plot04` + CSV; compatible context in `plot09` | Taxonomy abundance, duplicated sequence keys, shared composition, and non-independent kNN edges may inflate agreement. | Moderate for descriptive association; low for evolutionary meaning | Recompute at unique-sequence level; compare against taxonomy-frequency and GC/length matched nulls; report confidence intervals by query and taxon. |
| Strong embedding anisotropy | Dimensions 2984 and 3 | Maximum variance is 86.015, about 4,561× median dimension variance; dimension 3 also has a large negative mean and variance 25.711. | `plot01` and diagnostic CSV; motivates comparison in `plot02` | A learned channel may be meaningful, or a numerical/preprocessing artifact; summary alone cannot distinguish. | High for the variance anomaly, low for its cause | Inspect per-record values/IDs for extreme channels; compare norms, cosine neighbors, and kNN taxonomic metrics before/after excluding/robustly scaling channels. |
| Exact duplicate groups have elevated raw model score | 2-copy and 3+-copy categories in common cohort | Medians −3.255 and −2.557 vs −6.484 for single-copy sequences. | `06_duplication_summary.csv`, `06_likelihood_vs_duplication.html`, `plot08` | Identical sequences can be shorter/easier, compositionally atypical, or overrepresented; duplicate status may reflect sampling of records. | Moderate for association, low for mechanism | Compare sequence-length and gene-class matched unique sequence groups; collapse to one observation per sequence hash and re-evaluate score distribution and kNN agreement. |
| Gene-class score contrast | rRNA/tRNA vs CDS | rRNA and tRNA medians are less negative than CDS; group sizes are strongly unequal. | `03_biological_group_summary.csv`, `03_likelihood_by_biological_group.html` | Length, composition, RNA-specific sequence patterns, and tokenization effects. | Moderate for unadjusted group association | Fit a robust adjusted model including log length, GC, entropy, taxonomy, and exact-hash group; report within-clade contrasts. |

## D. Taxonomic structure and embedding geometry

The clearest direct evidence is the existing kNN-edge lineage agreement curve. The tested values are **ordered lineage-prefix depths**, not formally identified ranks such as genus or species. It is therefore accurate to say that neighborhoods often share broad prefixes and progressively less often share deeper prefixes; it is not accurate to map depth 5 to a named rank without taxonomy schema metadata. There are 2,495,502 valid edge comparisons for depths 1–5; eligible edges decrease at depths 6–12 (for example 1,803,031 at depth 10 and 574,846 at depth 12). The apparent increase from depth 10 to 12 is a warning about changing eligibility and sparse coverage, not evidence that the deepest level is more organized.

Plot 09 groups kNN edges by shared prefix and plots cosine distance, sampling at most 50,000 edges per category. Its category counts are 181,480 for 1–2 shared, 279,149 for 3–4, 675,944 for 5–6, 365,044 for 7–8, and 993,885 for 9+; the `0 shared` bin has no edges in the generated CSV. This is not a clean intra-clade versus inter-clade comparison: edges with no shared prefix are absent and the groups vary in depth and taxonomic eligibility. The HTML trace has the sampled distances, but the supplied CSV does not give median/quantile distances or tests. No exact “intra distances smaller than inter distances” claim is made here without extracting and comparing those arrays under a defined eligible cohort.

`01_taxonomy_embedding_dispersion.html` reports median cosine distance from each normalized embedding to the centroid of the deepest qualifying taxonomy ancestor, with minimum group size 20. It has 77 hierarchy nodes and 244,046 matched rows according to metadata. The chart supports exploratory comparison of displayed node medians, but the artifact does not export a node-level CSV or uncertainty estimates. No named clade is promoted here as unusually dispersed. The sunburst’s hierarchy levels remain lineage components, not validated ranks.

The raw/whitened chart contains 20,000 points per panel, using 50-component PCA then cosine UMAP. Whitening rescales retained PCA directions by their variance; it changes geometry and can amplify low-variance directions. The dimension diagnostics make this change especially consequential. The charts do not report neighborhood overlap across raw and whitened embeddings, nor a downstream taxonomic score by representation. Therefore they support sensitivity inspection, not a claim that whitening improves or degrades biology. PCA of trinucleotide frequencies and embeddings in plot 05 uses separate bases; visual separation in those panels is not directly comparable as distance or proof that composition explains the learned geometry.

## E. Composition, likelihood, and redundancy

**GC and likelihood.** The Spearman coefficient is 0.228 for GC versus mean log probability. The 60×60 heatmap covers 244,046 joined records. This is a modest positive monotonic association. It could reflect base composition effects on model token probabilities, taxonomic differences, or both. The raw score has not been composition-corrected.

**Entropy and likelihood.** The global coefficient is −0.068. This argues against a strong global monotonic association with Shannon entropy in these records, while leaving nonlinear or subgroup patterns unresolved. Both correlations are unadjusted.

**Trinucleotide composition and embedding geometry.** Plot 05 compares separate two-component PCAs colored by broad taxonomy labels. It establishes that the same intervals were used in both views; it does not quantify whether embedding coordinates can be predicted by 3-mer composition. No adjusted R², distance correlation, or residual taxonomic organization is available. Plot 07 overlays GC and entropy on raw UMAP but offers no cluster-level test.

**Norm and length.** Plot 03’s full-cohort Spearman ρ=0.056 (n=244,046) indicates a very weak global rank association between L2 norm and length. The 100,000 displayed points are a visualization subsample and are split into Metazoa and Fungi traces. This does not rule out group-specific effects, nor does norm equal direction-based embedding similarity.

**Duplication and density.** Duplication is exact SHA-256 frequency of strand-normalized sequence over the common cohort. It does not detect homologs, repeats, or approximate sequence similarity. Likelihood medians rise sharply from one-copy to multi-copy categories. Plot 08 can reveal visual overlap of copy counts with regions, but there is no reported local-density statistic or island enrichment test. Repeated intervals and identical sequences may create many highly similar neighbors, so taxonomic agreement is not duplication-controlled.

**CPU/GPU profiles.** CPU GC and GPU raw likelihood are joined on unique interval keys; no CPU duplicate keys remain after cache preparation, while 7,251 likelihood keys have duplicate source rows (10 disagreeing keys; maximum spread 0.00930). Duplicate likelihood rows were collapsed. The CPU/GPU sunburst bins GC and score into quantiles and is descriptive. It shows joint profiles by construction, but does not demonstrate complementary biological information. The CPU composition and GPU likelihood are related modestly at the global score level through GC; independent contribution requires adjusted models and held-out comparisons.

## F. Evidence-backed assertions

**Assertion:** Existing Carbon kNN neighbors share broad taxonomy lineage prefixes at high rates in this 244,046-key cohort.

**Evidence:** `plot04_knn_lineage_agreement.csv`: 2,495,502 eligible edges, agreement 1.000 at depth 1, 0.927 at depth 3, and 0.815 at depth 5.

**Interpretation:** The available representation-derived neighborhood graph carries broad taxonomic organization.

**Limitation:** These are existing edges, not held-out predictions; ranks are unnamed, edges may be correlated, and duplicate/composition effects are unadjusted.

**Confidence:** Moderate.

**Assertion:** Embedding coordinate variance is highly anisotropic, with a dominant high-variance dimension.

**Evidence:** `plot01_dimension_diagnostics.csv`: dimension 2984 variance 86.015 versus median 0.01886; dimension 3 has mean −15.370 and variance 25.711.

**Interpretation:** PCA/whitening and cosine neighborhood choices can materially alter the geometry.

**Limitation:** The cause and per-record biological relevance of the channels are unknown.

**Confidence:** High for anisotropy; low for biological interpretation.

**Assertion:** GC content is more strongly associated with raw mean log probability than Shannon entropy is in the global rank summaries.

**Evidence:** `08_summary_correlation_matrix.csv`: Spearman ρ=0.228 for GC, −0.068 for entropy (n=244,046 joined keys).

**Interpretation:** GC is a plausible score confounder; entropy has little global monotonic association.

**Limitation:** Unadjusted correlations cannot separate taxonomy, length, and composition effects; score is not Markov-corrected.

**Confidence:** Moderate.

**Assertion:** Exact repeated sequences have substantially higher raw likelihood medians than one-copy sequences in the common cohort.

**Evidence:** `06_duplication_summary.csv`: −6.484 (237,044 rows), −3.255 (1,034 rows), and −2.557 (5,968 rows) across 1, 2, and 3+ exact copies.

**Interpretation:** Redundancy/sampling is a major alternative explanation for score-enriched groups and potentially dense neighborhoods.

**Limitation:** No matched adjustment, independent sequence-level sampling, or causal interpretation is provided.

**Confidence:** Moderate.

**Assertion:** The global embedding norm–length association is weak.

**Evidence:** `plot03_embedding_norm_vs_length.html` title reports Spearman ρ=0.056, n=244,046, with 100,000 points shown.

**Interpretation:** Length alone is unlikely to explain a strong global norm gradient.

**Limitation:** Group-specific and nonlinear relationships remain possible; this says nothing about direction-based geometry.

**Confidence:** Moderate.

**Assertion:** Raw likelihood differs by annotated gene type in the observed cohort.

**Evidence:** `03_biological_group_summary.csv`: median rRNA −1.817 (n=1,963), tRNA −3.185 (9,560), CDS −6.489 (222,428), and ncRNA −7.255 (7,026).

**Interpretation:** The model score is sensitive to sequence class or correlated properties.

**Limitation:** Unequal groups and lack of adjustment prevent attributing the differences to biological class itself.

**Confidence:** Moderate.

## G. Negative findings and unresolved questions

- No island is assigned a reproducible membership label in the current deliverables. Point hover keys make recovery possible, but the report cannot infer island boundaries from a figure without a stated clustering rule.
- No formal cluster significance or raw-versus-whitened stability comparison is reported. Visual separation alone cannot establish a population.
- The available plots do not establish named-rank agreement. Lineage prefix depths are not guaranteed ranks, and taxonomy coverage gets sparse at deeper depths.
- No named clade is claimed to have exceptional dispersion: the sunburst has no accompanying tabular node summary, uncertainty, or sample-size-adjusted comparison.
- The intra/inter distance claim is unresolved. The bins omit zero-shared edges and the count CSV has no distance summaries/tests; the plot is sampled and categorized by prefix depth rather than clade identity.
- No evidence establishes that trinucleotide composition explains the embedding structure, or that taxonomic structure persists after conditioning on composition and length.
- No composition-only or Markov baseline correction is present. Raw likelihood cannot be read as biological importance, novelty, quality, or anomaly.
- Duplication is exact sequence identity only. It cannot identify homologous sequences, repeats, or near duplicates; no duplicate-controlled kNN statistic is included.
- Sampled projection plots, full-cohort likelihood plots, and graph-edge summaries have different observational units and are not interchangeable cohorts.
- The kNN and likelihood joins report zero missing keys in the common cohort and no CPU-cohort duplicate interval keys, which reduces join-cardinality concerns for those tables. However, embedding rows streamed (251,427) exceed metadata rows (244,046); the metadata does not explain all row-level correspondence beyond reporting full matches for the sunburst and duplicate likelihood key handling.
- The most useful missing artifacts are an interval-level analysis table for all plotted coordinates and covariates, node-level dispersion summaries with counts, and explicit taxonomy schema/rank mapping.

## H. Recommended follow-up analyses

1. **Test candidate islands with matched covariates and original-space geometry (highest value).**
   - **Question/hypothesis:** Do any UMAP-visible islands correspond to stable, taxonomically enriched neighborhoods beyond GC, entropy, length, and exact duplication?
   - **Inputs:** Existing 20,000 sampled interval keys and raw/whitened UMAP coordinates; original 3,072-D embeddings; taxonomy; GC, entropy, length, likelihood, gene type, sequence hash.
   - **Method:** Define clusters in the projected coordinates for discovery only; quantify their membership and taxonomy/covariate enrichments. Validate in original cosine space using neighborhood purity and bootstrap/held-out samples. Fit logistic/multinomial or conditional permutation models adjusting for log length, GC, entropy, and exact-hash groups; correct for multiple comparisons.
   - **Control:** Matched random sets within taxonomy-depth and length/GC strata; repeat after collapsing exact hashes.
   - **Interpretation:** Persistence across original space, representation transforms, and covariate controls supports a stable candidate population; disappearance after adjustment suggests projection/composition/redundancy confounding.

2. **Recalculate taxonomic neighborhood agreement after deduplication and covariate matching.**
   - **Question/hypothesis:** Is broad-prefix agreement still elevated when each exact sequence contributes once and compared with composition/length-matched null neighbors?
   - **Inputs:** Full kNN neighbor edges, exact sequence hashes, lineage strings with rank mapping, GC, entropy, length.
   - **Method:** Rebuild/ filter kNN at unique-hash level; report query-level bootstrap intervals by taxon, rank-specific eligible counts, and matched permutation baselines.
   - **Control:** Shuffle taxonomy within strata of GC, length, group size, and sequence-hash frequency.
   - **Interpretation:** Excess agreement beyond matched nulls would strengthen evidence for taxonomic organization; collapse would indicate composition or redundancy may explain a substantial share.

3. **Quantify raw-versus-whitened geometry and channel influence.**
   - **Question/hypothesis:** Is taxonomic organization robust to whitening and dominant-channel handling?
   - **Inputs:** Embeddings, raw/whitened PCA/UMAP coordinates, exact keys, taxonomy.
   - **Method:** Compare neighborhood overlap, cosine distances, prefix agreement, and cluster co-membership for raw embedding, whitened PCA representation, and sensitivity representations with dominant channels removed or robustly scaled. Inspect coordinate 2984 and 3 by record and metadata.
   - **Control:** Same records and deterministic projection settings across transformations; compare with random labels.
   - **Interpretation:** Similar original-space neighborhood results support representation robustness; large changes indicate geometry/preprocessing sensitivity.

4. **Estimate adjusted likelihood contrasts by class and taxonomy.**
   - **Question/hypothesis:** Do gene-type score differences remain after accounting for sequence length, GC, entropy, taxonomy, and duplicates?
   - **Inputs:** Per-interval likelihood, CPU features, taxonomy, gene type, duplicate hash/copy number.
   - **Method:** Robust regression or stratified matched contrasts on raw mean log probability; use taxon-clustered uncertainty and one observation per exact sequence as sensitivity analysis. Include sequence-length splines to avoid assuming linearity.
   - **Control:** Within-lineage comparisons and deduplicated cohort.
   - **Interpretation:** Persistent within-stratum differences motivate model/tokenization study; attenuation points to confounding.

5. **Export and test taxonomy dispersion summaries.**
   - **Question/hypothesis:** Which adequately sampled lineage groups have genuinely elevated within-group cosine dispersion?
   - **Inputs:** Per-interval normalized embedding and taxonomy, plus centroid assignment details.
   - **Method:** Export node n, median/mean cosine distance and bootstrap intervals; compare groups at the same lineage depth and matched n, controlling for composition and duplicate status.
   - **Control:** Permute taxon labels within broad lineage and composition strata; repeat with unique sequence hashes.
   - **Interpretation:** Reproducible excess dispersion may identify heterogeneous labels or biological diversity; unstable small groups indicate sampling noise.

6. **Recover the distance relationship with eligible comparisons.**
   - **Question/hypothesis:** Do cosine distances decrease with shared taxonomy depth within a balanced eligible cohort?
   - **Inputs:** kNN edges including cosine distance, query/neighbor keys, taxonomy ranks, duplicates.
   - **Method:** Report medians/quantiles and bootstrap intervals by explicitly named rank, balance query counts and clades, and separate within-terminal-taxon from between-taxon edges. Include edge-level dependence handling.
   - **Control:** Matched random pairs from same composition/length strata and taxonomy label permutations.
   - **Interpretation:** A monotone within-cohort effect beyond matched null supports association with taxonomy; overlap or collapse limits claims about geometric separation.

