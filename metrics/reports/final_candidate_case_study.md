# Final candidate case study: fungal taxonomy, confounders, and interpretation

## Executive conclusion

`raw_C2_e1_m20` is a reproducible density-connected region under the tested rule on the **raw two-dimensional UMAP**. It contains 3,501 of the sampled 20,000 intervals, including 3,419 Fungi. Its fungal composition is not representative of all sampled fungi: it is heavily weighted toward lineage components labelled `Ascomycota` and, at finer component depths, `Sordariomycetes`, `Eurotiomycetes`, and `Eurotiales`. It is not a uniform fungal grouping, and several particular full lineages are represented by many intervals from relatively few source records.

The raw likelihood association also persists within Fungi after a quadratic GC adjustment and categorical adjustment for broad lineage components. The candidate coefficient falls from +1.524 to +1.446 after GC adjustment, +1.310 after adding component-depth-4 fixed effects, and +1.293 in a well-supported common subset with component-depth-6 fixed effects. These are record-clustered descriptive model estimates in the same sample, not confirmatory inference: the candidate was discovered and evaluated in this cohort, unmeasured covariates remain, and record accessions are not verified independent genomes.

Exact-sequence repeats are uncommon in the candidate (8/3,501) and especially uncommon among candidate Fungi (4/3,419). Excluding every interval marked as repeated leaves the within-Fungi score difference essentially unchanged (+1.534 nats per supervised token). This argues against repeat frequency materially accounting for the association, but it is not exact one-per-sequence deduplication because the hash values themselves are not in the local artifacts.

**Strongest defensible conclusion:** this is a projection-level region associated with a skewed set of fungal lineage strings, higher GC, and higher raw model mean log probability. The association remains after the measured GC and broad lineage adjustments and is not explained by the small number of repeated-sequence rows. The separately fitted whitened UMAP does not retain it as a separate component, original embedding vectors are unavailable, and key sequence covariates are absent. The evidence does not establish a distinct biological population, function, fitness, or causal explanation for the UMAP geometry.

## Candidate, cohort, and data-lineage validation

The exact membership is from `umap_dbscan_memberships.csv` where `panel=raw` and `eps_1_min20_cluster=2` (DBSCAN eps 1.0, min_samples 20). There are 3,501 unique candidate keys and 16,499 other keys in the same 20,000-point raw UMAP cohort. The key is `(record_id, start, end)`; all keys are unique in the sample. Candidate kingdom counts are 3,419 Fungi and 82 Metazoa. The earlier report's sample size and kingdom counts reproduce exactly.

The keyed composition and likelihood tables each contain 20,000 unique rows and exactly the candidate UMAP cohort key set; no key is unmatched in either join. The composition table has finite GC and entropy values for all sample keys. The likelihood table has finite `mean_log_prob` for all sample keys. The authoritative local taxonomy file, `common_cohort/taxon_analysis/common_cohort_taxonomy.csv`, has 244,046 unique interval keys and a nonempty lineage for all 20,000 sample keys; the join is exact and one-to-one with zero unmatched sample keys and zero duplicate keys. It contains only a semicolon-delimited `taxonomy` lineage column, not separate named-rank fields.

The source record identifier available is `record_id` (accession-like values such as `NC_...`/`NW_...`). It identifies the source sequence record used for the interval; it is not documented here as a genome identifier, and multiple accessions may derive from one genome. Within sampled Fungi, candidate intervals occur across 912 distinct `record_id`s and non-candidate intervals across 1,663. Record-clustered model uncertainty below uses these IDs as source-record clusters, while acknowledging they may not capture dependence across accessions from the same source genome.

Source code shows that CPU metadata can include `sequence_length`, `gene_type`, `is_coding_region`, and `sequence_hash=SHA256(strand_normalized_sequence)`. These fields are part of the generator's source join, but the local `common_cohort_metadata.parquet` and `embedding_projection_sample.parquet` artifacts are absent. The existing joined sample tables do not contain those fields. Length was not approximated by `end-start`; coding status, gene type, and exact sequence fingerprints therefore cannot be joined or modeled from current local artifacts.

The existing duplication plot has explicit interval keys in `customdata`, all 20,000 keys match the sample exactly, and its color is the pipeline's `duplicate_count_in_common_cohort`. The plotted repeat-status text agrees with whether this count is greater than one. Per the generator, the count is based on SHA-256 identity of strand-normalized sequence over the full common cohort. The plot exposes counts/status, but not the hash itself or sequence text.

## Taxonomic resolution and coverage

### Named-rank limitation

The local taxonomy source gives ordered lineage components but no rank annotations. `phase4.py` explicitly warns that taxonomy strings are ordered semicolon-delimited lineages rather than rank-annotated data. `results/derived/distributions.py` defines a legacy fixed-position list of named ranks, but it extracts by array position without storing rank labels per component. Actual lineages can omit intermediate terms: for example, component depth 8 includes `Eurotiales` in some paths and `Cryptococcaceae` in others, while depth 10 includes both `Aspergillus` and `Suillaceae`. Thus fixed positions can mix taxonomic levels. I do not treat those positional slots as verified phylum/class/order/family/genus assignments.

Kingdom is the exception: the generator explicitly derives it from the second lineage component for `Eukaryota;...`; the sample has no missing kingdom labels. Formal phylum, class, order, family, and genus coverage cannot be reported reliably from this artifact as named ranks. Species is not in the repository's rank list, and terminal lineage strings include subgenera and species-complex labels rather than a consistently annotated species rank. A missing component at a given depth cannot be distinguished from an omitted rank versus absent/incomplete taxonomy metadata. The table below therefore reports **ordered component-depth coverage**, not named-rank coverage. The source's nominal rank slot is shown only as context and is not asserted as a validated rank.

### Component coverage by depth within Fungi

Percentages use candidate Fungi (n=3,419) and non-candidate Fungi (n=10,487) as separate denominators. “Unique components” counts distinct literal strings at that depth. “Full lineage” is the complete literal string, without assigning the terminal term a species rank.

| Taxonomy field | Candidate assigned; coverage; unique values | Non-candidate assigned; coverage; unique values | Interpretation |
|---|---:|---:|---|
| Kingdom | Fungi 3,419; 100%; 1 | Fungi 10,487; 100%; 1 | Derived by the documented kingdom helper. |
| Component depth 4 (legacy list's nominal phylum position) | 3,419; 100%; 6 | 10,487; 100%; 6 | The component labels are usable descriptively; rank status is not verified. |
| Component depth 6 (nominal class position) | 3,419; 100%; 26 | 10,468; 99.82%; 39 | Nonempty value coverage; absent values may reflect lineage truncation/omission. |
| Component depth 8 (nominal order position) | 3,418; 99.97%; 85 | 10,434; 99.49%; 111 | Component values can include different taxonomic levels. |
| Component depth 10 (nominal family position) | 2,829; 82.74%; 98 | 7,459; 71.13%; 122 | Mixed-level values are directly visible (e.g. `Aspergillus`, `Suillaceae`). |
| Component depth 11 (nominal genus position) | 1,436; 42.00%; 42 | 4,117; 39.26%; 50 | Terminal strings include subgenera and species complexes. |
| Complete lineage string | 3,419; 100%; 218 | 10,487; 100%; 284 | Exact lineage paths are available, but terminal rank is not. |
| Species rank | Not assignable | Not assignable | No explicit species field/rank; no species-level coverage estimate is made. |

Full coverage by every observed component depth is in `final_candidate_taxonomy_depth_summary.csv`. The complete lineage source has no missing values in this sample; this does not mean every lineage contains every formal taxonomic rank.

### Candidate versus non-candidate lineage components

Counts are intervals; percentages are within the assigned fungi group unless the depth has incomplete coverage, in which case the denominator is the assigned count. The percentage-point difference is candidate share minus non-candidate share, using the full fungal group sizes (3,419 and 10,487) for comparability. These are descriptive contrasts, not tests of enrichment.

| Component depth and literal component | Candidate n (% of candidate Fungi) | Non-candidate n (% of non-candidate Fungi) | Difference | Distinct candidate / non-candidate source records |
|---|---:|---:|---:|---:|
| 4: `Ascomycota` | 2,861 (83.68%) | 6,748 (64.35%) | +19.33 pp | 665 / 1,063 |
| 4: `Basidiomycota` | 510 (14.92%) | 3,161 (30.14%) | −15.22 pp | 219 / 481 |
| 4: `Mucoromycota` | 19 (0.56%) | 355 (3.39%) | −2.83 pp | 16 / 76 |
| 6: `Sordariomycetes` | 1,142 (33.40%) | 2,528 (24.11%) | +9.29 pp | 212 / 321 |
| 6: `Eurotiomycetes` | 1,134 (33.17%) | 2,315 (22.07%) | +11.10 pp | 241 / 338 |
| 6: `Agaricomycetes` | 299 (8.75%) | 2,203 (21.01%) | −12.26 pp | 140 / 353 |
| 6: `Dothideomycetes` | 272 (7.96%) | 665 (6.34%) | +1.62 pp | 100 / 194 |
| 8: `Eurotiales` | 860 (25.15%) | 1,601 (15.27%) | +9.88 pp | 174 / 241 |
| 8: `Glomerellales` | 476 (13.92%) | 959 (9.14%) | +4.78 pp | 70 / 110 |
| 8: `Hypocreales` | 430 (12.58%) | 905 (8.63%) | +3.95 pp | 79 / 115 |
| 8: `Boletales` | 123 (3.60%) | 948 (9.04%) | −5.44 pp | 71 / 181 |
| 10: `Aspergillus` | 556 (16.26%) | 1,083 (10.33%) | +5.93 pp | 110 / 164 |
| 10: `Colletotrichum` | 471 (13.78%) | 944 (9.00%) | +4.78 pp | 66 / 103 |
| 10: `Penicillium` | 226 (6.61%) | 354 (3.38%) | +3.23 pp | 49 / 56 |
| 10: `Suillaceae` | 122 (3.57%) | 930 (8.87%) | −5.30 pp | 70 / 172 |
| 11: `Colletotrichum gloeosporioides species complex` | 233 (6.81%) | 432 (4.12%) | +2.69 pp | 16 / 21 |
| 11: `Aspergillus subgen. Circumdati` | 233 (6.81%) | 468 (4.46%) | +2.35 pp | 76 / 120 |
| 11: `Suillus` | 122 (3.57%) | 930 (8.87%) | −5.30 pp | 70 / 172 |

The component-depth names are literal taxonomy strings; the labels at depths 10 and 11 illustrate why they should not be reported as formally assigned ranks. Full component counts and the top 15 candidate components at each depth are in `final_candidate_taxonomy_components.csv`.

The most common complete candidate fungal lineage is `Eukaryota;Fungi;Dikarya;Ascomycota;Pezizomycotina;Eurotiomycetes;Eurotiomycetidae;Eurotiales;Aspergillaceae;Aspergillus;Aspergillus subgen. Circumdati` (233 intervals, 6.81% of candidate Fungi, across 76 source records); the same literal lineage has 468 non-candidate fungal intervals (4.46%, 120 source records). Another full lineage ending in `Colletotrichum gloeosporioides species complex` has 233 candidate intervals (6.81%) across 16 records and 432 non-candidate intervals (4.12%) across 21 records. The candidate also has 216 intervals on a `Penicillium` lineage (6.32%, 48 records), while a `Fusarium` lineage has 128 (3.74%) but comes from only 2 source records. Conversely, a `Suillus` lineage has 122 candidate intervals (3.57%, 70 records) versus 930 non-candidate intervals (8.87%, 172 records). These examples show both over- and under-representation and variable record concentration; they do not establish formal rank-specific enrichment. The top exact paths are in `final_candidate_exact_lineages.csv`.

**Taxonomic answer:** the sampled candidate is dominated by fungal lineage strings containing `Ascomycota` at component depth 4, with large components labelled `Sordariomycetes` and `Eurotiomycetes` at depth 6 and `Eurotiales`, `Glomerellales`, and `Hypocreales` at depth 8. It also contains a broad tail of other fungal components. It does not represent all Fungi: Ascomycota components account for 83.68% and Basidiomycota 14.92% of candidate Fungi, and many candidate intervals come from particular lineages. Because rank annotations are absent, these are literal component-level findings rather than a claim of validated class/order/genus/species coverage.

## Final adjusted likelihood analysis within Fungi

The outcome is `mean_log_prob`, the mean natural-log probability per supervised token (nats per token) computed from model logits. Higher/less negative scores mean the model assigned greater average probability to supervised tokens. It is a raw score, not Markov-corrected and not a measure of fitness or function.

The table uses ordinary least squares coefficients with source-record-clustered sandwich standard errors and normal-approximation 95% intervals. The sample is the same 13,906 Fungi: 3,419 candidate and 10,487 non-candidate intervals in models 1–3. Model 2 adds centered GC and centered GC squared. Model 3 adds fixed effects for the six component-depth-4 categories. Model 4 is a finer sensitivity restricted to component-depth-6 categories with at least 20 candidate and 20 non-candidate observations; this retains 3,343 candidate and 9,583 non-candidate Fungi (12,926 total), excluding 76 candidate and 904 comparator intervals outside those common-support categories. The depth labels are not treated as formal ranks.

| Model | Terms | n | Source-record clusters | Candidate coefficient (nats/token) | Clustered 95% CI | R² | Residual SD (range) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Candidate only | 13,906 | 1,826 | +1.524 | [+1.471, +1.578] | 0.195 | 1.332 (−2.256, +6.330) |
| 2 | Candidate + centered GC + centered GC² | 13,906 | 1,826 | +1.446 | [+1.391, +1.501] | 0.241 | 1.294 (−3.077, +6.605) |
| 3 | Model 2 + component-depth-4 fixed effects | 13,906 | 1,826 | +1.310 | [+1.258, +1.362] | 0.300 | 1.243 (−2.875, +6.842) |
| 4 | Model 2 + component-depth-6 fixed effects; common-support subset | 12,926 | 1,616 | +1.293 | [+1.241, +1.346] | 0.310 | 1.247 (−2.803, +6.331) |

The candidate coefficient falls by about 5% after the quadratic GC adjustment, by about 14% from the GC model after adding depth-4 components, and changes little with the depth-6 common-support adjustment. Every estimate remains positive and is large relative to its record-clustered standard error. The models support a conditional association in this sample beyond measured GC and broad lineage components.

GC distributions overlap: all ten Fungi GC deciles contain candidate and non-candidate intervals. The earlier candidate-weighted GC-decile comparison reported +1.413 nats/token after GC standardization, consistent with the quadratic model's +1.446. The finer component-depth-6 analysis additionally excludes groups with insufficient candidate/comparator overlap rather than extrapolating to them.

The regression diagnostic summary includes R², residual SD/range, coefficient stability, complete-case sample counts, and record-clustered uncertainty. Residual means are numerically zero by OLS construction. No residual plot or formal functional-form test was conducted; the quadratic GC term is a parsimonious nonlinear adjustment, not proof that GC is modeled perfectly. Sequence length, coding status, and gene type cannot be added because their interval-level values are absent locally. No fine-grained rank adjustment is justified by the unranked lineage source and sparse component categories. Model coefficients and intervals remain descriptive because the same sample was used to discover the candidate, creating selection bias not corrected by these models.

## Exact-sequence redundancy and sensitivity

The duplication trace's counts are based on exact SHA-256 identity of strand-normalized sequence within the full 244,046-key common cohort. Repeated status is count >1. The plot has no sequence hash values, so it supports prevalence and count summaries but not exact grouping of candidate keys by hash.

| Group | Intervals | Repeated-sequence intervals | Repeated fraction | Singleton-count intervals | Mean copies in common cohort |
|---|---:|---:|---:|---:|---:|
| Candidate, all kingdoms | 3,501 | 8 | 0.229% | 3,493 | 1.006 |
| Non-candidate, all kingdoms | 16,499 | 570 | 3.455% | 15,929 | 1.993 |
| Candidate Fungi | 3,419 | 4 | 0.117% | 3,415 | 1.001 |
| Non-candidate Fungi | 10,487 | 50 | 0.477% | 10,437 | 1.012 |
| Candidate Metazoa | 82 | 4 | 4.878% | 78 | 1.195 |

The candidate's 8 repeat-marked intervals occur across 6 `record_id`s; the 4 candidate fungal repeat-marked intervals are from 4 records and four different full lineage strings. Non-candidate fungal repeat-marked intervals number 50 across 42 records. Candidate sequence repetition is therefore uncommon in the relevant fungal comparison, and not concentrated in one source accession. The Metazoa candidate subgroup is only 82 intervals and is not interpreted strongly.

Removing every repeat-marked interval (a conservative singleton-only sensitivity, not “one representative per hash”) gives mean-score differences of +1.364 nats/token overall (3,493 candidate vs 15,929 comparators) and +1.534 within Fungi (3,415 vs 10,437). The within-Fungi difference is effectively unchanged from +1.524 before this exclusion. Duplicate frequency does not materially account for the likelihood association in this sample.

The exact number of distinct sequence hashes within the sample cannot be recovered without hash IDs. The copy-count histogram implies conservative candidate bounds of 3,497–3,501 distinct hashes among 3,501 intervals (3,493 singleton hashes plus 4–8 possible groups among the eight repeated rows); for candidate Fungi the bound is 3,418–3,419. These are bounds, not exact deduplicated counts. A one-deterministic-representative-per-hash analysis and sequence-level lineage concentration cannot be performed. Exact repeats are not near-duplicates or overlapping intervals; those relationships are also unavailable.

## Integrated evidence assessment

1. **Which fungal taxa are represented, and at what resolution?** Exact semicolon-delimited lineages are available for all 3,419 candidate Fungi, with 218 unique complete lineage strings. Literal components show strong representation of Ascomycota, Sordariomycetes, Eurotiomycetes, Eurotiales, Glomerellales, and Hypocreales. Formal phylum-through-species coverage cannot be established because the local source has no rank annotations and component positions shift when lineage terms are omitted. Species-level claims are not supported.
2. **Is the candidate dominated by particular fungal lineages relative to the non-candidate cohort?** Yes descriptively. Component-depth-4 `Ascomycota` is 83.68% candidate Fungi versus 64.35% non-candidate Fungi; `Basidiomycota` is 14.92% versus 30.14%. Several component-depth-6 and -8 labels are also relatively more represented, while `Agaricomycetes`, `Boletales`, and `Suillus` components are less represented. Candidate lineages span 912 accession-like records, but some exact full lineages are concentrated in only a few records. This is not a representative sample of all Fungi.
3. **Does the higher likelihood association persist after available covariate adjustment?** Yes, in this cohort. It remains +1.446 after quadratic GC adjustment, +1.310 after component-depth-4 fixed effects, and +1.293 in the depth-6 common-support sensitivity. Length, coding, gene type, and rank-annotated taxonomy are unavailable, so this is not comprehensive adjustment.
4. **Does redundancy materially affect the result?** The available exact-repeat frequency measure suggests no: candidate repeat-marked intervals are rare, and the within-Fungi score contrast persists after removing all such rows. Exact hash-based deduplication cannot be completed because hash IDs are not present.
5. **Which observations are robust, exploratory, and unresolved?** Exact keys, cohort joins, lineage strings, finite GC/entropy/score availability, and plotted repeat counts are directly validated. The within-sample likelihood association is stable across measured GC and lineage-component adjustments and singleton-only sensitivity, with record-clustered intervals. Taxonomic proportions are descriptive. DBSCAN membership is stable only under the tested radii on this fixed 2D projection. Original-space neighborhood stability, formal rank assignments, biological mechanisms, independent-genome replication, and effects of length/coding/gene type remain unresolved.
6. **What is the strongest defensible conclusion?** The candidate is a raw-UMAP projection region whose sampled intervals have a strongly skewed fungal lineage composition and higher raw mean token log probability. The latter persists after measured GC and broad lineage-component adjustment, and exact-repeat prevalence is too low to explain the pattern by itself. This warrants describing the region as a composition- and taxonomy-associated model-score pattern in this sample, not as a biological population.
7. **What prevents a biological-cluster claim?** The region is defined in a two-dimensional UMAP, which can create, distort, merge, or hide density structure. Raw and whitened PCA/UMAP fits were independent; the candidate is not a distinct whitened component. Original 3,072-dimensional embeddings are absent. Taxonomy is unranked lineage text; sampled intervals may be dependent; exact hash IDs, length, coding, and gene-type values are absent; candidate discovery and evaluation reuse the same cohort. UMAP distances are not evolutionary distances, and higher model likelihood is not biological fitness or functional importance.

## Reproducibility files and stopping point

`final_candidate_case_study.py` reads the specified membership table, keyed composition/likelihood tables, full common-cohort taxonomy CSV, and existing plot08 duplication trace. It writes the compact summary tables:

- `final_candidate_taxonomy_depth_summary.csv`
- `final_candidate_taxonomy_components.csv`
- `final_candidate_exact_lineages.csv`
- `final_candidate_likelihood_models.csv`
- `final_candidate_duplication_summary.csv`

This is the final planned candidate-specific analysis. No new plot was created, no pipeline output or existing visualization was modified, and no GPU enrichment was rerun.
