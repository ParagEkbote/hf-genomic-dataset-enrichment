# Likelihood analysis of raw UMAP candidate `raw_C2_e1_m20`

## Conclusion

The candidate has a noticeably higher (less negative) raw mean log probability than non-candidate intervals in the same 20,000-point sample. The difference persists within Fungi and remains large after comparing candidate and non-candidate Fungi within GC deciles. Kingdom mixture and GC therefore do not account for the observed likelihood shift in this sample. This adds evidence that the candidate is associated with model-scoring behavior beyond its broad kingdom and GC composition, but it does not establish that likelihood explains the UMAP geometry or that the region is biologically distinct. The score is not Markov-corrected, and length, coding/gene type, finer taxonomy, record-level dependence, and exact duplication remain unresolved.

## Data lineage and join validation

Candidate membership was recovered from `umap_dbscan_memberships.csv` for `panel=raw`, `eps_1_min20_cluster=2` (eps 1.0, min_samples 20). It contains 3,501 unique `(record_id, start, end)` keys. Kingdom counts, verified from the keyed raw UMAP sample, are 3,419 Fungi and 82 Metazoa. The non-candidate comparison is the other 16,499 keys in that same 20,000-interval raw UMAP sample.

The source for `plot06_umap_by_likelihood.html` is `plot_6_likelihood` in `phase4.py`. It drops rows with missing `mean_log_prob`, plots raw UMAP x/y, colors by `mean_log_prob`, and includes `record_id`, `start`, `end`, and taxonomy in Plotly `customdata`. Thus the plot exposes interval keys directly; no point-order or coordinate-based identity inference was needed. The trace contains 20,000 rows and 20,000 unique keys, all with finite scores. Its key set exactly equals the raw plot02 key set: 0 missing raw-sample keys and 0 extra keys. Coordinates agree with plot02 for every key within 1e-5, and lineage-derived kingdom labels agree for every key. There are no duplicate likelihood plot keys. Candidate and comparison likelihood missingness are both 0/3,501 and 0/16,499 respectively.

The likelihood source is the `AINovice2005/carbon-likelihood-stats` dataset. The GPU source function `_extract_likelihood_stats` computes token log probabilities from logits using `logsumexp`, masks non-content positions, and calculates `mean_log_prob = sum_log_prob / supervised_position_count`. `logsumexp` is the natural logarithm, so this is a mean natural-log probability in nats per supervised token. Higher (less negative) means the model assigned greater average probability to the supervised tokens; it is a model score, not a biological fitness or quality measure. In phase4, duplicate likelihood rows are collapsed by averaging `mean_log_prob` per interval key before joining; run metadata reports 7,251 repeated likelihood keys, maximum within-key spread 0.00930, and no missing likelihood keys in the full common cohort. The score in plot06 is raw and not composition- or Markov-corrected.

GC content was joined from the existing keyed composition table `umap_candidate_composition_join.csv`. Its keys equal the same 20,000 raw sample keys with no duplicates or missing GC values. This makes the GC-stratified comparisons a same-cohort analysis.

## Overall and kingdom-stratified results

Mean and SD are followed by median [Q1, Q3] and P05–P95. Mean difference is candidate minus non-candidate. A positive value means higher average model probability in the candidate. SMD is the pooled-SD standardized mean difference. AUC is the probability a randomly drawn candidate has a higher score than a randomly drawn non-candidate, with ties half-counted; rank-biserial is `2*AUC - 1`. KS D is the maximum absolute difference between empirical cumulative distributions. These effect summaries describe distribution shifts and overlap; they are not tests of biological importance.

| Stratum | Candidate n; missing | Candidate mean ± SD; median [Q1, Q3]; P05–P95 | Non-candidate n; missing | Non-candidate mean ± SD; median [Q1, Q3]; P05–P95 | Mean difference (row-bootstrap 95% CI) | Median difference | SMD | AUC (rank-biserial) | KS D |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| All | 3,501; 0 | -5.143 ± 1.081; -5.233 [-5.992, -4.391]; [-6.732, -3.239] | 16,499; 0 | -6.379 ± 1.569; -6.784 [-7.621, -5.504]; [-8.162, -3.112] | +1.236 [+1.193, +1.281] | +1.551 | +0.827 | 0.765 (+0.530) | 0.468 |
| Fungi | 3,419; 0 | -5.143 ± 1.070; -5.227 [-5.988, -4.391]; [-6.722, -3.267] | 10,487; 0 | -6.667 ± 1.407; -7.099 [-7.824, -5.763]; [-8.224, -3.890] | +1.524 [+1.480, +1.569] | +1.872 | +1.144 | 0.807 (+0.613) | 0.543 |
| Metazoa | 82; 0 | -5.153 ± 1.507; -5.414 [-6.333, -4.351]; [-6.892, -1.823] | 6,012; 0 | -5.876 ± 1.703; -6.344 [-7.141, -5.107]; [-7.829, -2.396] | +0.723 [+0.425, +1.044] | +0.930 | +0.425 | 0.662 (+0.325) | 0.288 |

The Metazoa estimate is based on only 82 candidate intervals and is included as a cautious descriptive result, not a stable subgroup conclusion. Across all points and within Fungi, score distributions are shifted substantially but still overlap; for example the candidate and non-candidate Fungi P05–P95 ranges overlap broadly. This is a distribution shift, not a clean score-defined partition.

## GC comparison within Fungi

To assess GC confounding transparently without fitting a high-dimensional regression, I divided all 13,906 Fungi in the 20,000-point sample into ten equal-count GC bins using the Fungi GC empirical deciles. Each candidate interval was compared with non-candidate Fungi in its same GC bin. All 3,419 candidate Fungi had at least one non-candidate comparator in their bin (100% candidate coverage). Candidate counts by bin ranged from 132 to 442, and non-candidate counts from 949 to 1,259.

| Fungi GC bin (ascending) | Candidate n | Non-candidate n | Candidate mean log probability | Non-candidate mean log probability | Difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 132 | 1,259 | -5.349 | -6.879 | +1.530 |
| 2 | 221 | 1,169 | -5.749 | -7.222 | +1.473 |
| 3 | 304 | 1,087 | -5.593 | -7.088 | +1.495 |
| 4 | 315 | 1,075 | -5.315 | -6.911 | +1.596 |
| 5 | 344 | 1,047 | -5.365 | -6.738 | +1.374 |
| 6 | 382 | 1,002 | -5.136 | -6.573 | +1.437 |
| 7 | 428 | 969 | -5.073 | -6.458 | +1.386 |
| 8 | 427 | 963 | -4.951 | -6.335 | +1.384 |
| 9 | 424 | 967 | -4.830 | -6.112 | +1.282 |
| 10 | 442 | 949 | -4.732 | -6.083 | +1.351 |

Standardizing the non-candidate Fungi score to the candidate's GC-bin distribution gives a candidate mean of -5.143 and a GC-standardized non-candidate mean of -6.556: adjusted difference +1.413 nats per supervised token. A percentile 95% bootstrap interval is [+1.365, +1.456]. The bootstrap independently resampled candidate and non-candidate rows within each GC bin and weighted bin contrasts by candidate counts; bin cutpoints and candidate weights were held fixed. The result is similar across every bin, so it is not driven by only one GC range. This is a coarse, transparent adjustment; it does not control for other correlated sequence properties or finer taxonomy.

## Uncertainty

The overall, Fungi, and Metazoa mean-difference intervals are percentile intervals from 1,000 seeded row-bootstrap replicates (seed 15213), independently resampling candidate and comparator intervals. The GC-standardized interval uses the within-bin resampling described above (1,000 replicates). These intervals quantify variation under an interval-level resampling scheme only. Intervals may overlap, share a record/genome, or represent duplicate sequences; independence has not been established, and row-level intervals can be too narrow. The plot contains interval keys but not source-cluster or sequence-family bootstrap strata. No hypothesis tests or p values were used; no claim rests on a small p value.

## Answers and interpretation

1. **Does the candidate have a distinct likelihood distribution overall?** It has a higher-scoring, shifted distribution (mean +1.236, median +1.551 nats/token; AUC 0.765), with substantial overlap. “Distinct” here is descriptive and does not mean a separate biological class.
2. **Does the difference persist within Fungi?** Yes. The mean difference is +1.524, with SMD 1.144 and AUC 0.807. The candidate contains 3,419 Fungi, so this estimate is much better supported by count than the Metazoa subgroup.
3. **Does it persist after accounting for GC?** Yes in a ten-bin Fungi GC comparison. All decile contrasts are positive (+1.282 to +1.596), and the candidate-weighted standardized difference is +1.413 [bootstrap 95% CI +1.365, +1.456].
4. **Does likelihood add evidence beyond known composition and kingdom associations?** It adds a descriptive association not accounted for by broad kingdom or coarse GC stratification in this sample. That supports testing the candidate's model-score association further; it does not establish that likelihood drives UMAP separation. GC, kingdom, and likelihood may still be jointly related to unmeasured sequence features.
5. **What competing explanations remain?** Sequence length, coding status, gene type, finer taxonomic composition, duplicate-sequence frequency, source/record effects, overlap and dependence among intervals, embedding preprocessing, and nonlinear UMAP projection. The raw region is not a separate component in whitened UMAP, and original embeddings are unavailable locally. Raw mean log probability is also not a Markov-corrected score.
6. **What single analysis should follow?** Estimate the candidate association with mean log probability in a prespecified within-Fungi model that includes a flexible GC term and sequence length, coding status/gene type, and finer lineage where available, with uncertainty clustered by source record (and an exact-sequence sensitivity analysis if keyed data are available). This would test whether the residual likelihood shift survives the most relevant remaining covariates. Those interval-level covariates and cluster identifiers are not present in the local likelihood plot, so the current report does not fit that model.

## Reproducibility artifacts and limits

`umap_candidate_likelihood_join.csv` contains the exact same-cohort keys, candidate flag, kingdom, GC, likelihood, and raw coordinates. `umap_candidate_likelihood_summary.csv` contains the summary/effect table. `umap_candidate_likelihood_gc_strata.csv` contains the Fungi GC-bin counts and contrasts. `umap_candidate_likelihood_gc_adjusted.csv` records the candidate-weighted standardized estimate and bootstrap interval. `umap_candidate_likelihood_analysis.py` documents extraction, key validation, and calculations.

No existing visualization or pipeline output was modified, no new plot was created, and no GPU job was run. The original embedding matrix is unavailable, so these results do not test original-space neighborhoods or demonstrate biological meaning or causality.
