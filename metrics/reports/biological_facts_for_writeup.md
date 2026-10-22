# Biological facts available for the write-up

## Scope

The binding audit supports descriptive cohort, score, composition and embedding facts. The footer-first retrieval attempt did not produce a complete footer or any sequence. No external annotation was run. The candidate rule is `raw_C2_e1_m20`: raw 2-D UMAP, DBSCAN `eps=1.0`, `min_samples=20`; it is post hoc and sample-specific. It must not be interpreted as a biological class or a full-cohort selection rule.

## Fact table

| Fact | Label | Evidence (file/tool/accession) | n and denominator | Effect size and CI | Caveat |
|---|---|---|---|---|---|
| The projection sample contains 20,000 intervals, of which 3,501 belong to the specified raw-UMAP DBSCAN region; 3,419 of these are labelled Fungi. | [DATA] | `metrics/derived/biological_visualization_analysis/umap_dbscan_memberships.csv`; candidate case study/audit | 3,501/20,000; fungal label 3,419/3,501 | Counts/proportions: 17.505%; 97.66% | Sample-specific, post hoc projection-defined region; not full-cohort prevalence or a biological cluster. |
| Within sampled fungal candidate/comparison rows, the literal Ascomycota lineage component occurs in 2,861/3,419 versus 6,748/10,487; Basidiomycota occurs in 14.92% versus 30.14%. | [DATA] | `reports/biological_evidence_audit.md`; `final_candidate_taxonomy_components.csv` | Fungal candidate n=3,419; comparison n=10,487 | Ascomycota shares 83.68% vs 64.35%; Basidiomycota 14.92% vs 30.14% | Literal lineage components, not rank-resolved within-taxon candidate rates; sample composition only. |
| Candidate fungi have a higher raw mean log probability than sampled noncandidate fungi in the audited in-sample model; the record-clustered estimate is +1.524 nats/token, falling to +1.310 after the available depth-4 component adjustment. | [DATA] | `final_candidate_likelihood_models.csv`; `final_candidate_case_study.md` | 3,419 candidate + 10,487 comparison Fungi; total model n=13,906 | +1.524 (95% CI +1.471 to +1.578); adjusted +1.310 (95% CI details in source table) | In-sample exploratory association; no length/gene class adjustment, composition-only baseline, or held-out validation. Do not call biological enrichment. |
| Across 244,046 common-cohort intervals, raw likelihood has Spearman association ρ=0.228 with GC; associations are weaker for entropy (−0.068) and length (0.074). | [DATA] | `metrics/derived/likelihood_visualizations/08_summary_correlation_matrix.csv` | n=244,046 | Spearman ρ as stated; no CI reported | Unadjusted pairwise associations; raw model score. |
| Exact-sequence repeats have higher raw score medians than singleton rows in the common cohort. | [DATA] | `metrics/derived/likelihood_visualizations/06_duplication_summary.csv` | Singleton n=237,044; 2 copies n=1,034; 3+ copies n=5,968 | Medians: −6.484, −3.255, −2.557 respectively | Exact identity count, not homology; row dependence and unequal groups; candidate fungi include only 4/3,419 repeated rows per audit. |
| No sequence-level homology, protein-domain, repeat annotation, or functional gene-class result is available for the candidate case set in this workspace. | [DATA] | Local artifact inventory; `reports/biological_evidence_audit.md` | Case set n=0; denominator not applicable | Not estimable | Preflight halted because required keyed metadata/rank taxonomy were missing; this is absence of evidence, not evidence of absence. |

## Per-interval case cards

No valid per-interval cards can be reported. `case_set_keys.csv` contains headers only: selection keys were not finalized because remote footer inspection stalled before scalar metadata retrieval, and local joins lack the required matching covariates, sequence hashes, and rank-resolved taxonomy. No sequence data or external annotation was obtained. Length/GC, raw/baseline-adjusted scores, homology, domains and repeats therefore remain unavailable at the required per-key resolution.

## Claim disposition

### Supported now

- Descriptive counts and score/composition summaries shown in the fact table, with their stated sample and denominator.
- The candidate definition and its post hoc, sample-specific status.

### Retained as hypotheses / exploratory

- Any biological interpretation of the candidate region, including taxon-specific selection rates or a functional sequence class. These require a frozen full-cohort or held-out candidate/control rule, rank-resolved taxonomy, and matched sequence-level annotations.
- The candidate likelihood contrast as a biological signal. A disjoint composition-only baseline and independent held-out validation would test its survival.

### Abandoned

- Function, novelty, adaptation, pathogenicity, or contamination claims from taxonomy, likelihood, or embedding proximity: none is established by current artifacts. Sequence/reference annotation and appropriate biological evidence would be required.

## Logs

### Contamination / mismatch

No sequence was retrieved; contamination and sequence/coordinate/hash mismatches were not tested. No discrepancy should be inferred from missing data. Details: `metrics/derived/biological_annotation/contamination_mismatch_log.json`.

### Retrieval audit

See `metrics/derived/biological_annotation/retrieval_report.md` and `reports/retrieval_failures.md`. Three footer-range attempts were made against the first file; none yielded a complete footer. One returned a matching HTTP 206 Content-Range header but did not deliver a complete body. No footer parsed, no row groups or rows were read, and no sequence retrieval occurred. Exact network byte totals cannot be certified because two responses stalled before headers and API-response bytes were not instrumented; the incomplete verified response was bounded to the requested 64 KiB range. The pre-registered selection algorithm is in `retrieval_plan.json`; keys were not frozen.

### Tool/database manifest

No external annotation tools or databases were run. No database accession or release applies. Local evidence sources are listed in the fact table. The audit reports that candidate joins are keyed by `(record_id,start,end)` and lack the needed per-key CPU metadata; see `metrics/derived/biological_visualization_analysis/final_candidate_case_study.md`.
