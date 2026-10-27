# Lineage agreement validation

## Scope and inputs inspected

This targeted analysis tests whether the existing cosine-neighbor lineage agreement exceeds a graph-preserving taxonomy-abundance null and whether exact-sequence deduplication is possible from local artifacts. It does not repeat the general visualization analysis and creates no plots.

Inputs and schemas checked:

- `metrics/derived/phase4/knn_neighbors.csv`: 2,510,520 rows; columns include query and neighbor interval keys, source row indices, neighbor rank, cosine similarity, and cosine distance. The source implementation (`carbon-enrichment/carbon_enrichment/results/derived/embedding_features.py`) uses cosine search and requests up to 10 neighbors per query.
- `metrics/derived/common_cohort/taxon_analysis/common_cohort_taxonomy.csv`: 244,046 data rows; exactly 244,046 unique `(record_id,start,end)` keys; columns are `record_id,start,end,taxonomy`; no empty taxonomy strings or duplicate interval keys. All lines have 5–19 nonempty components.
- `metrics/derived/embedding_visualizations/plot04_knn_lineage_agreement.csv` and `run_metadata.json`.
- Visualization and feature-generation source code, to verify key matching, self-edge rules, rank-position conventions, and the sequence-hash definition.

No joinable interval-level `sequence_hash` artifact or parquet metadata cache is present in the workspace. `phase4.py` defines the hash as `hex(SHA256(cpu.strand_normalized_sequence))`, but its generated `common_cohort_metadata.parquet` is absent. The available duplication summary is aggregate-only and contains no interval-to-hash mapping.

## Analysis 1 — Baseline reproduction

I joined the current common-cohort taxonomy table to both edge endpoints by exact interval key and reproduced the implementation’s rule: exclude an edge if query and neighbor have the same `(record_id,start,end)`; at depth *d*, include an edge only when both lineages have at least *d* nonempty components; count a match when all first *d* components are identical. This reproduces an **edge-weighted** ratio, not a per-query average. Self matches are interval-key matches, even if the source row indices differ. The query-bootstrap unit used below is the query interval key, so outgoing edges from repeated source rows for the same interval remain in one resampling cluster.

The reproduction exactly matches the run metadata on total edges: 2,510,520 read; 15,018 matching-interval self edges excluded; 0 endpoint keys missing; 0 eligible edges without lineage; 2,495,502 retained. There are 243,671 distinct query interval keys represented among retained edges. The current taxonomy file has unique interval keys, so its interval-to-taxonomy mapping is unambiguous.

The reported estimates reproduce exactly through depth 8. At depth 9 and 10, eligible edge counts still match exactly but agreement differs, indicating a mismatch in deeper taxonomy labels between the current taxonomy CSV and the metadata used when the original plot CSV was produced. At depths 11–12, estimates differ only slightly. The original cached joined metadata is not present, so I cannot determine which taxonomy version is authoritative for those deeper labels. I retain both the original plotted values and the current-table recalculation; I do not silently replace one with the other.

| Prefix depth | Original plot CSV | Recalculated from current taxonomy | Difference (percentage points) | Eligible edges (both) |
|---:|---:|---:|---:|---:|
| 1 | 100.00% | 100.00% | 0.000 | 2,495,502 |
| 2 | 97.32% | 97.32% | 0.000 | 2,495,502 |
| 3 | 92.73% | 92.73% | 0.000 | 2,495,502 |
| 4 | 84.37% | 84.37% | 0.000 | 2,495,502 |
| 5 | 81.54% | 81.54% | 0.000 | 2,495,502 |
| 6 | 61.12% | 61.12% | 0.000 | 2,492,883 |
| 7 | 54.68% | 54.68% | 0.000 | 2,485,337 |
| 8 | 45.70% | 45.70% | 0.000 | 2,478,679 |
| 9 | 42.89% | 42.74% | −0.147 | 2,317,311 |
| 10 | 42.21% | 40.57% | −1.641 | 1,803,031 |
| 11 | 57.30% | 57.29% | −0.005 | 1,030,980 |
| 12 | 70.55% | 70.55% | −0.005 | 574,846 |

The strong change in eligible counts at deeper prefixes is material: depth-specific values do not describe a fixed population of edges. The increase at depths 11–12 is not evidence of increasing agreement in a constant cohort.

## Analysis 2 — Exact-sequence deduplication

**A valid deduplicated graph cannot be reconstructed from the available files.** The edge CSV has interval keys and source row indices, but no sequence hashes. The common-cohort taxonomy CSV has no sequence or hash column. The hash definition is available in source, but the interval-level result of that hash calculation is not. Aggregate copy-category summaries do not identify which endpoints share a hash and cannot resolve one hash mapping to multiple intervals or taxa.

Therefore these requested quantities are not estimable here: number of unique sequence hashes; taxonomy conflicts per hash; deduplicated edge and query counts; and deduplicated agreement by depth. No deduplicated result is imputed from category totals. The minimum additional input is a table with one row per common-cohort interval key containing `sequence_hash` (computed with the pipeline’s strand-normalized SHA-256 definition) and taxonomy. If multiple source rows map to an interval key, that table must also preserve or document the conflict-resolution rule. With it, one can test whether each hash has a single lineage, collapse each hash to one representative, and rebuild the graph over unique sequence identities; simply deleting repeated edges would be invalid.

## Analysis 3 — Taxonomy-abundance null

I ran a **node-label permutation with the directed neighbor graph fixed**. Each interval node retains its outgoing and incoming graph edges. Whole lineage labels are shuffled among nodes within strata that preserve:

- exact lineage length, so each depth has exactly the same eligible edges in every permutation;
- exact outgoing degree after self-edge removal; and
- incoming-degree decile.

This preserves observed lineage abundance within the degree/lineage-length strata, the graph’s query and neighbor degree distributions, and edge dependence induced by the fixed graph. It tests whether neighbors share more lineage prefixes than expected from the observed label frequencies and these graph-structure controls. It does not test independence from sequence composition, length, or exact duplication; those covariates are not available in the joinable local metadata. The null also does not condition on higher-order graph communities or genome/record identity.

The random seed was **20261021**, with **499 permutations**. The one-sided empirical p-value is `(1 + number of permuted estimates >= observed)/(499 + 1)`. For depths 2–12, no permuted statistic equaled or exceeded the observed estimate, so the minimum resolvable p-value is **0.002**. Holm correction across all 12 depth tests gives adjusted p=**0.024** for each of depths 2–12; depth 1 has p=1 because its observed agreement is fixed at 100%. The finite permutation count limits tail precision; p=0.002 should be read as “at or below this resolution,” not as a precise tail probability.

## Analysis 4 — Query-level uncertainty

For each of **1,000 bootstrap replicates** (seed **20261020**), I sampled the 243,671 represented query interval keys with replacement and retained all outgoing eligible edges for every sampled query. The edge-weighted ratio was recomputed from pooled match and eligible-edge counts. This keeps edges from a query together and avoids an edge-independent Bernoulli interval. It does not account for sequence identities shared across different query interval keys because the sequence-hash join is missing, nor does it remove dependence induced by neighbor nodes shared across queries.

The table below gives current-taxonomy point estimates, query-bootstrap percentile 95% intervals, and the degree-stratified abundance-null mean and central 95% permutation range. The full-precision 12-depth table is saved in `lineage_agreement_comparison.csv` and `lineage_validation_statistics.csv`.

| Depth | Published estimate | Current estimate (query-bootstrap 95% CI) | Null mean (central 95% permutation range) | Current minus null | Holm p (12 depths) | Deduplicated estimate |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 100.00% | 100.00% (100.00–100.00%) | 100.00% (100.00–100.00%) | 0.00 pp | 1.000 | Not estimable |
| 3 | 92.73% | 92.73% (92.64–92.80%) | 76.34% (76.28–76.40%) | +16.39 pp | 0.024 | Not estimable |
| 5 | 81.54% | 81.54% (81.42–81.66%) | 48.22% (48.14–48.31%) | +33.32 pp | 0.024 | Not estimable |
| 8 | 45.70% | 45.70% (45.55–45.86%) | 21.48% (21.43–21.53%) | +24.22 pp | 0.024 | Not estimable |
| 10 | 42.21% | 40.57% (40.39–40.76%) | 10.75% (10.69–10.81%) | +29.82 pp | 0.024 | Not estimable |

For the 10th prefix, the published value is shown alongside the current-table recalculation because they disagree. Null comparison and bootstrap interval use the current table and should not be conflated with an exact reanalysis of the historical taxonomy join.

## Analysis 5 — Rank interpretation

The plot-generation code itself labels x as prefix depth and explicitly avoids naming ranks. However, a separate pipeline schema in `carbon-enrichment/carbon_enrichment/results/derived/distributions.py` defines the positional order as domain, kingdom, subkingdom, phylum, subphylum, class, subclass, order, suborder, family, genus; example taxonomy strings in the current cohort follow this order. I therefore report depths 1–11 as those **pipeline positional ranks**, while retaining prefix depth as the primary label because the lineage-agreement plot did not embed rank annotations. Depth 12 has no named rank in that schema. The relevant eligible edge counts are the depth counts in the tables above; at named genus depth 11, only 1,030,980 edges are eligible. No rank name is extrapolated to depth 12.

Taxonomic neighborhood agreement is not evidence of phylogenetic validity or evolutionary distance.

## Original, deduplicated, and null-adjusted result summary

- **Original:** Shallow and intermediate prefix agreement is high in the published plot. Baseline reproduction is exact through depth 8; deeper labels have a measurable source discrepancy at depths 9–10.
- **Deduplicated:** Not estimable. The interval-to-sequence-hash map is absent; conflict counts and deduplicated edge/query counts cannot be determined.
- **Null-adjusted:** On the current taxonomy table, depths 2–12 exceed the degree- and lineage-depth-stratified abundance null by 13.27–55.91 percentage points. The one-sided permutation results remain below Holm-adjusted 0.05 across the 12 depth tests, but are conditional on this null design and have minimum raw p resolution 0.002.

## Negative findings and unresolved limitations

1. The requested exact-sequence deduplication test is blocked by missing interval-level SHA-256 values. The number of unique hashes and number of hashes with conflicting taxonomy cannot be reported.
2. The current taxonomy table does not reproduce the historical agreement values at prefix depths 9 and 10, despite matching eligible-edge counts. The original joined metadata/taxonomy snapshot is needed to resolve this discrepancy.
3. The abundance null controls lineage frequencies, lineage length, fixed graph topology, exact outdegree, and approximate indegree strata. It does not control GC, entropy, sequence length, genome identity, sequence similarity, or taxon-dependent graph communities.
4. Query bootstrap intervals account for multiple outgoing edges per query interval, but not exact sequence dependence between distinct interval keys or shared-neighbor dependence across queries.
5. The null comparison is a randomization test under its specified exchangeability assumptions; it does not establish generalization to other genomic populations.
6. Ranks 1–11 have a positional mapping in another pipeline module, but the plotted artifact intentionally reports only prefix depth. Depths 9–10 have the taxonomy-source discrepancy noted above.

## Conclusion

The evidence for taxonomic organization is **strengthened against an explanation based only on taxon abundance and graph degree**: on the current taxonomy table, neighbor-prefix agreement at depths 2–12 is well above a fixed-graph, degree- and lineage-depth-stratified label-permutation null, with Holm-adjusted p=0.024. The result is **not yet shown to persist after exact-sequence deduplication**, because the required interval-to-hash mapping is missing. The baseline mismatch at depths 9–10 also requires the original joined taxonomy snapshot before deeper-rank estimates can be considered fully reproduced. The defensible conclusion is broad taxonomic structure beyond this abundance null, with redundancy and taxonomy-version sensitivity unresolved.

## Reproducibility artifacts

- `lineage_agreement_comparison.csv`: original versus current-table reproduction, query-bootstrap intervals, null summaries, multiplicity-adjusted p-values, and explicit unavailable deduplicated fields.
- `baseline_reproduction_current_taxonomy.csv`: edge-weighted recalculation at all 12 depths.
- `lineage_validation_statistics.csv`: query-bootstrap and permutation summary statistics at all depths.

Random seeds: bootstrap 20261020; node-label permutation 20261021. Bootstrap replicates: 1,000. Permutations: 499.
