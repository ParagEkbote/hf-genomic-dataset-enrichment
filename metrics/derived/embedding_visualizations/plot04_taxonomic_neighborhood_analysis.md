# Focused analysis: lineage agreement among embedding neighbors

## Visualization and provenance

`plot04_knn_lineage_agreement.html` plots the fraction of existing embedding-neighbor edges whose two interval records share the entire taxonomy prefix through a given number of lineage components. The x-axis is **number of shared lineage components tested**, not a named rank; the y-axis is **fraction of eligible neighbor edges sharing that full prefix**.

The plotted values are recoverable from `plot04_knn_lineage_agreement.csv`. The visualization generator in `carbon-enrichment/carbon_enrichment/results/visualization/phase4.py` joins both edge endpoints to the common cohort by `(record_id, start, end)`, removes edges where both keys are the same interval, and calculates the agreement for each depth. The generator explicitly describes this as descriptive agreement, not held-out classification accuracy. The run metadata reports 2,510,520 input edges, 15,018 self-matching interval edges excluded, zero missing endpoint keys, zero eligible edges without lineage, and 2,495,502 edges used.

The edge-generation code in `carbon-enrichment/carbon_enrichment/results/derived/embedding_features.py` uses cosine distance and requests the top 10 neighbors per query. Thus the plotted units are **edges**, not independent genomic sequences. Multiple edges share a query, and repeated interval keys or overlapping genomic intervals may introduce further dependence. The original kNN CSV is available at `metrics/derived/phase4/knn_neighbors.csv`; the figure-generation metadata points to a historical absolute path, but the corresponding CSV is present in the repository at this relative path.

## Quantitative observations

| Shared lineage components tested | Agreement | Eligible edges |
|---:|---:|---:|
| 1 | 100.00% | 2,495,502 |
| 2 | 97.32% | 2,495,502 |
| 3 | 92.73% | 2,495,502 |
| 4 | 84.37% | 2,495,502 |
| 5 | 81.54% | 2,495,502 |
| 6 | 61.12% | 2,492,883 |
| 7 | 54.68% | 2,485,337 |
| 8 | 45.70% | 2,478,679 |
| 9 | 42.89% | 2,317,311 |
| 10 | 42.21% | 1,803,031 |
| 11 | 57.30% | 1,030,980 |
| 12 | 70.55% | 574,846 |

Agreement declines from depth 1 through depth 10. This supports a descriptive pattern in which existing neighbors commonly share shallow lineage prefixes, with lower agreement at deeper prefixes among edges eligible at those depths.

The last two values must be interpreted with care: the eligible set falls from 1.80 million edges at depth 10 to 1.03 million at depth 11 and 0.57 million at depth 12. The increase to 57.30% and 70.55% therefore does not show that agreement rises as one moves to finer taxonomy in a fixed cohort. It reflects a much smaller subset of edges whose endpoint lineages are long enough to be evaluated. More generally, the depth-specific denominators mean the curve is not a controlled comparison across identical edge sets.

## What this says about the UMAP observation

In `plot02_umap_raw_vs_whitened.html`, the 20,000 projected intervals are labeled Fungi or Metazoa, and each panel showed high same-label fractions among 10 nearest neighbors in the 2D UMAP coordinates (91.2% raw and 95.9% whitened in the preceding coordinate-based calculation). The present plot adds evidence that the **existing cosine-neighbor graph** also has substantial agreement with broad taxonomy prefixes. This is useful corroboration because it comes from a different representation of neighborhoods than 2D UMAP coordinates.

It still does not establish that individual UMAP islands are taxonomically coherent: plot 04 has no UMAP coordinates or island memberships, and it pools edges across the cohort. It also does not identify a named rank because the lineage components are not mapped to formal rank names in the metadata. The 100% agreement at depth 1 may reflect a highly shared first lineage component in this cohort, rather than fine taxonomic discrimination.

## Interpretation and competing explanations

**Observation:** Existing cosine-neighbor edges are highly likely to share shallow lineage prefixes, and agreement falls substantially through depth 10.

**Plausible interpretation:** The embedding-derived neighbor graph contains broad taxonomic organization in the analyzed interval cohort.

**Alternatives and caveats:** Uneven taxon abundance, shared sequence composition, repeated or overlapping intervals, exact-sequence redundancy, and correlated edges could contribute to this pattern. The summary does not compare against a taxonomy-frequency-preserving null, composition/length-matched neighbors, or a one-record-per-sequence cohort. No confidence intervals or query-level aggregation are provided. These are associations with the supplied taxonomy strings, not evidence that embedding distance is an evolutionary distance or that taxonomy is independently validated.

## Negative findings and limits

- The plot does not identify any specific island, clade, species, or biological mechanism.
- It cannot determine whether taxonomy agreement remains after controlling for GC content, entropy, sequence length, or sequence duplication.
- It provides no rank-specific results because the prefix components are not named ranks.
- It gives no uncertainty estimates or null comparison; do not call the agreement statistically significant based on this plot alone.
- Counts are edge counts, not independent sample sizes. Ten neighbors per query and repeated interval keys can create dependence.
- The deeper-depth increase is not directly comparable with shallower depths because eligible edge counts decline sharply.
- It does not establish held-out classification performance, phylogenetic validity, or evolutionary distance.

## Finding

This plot provides **moderate descriptive evidence** that the cosine-neighbor graph derived from Carbon embeddings reflects broad taxonomic lineage structure in the cohort. It is consistent with the broad label-associated organization seen in `plot02_umap_raw_vs_whitened.html`, but it neither validates individual UMAP islands nor isolates taxonomy from composition, redundancy, and sampling effects. The most informative next test would compare this agreement against a taxonomy-frequency-preserving null and after collapsing exact sequence duplicates, with eligibility and uncertainty reported by named taxonomic rank.
