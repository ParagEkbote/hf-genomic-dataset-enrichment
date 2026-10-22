# Quantitative candidate regions in raw-versus-whitened UMAP

## Scope and conclusion

This exploratory analysis uses only the embedded data in `metrics/derived/embedding_visualizations/plot02_umap_raw_vs_whitened.html`, its generator (`carbon-enrichment/carbon_enrichment/results/visualization/phase4.py`), and `run_metadata.json`. No other visualization HTML was opened or analyzed. The decoded traces contain identical 20,000 interval keys. A compact DBSCAN sweep finds density-connected components that are highly stable across nearby radii within each panel. However, candidate components are representation-dependent: the strongest raw component is largely absorbed into one very large whitened component. This supports a reproducible **2D raw-UMAP region** as a follow-up target, not a stable biological population or original-embedding cluster.

**Recommended candidate:** raw panel DBSCAN cluster `raw_C2_e1_m20` (3,501 points). Its membership is exactly unchanged across the three tested radii. Only 119 of these points fall outside the largest whitened cluster; the corresponding whitened clustering does not retain it as a separate density component. Follow-up priority is therefore moderate for investigating what the raw projection is organizing, and low for any claim that it represents a representation-invariant population.

## Data extraction and validation

The HTML's `Plotly.newPlot` payload contains two traces named `Raw` and `Whitened`. Both x and y are Plotly typed-array objects with `dtype: "f4"` and base64 `bdata`. Decoding the bytes as little-endian IEEE-754 float32 yields exactly 80,000 bytes, or 20,000 values, for every x and y array. `customdata` and `text` are ordinary arrays, each of length 20,000; the former has `(record_id, start, end)` and the latter the kingdom label. Coordinates were paired row-for-row with these key and label arrays; no coordinates were approximated.

Validation passed:

- Raw: 20,000 points, 20,000 unique interval keys, all coordinates finite.
- Whitened: 20,000 points, 20,000 unique interval keys, all coordinates finite.
- The raw and whitened key sets are exactly equal (20,000/20,000); no key is duplicated within a panel.
- Coordinate arrays and key/label arrays have aligned lengths and were decoded from the same trace object.

Exact coordinate ranges, in decoded UMAP units:

| Panel | x min | x max | y min | y max |
|---|---:|---:|---:|---:|
| Raw | -6.122071743011475 | 14.031242370605469 | -5.799743175506592 | 10.628189086914062 |
| Whitened | -3.9754838943481445 | 19.633821487426758 | -5.17179012298584 | 11.931591987609863 |

The full decoded point table is `umap_decoded_points.csv`, with panel, coordinates, interval key, and label. `umap_dbscan_memberships.csv` adds the labels for each tested configuration. These tables preserve interval-level tracking across the independently projected panels.

## Clustering method

HDBSCAN, NumPy, SciPy, and scikit-learn are not installed in the current environment, so no new dependency was added. I used standard DBSCAN with Euclidean distance in each panel's own 2D coordinate system, `min_samples=20`, and radii `eps=0.8, 1.0, 1.2` UMAP units. The ±20% radius sweep is a small, explicit local sensitivity check. It is not an optimized or inferential selection of a “correct” radius. A uniform-grid neighbor index (cell width equal to eps) checks adjacent cells and uses exact squared Euclidean distance to include points at distance ≤ eps. Cluster IDs are DBSCAN traversal labels, so IDs are only meaningful within a panel and configuration; memberships, not label numbers, define regions.

Noise means points not assigned to any DBSCAN density-connected component. It is not a separate candidate. This density rule identifies connected high-density regions and can merge distinct visual lobes through bridges; it also labels sparse peripheral observations as noise. It cannot establish a natural cluster boundary.

### Cluster counts and noise

| Panel | eps | Clusters | Noise points | Noise fraction | Cluster sizes, descending |
|---|---:|---:|---:|---:|---|
| Raw | 0.8 | 5 | 27 | 0.135% | 15,479; 3,501; 730; 208; 55 |
| Raw | 1.0 | 5 | 19 | 0.095% | 15,479; 3,501; 738; 208; 55 |
| Raw | 1.2 | 5 | 19 | 0.095% | 15,479; 3,501; 738; 208; 55 |
| Whitened | 0.8 | 6 | 19 | 0.095% | 18,626; 491; 326; 255; 205; 78 |
| Whitened | 1.0 | 6 | 19 | 0.095% | 18,626; 491; 326; 255; 205; 78 |
| Whitened | 1.2 | 6 | 19 | 0.095% | 18,626; 491; 326; 255; 205; 78 |

Raw clusters at eps 1.0 match their nearest components at both neighboring radii with Jaccard 1.000 for four of five clusters and 0.989 for the 738-point cluster at eps 0.8 (the latter has 8 fewer members at eps 0.8). All six whitened clusters have Jaccard 1.000 to a corresponding membership at both neighboring radii. The recommended raw candidate has the same 3,501 keys at all three radii (Jaccard 1.000).

These values measure stability under the particular radius perturbations at fixed `min_samples=20`; they do not show stability to another UMAP fit, another sample, a different metric, or original-space neighborhoods.

## Raw-versus-whitened membership correspondence

Coordinates are not compared directly because PCA and UMAP were fitted independently. Memberships are matched only by interval key. At eps 1.0, the largest whitened component contains 18,626 points. Raw cluster `raw_C2_e1_m20` contains 3,501 points; 3,382 (96.60%) are in that whitened component. Its Jaccard overlap with the whitened component is only 0.180 (intersection 3,382; union 18,745), since the whitened component includes many other keys. The raw region therefore persists as a subset but not as a separate density-defined region in the whitened panel.

For reference, the largest raw component (15,479) overlaps the largest whitened component in 15,059 keys (Jaccard 0.791), while the raw 738-point component's best match is a 326-point whitened component at Jaccard 0.442 (326 shared keys). These results show that similar overall mass can correspond across representations, but the smaller components do not yield a clean one-to-one set of islands. Similar plotted position is not used as correspondence evidence.

## Candidate characterization and ranking

DBSCAN finds other small, parameter-stable components (208 and 55 points in raw; 491, 326, 255, 205, and 78 in whitened). Their small size and/or weak cross-panel component correspondence make them less suitable for an initial follow-up than raw cluster 2. The candidate was chosen from geometry and parameter stability before examining its kingdom counts.

| Candidate | Observable geometry and size | Kingdom composition | Raw/whitened correspondence | Main uncertainty | Priority |
|---|---|---|---|---|---|
| `raw_C2_e1_m20` | 3,501/20,000 (17.505%) DBSCAN members at eps 1.0, min_samples 20; same membership across eps 0.8, 1.0, 1.2. Its raw coordinate bounding box is x=[-1.940, 2.119], y=[-5.800, -1.532] (centroid x=0.095, y=-3.929). It is a separate density-connected component under this rule. | 3,419 Fungi (97.66%); 82 Metazoa (2.34%). Whole-sample descriptive reference: 13,906 Fungi (69.53%) and 6,094 Metazoa (30.47%). This difference is descriptive only; no significance test or adjustment was performed. | 3,382/3,501 (96.60%) are inside the largest whitened component, but it is not a separate whitened component (Jaccard 0.180 against that full component). | DBSCAN connectivity, two-dimensional UMAP distortion, separate PCA/UMAP fits, broad kingdom imbalance, and unmeasured sequence covariates. The original 3,072-D embeddings are unavailable locally; no original-space stability claim is possible. | **Moderate for exploratory follow-up; low for biological interpretation.** |

A broad kingdom difference is not evidence of taxonomic enrichment: kingdom labels are coarse, the sample is not an independent random draw of genomes, and there is no inferential comparison controlling for sequence composition, length, duplication, or sampling structure. Fungi 97.66% is a descriptive property of this DBSCAN membership only.

## Selected follow-up and next visualization question

**Selected region:** `raw_C2_e1_m20`.

**Reproducible membership definition:** from the raw trace, decode x/y from the float32 `bdata` arrays, then run Euclidean DBSCAN with `eps=1.0`, `min_samples=20`; select the component with 3,501 keys that is identical to the corresponding component at eps 0.8 and 1.2. For stable tracking, use the key rows in `umap_dbscan_memberships.csv` where `panel=raw` and `eps_1_min20_cluster=2`. The numeric component ID reflects the current traversal order; the 3,501 interval keys are the definitive membership. The whitened memberships for the same keys are present in the same table.

**Strongest supported observation:** this is a reproducible density-separated region in the existing raw 2D projection across the chosen nearby radii, and almost all its intervals are incorporated into the largest whitened density component rather than remaining a separate whitened region.

**Main competing explanation:** this separation may be induced or exaggerated by raw PCA score geometry and the nonlinear 2D UMAP projection; the whitened view's merging is consistent with sensitivity to preprocessing/projection rather than a robust cluster in the underlying embeddings. Taxonomic composition could also track sequence properties or redundant sampling, none of which is tested here.

**Next existing visualization to inspect:** `plot07_umap_by_composition.html`.

**Question it should answer:** do the selected raw-cluster keys occupy a distinct GC-content or Shannon-entropy range compared with the other sampled intervals, suggesting a simple sequence-composition gradient as a competing explanation for the raw-UMAP separation? This next inspection should use the candidate interval keys rather than infer membership from visual position. A positive association would motivate adjusted composition testing; it would not establish that composition caused the projection structure.

## Interpretation limits

PCA/whitening and cosine UMAP settings are sourced from the generator: randomized 50-component PCA with seed 42; the whitened branch uses `whiten=True`; each branch then receives its own 2D UMAP with cosine metric, 25 neighbors, `min_dist=0.1`, seed 42, and one job. The run metadata reports a 20,000-point projection sample from 244,046 metadata rows and 3,072-dimensional embeddings. The generated plot exposes kingdom labels and interval keys, but no point-level covariates beyond those are embedded in this chart.

UMAP can create, distort, or conceal density separation. Its coordinates and distances are not evolutionary distances. A DBSCAN component is a reproducible outcome of a stated rule on this fixed projection, not proof of a naturally discrete or biologically distinct population. The original embeddings and projection parquet are absent from the available local artifact set, so this report makes no claim of original-space neighborhood stability. No GPU enrichment was rerun, no original HTML/pipeline artifact was modified, and no other visualization HTML was analyzed.
