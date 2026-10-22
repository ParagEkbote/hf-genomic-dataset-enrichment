# Candidate Retrieval and Biological Evidence Audit

- Candidate interval keys: **3501**
- Invalid candidate rows excluded: **0**
- Found in CPU dataset: **3501**
- Found in likelihood dataset: **3501**
- Found in embedding dataset: **3501**
- Found in all three datasets: **3501**
- Pairwise comparisons enabled: **False**
- Pairwise comparisons generated: **0**

## Interpretation constraints

1. `start` and `end` are corpus coordinates in the supplied schemas. Their difference is not assumed to equal nucleotide sequence length.
2. Duplicate-key reports must be reviewed. Likelihood duplicates are summarized with means and ranges; conflicting sequence/vector values are not silently selected.
3. Sequence identity and embedding cosine similarity are comparative signals, not standalone proof of duplication, homology, function, or evolutionary mechanism.
4. Taxonomy is inherited from the CPU dataset. Missing taxonomic lineage is not inferred.
5. Inspect `retrieval_audit.csv`, `duplicate_key_audit.csv`, and `candidate_pairwise_comparisons.csv` before making biological claims.

## Scan performance

| source | shards total | phase-2 shards | rows | fetch s |
|---|---|---|---|---|
| cpu | 649 | 649 | 3,501 | 629.695 |
| likelihood | 146 | 146 | 3,590 | 64.984 |
| embeddings | 146 | 146 | 3,590 | 74.036 |

## Stage timings (s)

| stage | seconds |
|---|---|
| load_candidates | 0.10 |
| discover_shards | 0.45 |
| write_key_file | 0.01 |
| schemas | 2.87 |
| phase1_find_hit_shards | 560.71 |
| phase2_fetch_rows | 630.12 |
| duplicate_audit | 0.04 |
| collapse_sources | 27.83 |
| integrate_and_write_evidence | 8.44 |
| pairwise | 0.00 |

## Outputs

- `candidate_keys_frozen.csv`: frozen candidate keys and retained input metadata
- `candidate_keys_for_clickhouse.parquet`: key relation used by ClickHouse filters
- `retrieval_audit.csv`: per-key availability across all three datasets
- `duplicate_key_audit.csv` and `duplicate_keys_*.csv`: duplicate-key evidence
- `retrieved_*_raw.parquet`: matching source rows (with `source_shard`) before duplicate handling
- `candidate_evidence.parquet` / `.csv`: integrated candidate-level evidence
- `candidate_pairwise_comparisons.csv`: sequence, embedding, and taxonomy comparisons
- `shard_index.json`: shards that held hits, reused on later runs
- `timings.json`: per-stage timings
- `pipeline_summary.json`: machine-readable run summary
