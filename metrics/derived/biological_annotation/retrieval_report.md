# Retrieval report

Status: **aborted: footer range body stalled; retry ceiling reached**.

Candidate definition: `raw_C2_e1_m20` (raw 2-D UMAP, DBSCAN `eps=1.0`, `min_samples=20`, cluster 2), post hoc and sample-specific. Hugging Face resolved CPU repository revision `0ebca2601fde5be3d5d66f309deba7f08e8c0c69`; the stats API revision was fetched but was not persisted before interruption. The CPU API file listing succeeded and indicated 649 Parquet files, but the listing itself was not persisted.

Three footer range attempts were made for the first file. Two stalled before HTTP response headers. The third returned HTTP 206 and a matching `Content-Range`, then stalled before a complete footer body arrived. The app received 0 complete range-body bytes; up to 65,536 bytes from the verified final range may have arrived below the application read layer. API response byte counts were not instrumented, so total Tier 0 byte compliance cannot be certified. No footer parsed, shard was confirmed examined, row group queried, or row/sequence data read. Tier 2 reads: zero. No further attempt was made after the third attempt.

The script was updated to use explicit redirect resolution, streamed range responses, response-header validation, and interrupt-aware reporting. A future run should add transport-level byte accounting and a bounded read timeout to the signed-object range response before resuming; any further retry requires a fresh run with a verified accounting mechanism. Embeddings and `kmer_frequency_vector` were not accessed.
