from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq
from carbon_enrichment.resources.clickhouse import ClickHouseResource, ClickHouseConfig

EMBEDDINGS_URL = "https://huggingface.co/datasets/AINovice2005/carbon-embeddings/resolve/main/*.parquet"
LIKELIHOOD_URL = "https://huggingface.co/datasets/AINovice2005/carbon-likelihood-stats/resolve/main/*.parquet"
CPU_URL = "https://huggingface.co/datasets/AINovice2005/carbon-pilot-corpus-dedup/resolve/main/*.parquet"

DEFAULT_OUTPUT = Path("results/case_study/phase45")
DEFAULT_TAXON_INDEX = -1
DEFAULT_OUTLIER_Z = 3.0
DEFAULT_TOP_DIMS = 10
DEFAULT_MIN_TAXON_N = 20
LENGTH_ROBUST_SCALE_DIVISOR = 1.349
DEFAULT_MIN_BOUNDARY_DIST = 50
DEFAULT_ENTROPY_MIN = 1.5
DEFAULT_GC_BIO_MIN = 0.25
DEFAULT_GC_BIO_MAX = 0.70
DEFAULT_PERPLEXITY_BIO_MIN = 50.0
DEFAULT_PERPLEXITY_BIO_MAX = 4000.0
RECURRENT_EMBEDDING_DIMS = [2985, 1427, 585, 1491, 1895]
COL_START = "`start`"
COL_END = "`end`"
STREAM_BATCH_ROWS = 200_000

def sql_quote(v: str) -> str: return "'" + v.replace("'", "''") + "'"
def source_for(r, n): return r.source_expr(n)
def table_to_parquet(t, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, p, compression="zstd")
def _csv_safe_table(t):
    arrs, flds = [], []
    for i, f in enumerate(t.schema):
        col = t.column(i)
        if pa.types.is_list(f.type) or pa.types.is_large_list(f.type):
            vals = col.to_pylist()
            svals = [json.dumps(v, separators=(",", ":")) if v is not None else None for v in vals]
            arrs.append(pa.array(svals, type=pa.string()))
            flds.append(pa.field(f.name, pa.string()))
        else:
            arrs.append(col); flds.append(f)
    return pa.Table.from_arrays(arrs, schema=pa.schema(flds))
def table_to_csv(t, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    pc.write_csv(_csv_safe_table(t), p)
def parquet_to_csv_streaming(parq, csv, batch_rows=STREAM_BATCH_ROWS):
    csv.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parq)
    w = None
    try:
        for b in pf.iter_batches(batch_size=batch_rows):
            st = _csv_safe_table(pa.Table.from_batches([b]))
            if w is None: w = pc.CSVWriter(csv, st.schema)
            w.write_table(st)
    finally:
        if w: w.close()
def inspect_columns(ch, src):
    tbl = ch.query_arrow(f"DESCRIBE TABLE {src}")
    return {str(x) for x in tbl.column("name").to_pylist()}

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4.5 bio v3 fixed")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--taxon-index", type=int, default=DEFAULT_TAXON_INDEX)
    p.add_argument("--outlier-z", type=float, default=DEFAULT_OUTLIER_Z)
    p.add_argument("--top-dims", type=int, default=DEFAULT_TOP_DIMS)
    p.add_argument("--min-taxon-n", type=int, default=DEFAULT_MIN_TAXON_N)
    p.add_argument("--content-length-min", type=int, default=300)
    p.add_argument("--content-length-max", type=int, default=5000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-memory-bytes", type=int, default=0)
    p.add_argument("--stream-batch-rows", type=int, default=STREAM_BATCH_ROWS)
    p.add_argument("--min-boundary-distance", type=int, default=DEFAULT_MIN_BOUNDARY_DIST)
    p.add_argument("--entropy-min", type=float, default=DEFAULT_ENTROPY_MIN)
    p.add_argument("--gc-bio-min", type=float, default=DEFAULT_GC_BIO_MIN)
    p.add_argument("--gc-bio-max", type=float, default=DEFAULT_GC_BIO_MAX)
    p.add_argument("--bio-perplexity-min", type=float, default=DEFAULT_PERPLEXITY_BIO_MIN)
    p.add_argument("--bio-perplexity-max", type=float, default=DEFAULT_PERPLEXITY_BIO_MAX)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort_path = Path("/teamspace/studios/this_studio/hf-genomic-dataset-enrichment/results/derived/common_cohort/common_cohort.csv")
    if not cohort_path.exists(): raise FileNotFoundError(cohort_path)
    tbl = pc.read_csv(cohort_path)
    keys = set(zip(tbl["record_id"].to_pylist(), tbl["start"].to_pylist(), tbl["end"].to_pylist()))
    print(f"Keys {len(keys):,} -> {args.output}")
    sorted_keys = sorted(keys, key=lambda k: (k[0], k[1], k[2]))
    keys_table = pa.Table.from_arrays(
        [pa.array([k[0] for k in sorted_keys]), pa.array([k[1] for k in sorted_keys], type=pa.int64()), pa.array([k[2] for k in sorted_keys], type=pa.int64())],
        schema=pa.schema([("record_id", pa.string()), ("start", pa.int64()), ("end", pa.int64())]),
    )
    keys_path = args.output / "_cohort_keys.parquet"
    record_path = args.output / "_record_metrics.parquet"
    table_to_parquet(keys_table, keys_path)
    started = time.time()
    try:
        with ClickHouseResource(ClickHouseConfig(threads=args.threads)) as ch:
            ch.register_hf_dataset("embeddings", EMBEDDINGS_URL)
            ch.register_hf_dataset("likelihood", LIKELIHOOD_URL)
            ch.register_hf_dataset("cpu", CPU_URL)
            ch.register_dataset("cohort_keys", keys_path)
            embeddings = source_for(ch, "embeddings")
            likelihood = source_for(ch, "likelihood")
            cpu = source_for(ch, "cpu")
            cohort_keys = source_for(ch, "cohort_keys")
            lcols = inspect_columns(ch, likelihood)
            ccols = inspect_columns(ch, cpu)
            has_token = "per_token_logprob" in lcols
            boundary_candidates = [("gene_start","gene_end"),("gene_start_position","gene_end_position"),("begin_of_gene_position","end_of_gene_position")]
            boundary_pair = next((p for p in boundary_candidates if p[0] in ccols and p[1] in ccols), None)
            taxonomy_expr = f"arrayElement(splitByChar(';', taxonomy), {args.taxon_index + 1})" if args.taxon_index != -1 else "arrayElement(splitByChar(';', taxonomy), -1)"
            token_fields = ",length(per_token_logprob) AS token_profile_length ,arrayMin(per_token_logprob) AS token_profile_min ,arrayMax(per_token_logprob) AS token_profile_max ,arrayAvg(per_token_logprob) AS token_profile_mean" if has_token else ""
            boundary_fields = f",toFloat64({boundary_pair[0]}) AS gene_start_position ,toFloat64({boundary_pair[1]}) AS gene_end_position" if boundary_pair else ""
            top_dims = max(1, int(args.top_dims))
            if boundary_pair:
                final_boundary = ",gene_start_position, gene_end_position, (argmin_position >= gene_start_position AND argmin_position <= gene_end_position) AS argmin_inside_gene_boundary, least(abs(toFloat64(argmin_position)-gene_start_position), abs(toFloat64(argmin_position)-gene_end_position)) AS distance_to_nearest_gene_boundary, 'exact_numeric_gene_boundaries' AS boundary_method"
            else:
                final_boundary = ",CAST(NULL AS Nullable(Float64)) AS gene_start_position, CAST(NULL AS Nullable(Float64)) AS gene_end_position, CAST(NULL AS Nullable(UInt8)) AS argmin_inside_gene_boundary, CAST(NULL AS Nullable(Float64)) AS distance_to_nearest_gene_boundary, 'sequence_boundary_proxy_only' AS boundary_method"
            mem = f", max_memory_usage = {args.max_memory_bytes}" if args.max_memory_bytes>0 else ""
            rec_dims_sql = str(RECURRENT_EMBEDDING_DIMS)

            sql = f"""
            WITH
            cohort AS (SELECT record_id, {COL_START}, {COL_END} FROM {cohort_keys}),
            filtered_cpu AS (
                SELECT record_id, {COL_START}, {COL_END}, sequence_length, gc_content, shannon_entropy, toUInt8(is_coding_region) AS is_coding_region, gene_length, taxonomy, {taxonomy_expr} AS taxon_group {boundary_fields}
                FROM {cpu}
                PREWHERE record_id IN (SELECT record_id FROM cohort)
                WHERE (record_id, {COL_START}, {COL_END}) IN (SELECT record_id, {COL_START}, {COL_END} FROM cohort)
            ),
            filtered_likelihood AS (
                SELECT record_id, {COL_START}, {COL_END}, mean_log_prob, sum_log_prob, perplexity, supervised_position_count, min_token_logprob, argmin_position, per_token_logprob_std {token_fields}
                FROM {likelihood}
                PREWHERE record_id IN (SELECT record_id FROM cohort)
                WHERE (record_id, {COL_START}, {COL_END}) IN (SELECT record_id, {COL_START}, {COL_END} FROM cohort)
            ),
            filtered_embeddings AS (
                SELECT record_id, {COL_START}, {COL_END}, embedding_norm,
                    if(length(embedding)=0, CAST([], 'Array(UInt16)'), arrayMap(x -> tupleElement(x,2), arrayResize(arrayPartialReverseSort(x -> tupleElement(x,1), {top_dims}, arrayZip(arrayMap(x -> abs(x), embedding), arrayEnumerate(embedding))), {top_dims}))) AS top_embedding_dimensions,
                    if(length(embedding)=0, CAST([], 'Array(Float32)'), arrayMap(x -> tupleElement(x,1), arrayResize(arrayPartialReverseSort(x -> tupleElement(x,1), {top_dims}, arrayZip(arrayMap(x -> abs(x), embedding), arrayEnumerate(embedding))), {top_dims}))) AS top_embedding_abs_values
                FROM {embeddings}
                PREWHERE record_id IN (SELECT record_id FROM cohort)
                WHERE (record_id, {COL_START}, {COL_END}) IN (SELECT record_id, {COL_START}, {COL_END} FROM cohort)
            ),
            base AS (
                SELECT cpu.record_id, cpu.{COL_START}, cpu.{COL_END}, cpu.sequence_length, cpu.gc_content, cpu.shannon_entropy, cpu.is_coding_region, cpu.gene_length, cpu.taxonomy, cpu.taxon_group,
                       lk.mean_log_prob, lk.sum_log_prob, lk.perplexity, lk.supervised_position_count, lk.min_token_logprob, lk.argmin_position, lk.per_token_logprob_std
                       {", lk.token_profile_length" if has_token else ""} {", lk.token_profile_min" if has_token else ""} {", lk.token_profile_max" if has_token else ""} {", lk.token_profile_mean" if has_token else ""}
                       ,emb.embedding_norm, emb.top_embedding_dimensions, emb.top_embedding_abs_values
                       {", cpu.gene_start_position" if boundary_pair else ""} {", cpu.gene_end_position" if boundary_pair else ""}
                FROM filtered_cpu AS cpu INNER JOIN filtered_likelihood AS lk USING (record_id, {COL_START}, {COL_END}) INNER JOIN filtered_embeddings AS emb USING (record_id, {COL_START}, {COL_END})
            ),
            stats AS (
                SELECT avg(gc_content) AS gc_mu, stddevPop(gc_content) AS gc_sd, avg(sequence_length) AS length_mu, stddevPop(sequence_length) AS length_sd,
                       quantileTDigest(0.50)(sequence_length) AS length_median, quantileTDigest(0.25)(sequence_length) AS length_q1, quantileTDigest(0.75)(sequence_length) AS length_q3,
                       avg(perplexity) AS ppl_mu, stddevPop(perplexity) AS ppl_sd, avg(embedding_norm) AS emb_mu, stddevPop(embedding_norm) AS emb_sd,
                       avg(shannon_entropy) AS entropy_mu, stddevPop(shannon_entropy) AS entropy_sd, avg(min_token_logprob) AS min_lp_mu, stddevPop(min_token_logprob) AS min_lp_sd, avg(per_token_logprob_std) AS token_std_mu, stddevPop(per_token_logprob_std) AS token_std_sd
                FROM base
            ),
            taxon_stats AS (
                SELECT taxon_group AS ts_taxon_group, avg(gc_content) AS taxon_gc_mu, stddevPop(gc_content) AS taxon_gc_sd, avg(perplexity) AS taxon_ppl_mu, stddevPop(perplexity) AS taxon_ppl_sd, avg(embedding_norm) AS taxon_emb_mu, stddevPop(embedding_norm) AS taxon_emb_sd
                FROM base GROUP BY taxon_group HAVING count() >= {args.min_taxon_n}
            ),
            scored AS (
                SELECT
                    b.*,
                    ts.taxon_gc_mu, ts.taxon_gc_sd, ts.taxon_ppl_mu, ts.taxon_ppl_sd, ts.taxon_emb_mu, ts.taxon_emb_sd,
                    s.gc_mu, s.gc_sd, s.length_mu, s.length_sd, s.length_median, s.length_q1, s.length_q3, s.ppl_mu, s.ppl_sd, s.emb_mu, s.emb_sd, s.entropy_mu, s.entropy_sd, s.min_lp_mu, s.min_lp_sd, s.token_std_mu, s.token_std_sd,
                    if(s.gc_sd=0, 0., (b.gc_content - s.gc_mu)/s.gc_sd) AS gc_z,
                    if(s.length_sd=0, 0., (b.sequence_length - s.length_mu)/s.length_sd) AS length_z,
                    if((s.length_q3 - s.length_q1)=0, 0., (b.sequence_length - s.length_median)/((s.length_q3 - s.length_q1)/{LENGTH_ROBUST_SCALE_DIVISOR})) AS length_z_robust,
                    if(s.ppl_sd=0, 0., (b.perplexity - s.ppl_mu)/s.ppl_sd) AS perplexity_z,
                    if(s.emb_sd=0, 0., (b.embedding_norm - s.emb_mu)/s.emb_sd) AS embedding_norm_z,
                    if(s.entropy_sd=0, 0., (b.shannon_entropy - s.entropy_mu)/s.entropy_sd) AS entropy_z,
                    if(s.min_lp_sd=0, 0., (b.min_token_logprob - s.min_lp_mu)/s.min_lp_sd) AS min_token_logprob_z,
                    if(s.token_std_sd=0, 0., (b.per_token_logprob_std - s.token_std_mu)/s.token_std_sd) AS token_std_z,
                    if(ts.taxon_gc_sd=0 OR ts.taxon_gc_sd IS NULL, 0., (b.gc_content - ts.taxon_gc_mu)/ts.taxon_gc_sd) AS taxon_gc_z,
                    if(ts.taxon_ppl_sd=0 OR ts.taxon_ppl_sd IS NULL, 0., (b.perplexity - ts.taxon_ppl_mu)/ts.taxon_ppl_sd) AS taxon_perplexity_z,
                    if(ts.taxon_emb_sd=0 OR ts.taxon_emb_sd IS NULL, 0., (b.embedding_norm - ts.taxon_emb_mu)/ts.taxon_emb_sd) AS taxon_embedding_norm_z
                FROM base AS b
                LEFT JOIN taxon_stats AS ts ON b.taxon_group = ts.ts_taxon_group
                CROSS JOIN stats AS s
            ),
            enriched AS (
                SELECT *,
                    (abs(gc_z)+abs(length_z)+abs(perplexity_z)+abs(embedding_norm_z))/4.0 AS multi_layer_anomaly_score,
                    (abs(gc_z)+abs(perplexity_z)+abs(embedding_norm_z))/3.0 AS content_anomaly_score,
                    (abs(gc_z)+abs(length_z_robust)+abs(perplexity_z)+abs(embedding_norm_z))/4.0 AS multi_layer_anomaly_score_robust,
                    (abs(taxon_gc_z)+abs(taxon_perplexity_z)+abs(taxon_embedding_norm_z))/3.0 AS taxon_aware_content_score,
                    abs(gc_z) >= {args.outlier_z} AS is_gc_outlier,
                    abs(length_z) >= {args.outlier_z} AS is_length_outlier,
                    abs(length_z_robust) >= {args.outlier_z} AS is_length_outlier_robust,
                    abs(perplexity_z) >= {args.outlier_z} AS is_likelihood_outlier,
                    abs(embedding_norm_z) >= {args.outlier_z} AS is_embedding_outlier,
                    (toUInt8(abs(gc_z) >= {args.outlier_z})+toUInt8(abs(length_z) >= {args.outlier_z})+toUInt8(abs(perplexity_z) >= {args.outlier_z})+toUInt8(abs(embedding_norm_z) >= {args.outlier_z})) AS outlier_layer_count,
                    if(sequence_length=0, NULL, argmin_position/toFloat64(sequence_length)) AS normalized_argmin_position,
                    least(greatest(toFloat64(argmin_position),0.0), greatest(toFloat64(sequence_length-1),0.0)) AS bounded_argmin_position,
                    least(abs(toFloat64(argmin_position)), abs(toFloat64(sequence_length - argmin_position))) AS distance_to_sequence_boundary,
                    (abs(gc_z)+abs(length_z)+abs(perplexity_z))/3.0 AS non_embedding_anomaly,
                    (abs(perplexity_z)+abs(min_token_logprob_z)+abs(token_std_z))/3.0 AS likelihood_signal,
                    (abs(gc_z)+abs(length_z)+abs(entropy_z)+abs(toFloat64(is_coding_region)-0.5)*2)/4.0 AS cpu_signal,
                    abs(embedding_norm_z) AS embedding_signal,
                    count(*) OVER (PARTITION BY taxon_group) AS taxon_group_n,
                    percent_rank() OVER (PARTITION BY taxon_group ORDER BY gc_content) AS raw_intra_taxon_gc_percentile,
                    percent_rank() OVER (PARTITION BY taxon_group ORDER BY sequence_length) AS raw_intra_taxon_length_percentile,
                    percent_rank() OVER (PARTITION BY taxon_group ORDER BY perplexity) AS raw_intra_taxon_perplexity_percentile,
                    percent_rank() OVER (PARTITION BY taxon_group ORDER BY embedding_norm) AS raw_intra_taxon_embedding_norm_percentile
                FROM scored
            ),
            guarded AS (
                SELECT *,
                    if(taxon_group_n >= {args.min_taxon_n}, raw_intra_taxon_gc_percentile, NULL) AS intra_taxon_gc_percentile,
                    if(taxon_group_n >= {args.min_taxon_n}, raw_intra_taxon_length_percentile, NULL) AS intra_taxon_length_percentile,
                    if(taxon_group_n >= {args.min_taxon_n}, raw_intra_taxon_perplexity_percentile, NULL) AS intra_taxon_perplexity_percentile,
                    if(taxon_group_n >= {args.min_taxon_n}, raw_intra_taxon_embedding_norm_percentile, NULL) AS intra_taxon_embedding_norm_percentile,
                    distance_to_sequence_boundary < {args.min_boundary_distance} AS is_edge_artifact,
                    shannon_entropy < {args.entropy_min} AS is_low_entropy,
                    (gc_content < {args.gc_bio_min} OR gc_content > {args.gc_bio_max}) AS is_extreme_gc_bio,
                    (perplexity < {args.bio_perplexity_min} OR perplexity > {args.bio_perplexity_max}) AS is_perplexity_artifact,
                    hasAny(top_embedding_dimensions, {rec_dims_sql}) AS has_recurrent_embedding_dim,
                    (content_anomaly_score * if(distance_to_sequence_boundary < {args.min_boundary_distance}, 0.1, 1.) * if(shannon_entropy < {args.entropy_min}, 0.2, 1.)) AS bio_content_score
                FROM enriched
            )
            SELECT record_id, {COL_START}, {COL_END}, sequence_length, gc_content, shannon_entropy, is_coding_region, gene_length, taxonomy, taxon_group, taxon_group_n,
                   mean_log_prob, sum_log_prob, perplexity, supervised_position_count, min_token_logprob, argmin_position, per_token_logprob_std
                   {", token_profile_length" if has_token else ""} {", token_profile_min" if has_token else ""} {", token_profile_max" if has_token else ""} {", token_profile_mean" if has_token else ""}
                   ,embedding_norm, gc_z, length_z, length_z_robust, perplexity_z, embedding_norm_z, entropy_z, min_token_logprob_z, token_std_z,
                   taxon_gc_z, taxon_perplexity_z, taxon_embedding_norm_z,
                   multi_layer_anomaly_score, multi_layer_anomaly_score_robust, content_anomaly_score, taxon_aware_content_score, bio_content_score,
                   is_gc_outlier, is_length_outlier, is_length_outlier_robust, is_likelihood_outlier, is_embedding_outlier, outlier_layer_count,
                   normalized_argmin_position, bounded_argmin_position, distance_to_sequence_boundary,
                   is_edge_artifact, is_low_entropy, is_extreme_gc_bio, is_perplexity_artifact, has_recurrent_embedding_dim,
                   non_embedding_anomaly, likelihood_signal, cpu_signal, embedding_signal,
                   intra_taxon_gc_percentile, intra_taxon_length_percentile, intra_taxon_perplexity_percentile, intra_taxon_embedding_norm_percentile,
                   top_embedding_dimensions, top_embedding_abs_values
                   {", gene_start_position" if boundary_pair else ""} {", gene_end_position" if boundary_pair else ""} {final_boundary}
            FROM guarded
            SETTINGS max_threads={args.threads}, max_bytes_before_external_group_by=1073741824, max_bytes_before_external_sort=1073741824,
                     input_format_parquet_use_native_reader_v3=1, input_format_parquet_filter_push_down=1, input_format_parquet_bloom_filter_push_down=1, input_format_parquet_dictionary_filter_push_down=1 {mem}
            """
            print("Executing v3...")
            qs=time.time()
            writer=None; total=0
            try:
                for batch in ch.stream_arrow_batches(sql):
                    if batch.num_rows==0: continue
                    bt=pa.Table.from_batches([batch])
                    if writer is None: writer=pq.ParquetWriter(record_path, bt.schema, compression="zstd")
                    writer.write_table(bt); total+=bt.num_rows
            finally:
                if writer: writer.close()
            print(f"Rows {total} in {time.time()-qs:.1f}s")
            if total==0: raise RuntimeError("Zero rows")
            parquet_to_csv_streaming(record_path, args.output/"record_metrics.csv")
            summary_sql = f"SELECT count() AS n, avg(bio_content_score) AS bio_mean, avg(taxon_aware_content_score) AS taxon_mean, sum(toUInt64(is_edge_artifact)) AS edge_art, sum(toUInt64(is_low_entropy)) AS low_ent, sum(toUInt64(has_recurrent_embedding_dim)) AS recur FROM file({sql_quote(str(record_path))}, Parquet)"
            table_to_csv(ch.query_arrow(summary_sql), args.output/"metric_summary.csv")
            top_sql = f"SELECT record_id, {COL_START}, {COL_END}, taxon_group, sequence_length, gc_content, perplexity, multi_layer_anomaly_score, bio_content_score, taxon_aware_content_score, distance_to_sequence_boundary, is_edge_artifact, has_recurrent_embedding_dim, boundary_method FROM file({sql_quote(str(record_path))}, Parquet) ORDER BY multi_layer_anomaly_score DESC LIMIT 1000"
            table_to_csv(ch.query_arrow(top_sql), args.output/"top_anomalies.csv")
            content_sql = f"SELECT record_id, {COL_START}, {COL_END}, taxon_group, sequence_length, gc_content, perplexity, content_anomaly_score, bio_content_score, taxon_aware_content_score, gc_z, taxon_gc_z, distance_to_sequence_boundary, is_edge_artifact, boundary_method FROM file({sql_quote(str(record_path))}, Parquet) WHERE sequence_length >= {args.content_length_min} AND sequence_length <= {args.content_length_max} AND taxon_group_n >= {args.min_taxon_n} ORDER BY content_anomaly_score DESC LIMIT 1000"
            table_to_csv(ch.query_arrow(content_sql), args.output/"top_content_anomalies.csv")
            bio_sql = f"SELECT record_id, {COL_START}, {COL_END}, taxon_group, sequence_length, gc_content, perplexity, shannon_entropy, content_anomaly_score, bio_content_score, taxon_aware_content_score, taxon_gc_z, taxon_perplexity_z, intra_taxon_perplexity_percentile, is_coding_region, distance_to_sequence_boundary, boundary_method, top_embedding_dimensions FROM file({sql_quote(str(record_path))}, Parquet) WHERE sequence_length >= {args.content_length_min} AND sequence_length <= {args.content_length_max} AND taxon_group_n >= {args.min_taxon_n} AND distance_to_sequence_boundary >= {args.min_boundary_distance} AND shannon_entropy >= {args.entropy_min} AND gc_content BETWEEN {args.gc_bio_min} AND {args.gc_bio_max} AND perplexity BETWEEN {args.bio_perplexity_min} AND {args.bio_perplexity_max} AND NOT has_recurrent_embedding_dim ORDER BY bio_content_score DESC, taxon_aware_content_score DESC LIMIT 1000"
            bt=ch.query_arrow(bio_sql)
            table_to_csv(bt, args.output/"top_biological_content_anomalies.csv")
            print(f"Bio rows {bt.num_rows}")
            high_gc_sql = f"SELECT record_id, {COL_START}, {COL_END}, taxon_group, sequence_length, gc_content, gc_z, taxon_gc_z, intra_taxon_gc_percentile, is_coding_region FROM file({sql_quote(str(record_path))}, Parquet) WHERE gc_content > 0.70 AND taxon_group_n >= {args.min_taxon_n} AND intra_taxon_gc_percentile > 0.95 ORDER BY gc_z DESC LIMIT 1000"
            table_to_csv(ch.query_arrow(high_gc_sql), args.output/"top_high_gc_islands.csv")
            noncode_sql = f"SELECT record_id, {COL_START}, {COL_END}, taxon_group, sequence_length, gc_content, perplexity, bio_content_score, distance_to_sequence_boundary FROM file({sql_quote(str(record_path))}, Parquet) WHERE is_coding_region=0 AND taxon_group_n >= {args.min_taxon_n} AND distance_to_sequence_boundary >= {args.min_boundary_distance} AND shannon_entropy >= {args.entropy_min} ORDER BY bio_content_score DESC LIMIT 1000"
            table_to_csv(ch.query_arrow(noncode_sql), args.output/"top_non_coding_anomalies.csv")
            (args.output/"run_metadata.json").write_text(json.dumps({"phase":"4.5_bio_v3","bio_filters":{"dist":args.min_boundary_distance,"ent":args.entropy_min,"gc":[args.gc_bio_min,args.gc_bio_max],"ppl":[args.bio_perplexity_min,args.bio_perplexity_max]},"elapsed":time.time()-started},indent=2))
            keys_path.unlink(missing_ok=True); record_path.unlink(missing_ok=True)
            print("Done"); return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1

if __name__ == "__main__": sys.exit(main())