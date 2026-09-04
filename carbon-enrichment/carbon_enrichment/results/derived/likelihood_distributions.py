from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from carbon_enrichment.resources.clickhouse import ClickHouseResource


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LIKELIHOOD_NAME = "likelihood"
DEFAULT_FEATURES_NAME = "features"

KEY_COLUMNS = ("record_id", "start", "end")

FEATURE_ALIASES = {
    "sequence_length": ("sequence_length",),
    "gc_content": ("gc_content", "GC", "gc"),
    "entropy": ("shannon_entropy", "entropy"),
    "is_coding_region": ("is_coding_region",),
}


def _sql_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _source(ch: ClickHouseResource, name: str) -> str:
    return ch.source_expr(name)


def _write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _query_write_csv(ch: ClickHouseResource, sql: str, path: Path) -> None:
    table = ch.query_arrow(sql)
    path.parent.mkdir(parents=True, exist_ok=True)
    pacsv.write_csv(table, path)


def _write_stream(ch: ClickHouseResource, sql: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = 0
    try:
        for batch in ch.stream_arrow_batches(sql):
            if writer is None:
                writer = pq.ParquetWriter(
                    path,
                    batch.schema,
                    compression="zstd",
                )
            writer.write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------

def _columns(ch: ClickHouseResource, dataset_name: str) -> set[str]:
    table = ch.query_arrow(
        f"DESCRIBE TABLE {ch.source_expr(dataset_name)}"
    )
    return {str(x) for x in table.column("name").to_pylist()}


def _pick(columns: set[str], aliases: Iterable[str]) -> str | None:
    for name in aliases:
        if name in columns:
            return name
    return None


def discover_feature_columns(
    ch: ClickHouseResource,
    features_name: str,
) -> dict[str, str | None]:
    cols = _columns(ch, features_name)
    return {
        logical: _pick(cols, aliases)
        for logical, aliases in FEATURE_ALIASES.items()
    }


# ---------------------------------------------------------------------------
# Base relation
# ---------------------------------------------------------------------------

REQUIRED_LIKELIHOOD_COLUMNS = (
    "record_id",
    "start",
    "end",
    "mean_log_prob",
    "perplexity",
    "supervised_position_count",
    "min_token_logprob",
    "argmin_position",
    "per_token_logprob_std",
)

BASE_RELATION_COLUMNS = (
    "record_id",
    "start",
    "end",
    "sequence_length",
    "effective_length",
    "corpus_span",
    "mean_log_prob",
    "perplexity",
    "min_token_logprob",
    "argmin_position",
    "per_token_logprob_std",
    "gc_content",
    "entropy",
    "is_coding_region",
    # Added native dataset metrics
    "local_anomaly_z_score",
    "is_burn_in_artifact",
    "background_mean_log_prob",
    "anomaly_depth_ratio",
)


def _validate_likelihood_schema(ch: ClickHouseResource, likelihood_name: str) -> None:
    cols = _columns(ch, likelihood_name)
    missing = [c for c in REQUIRED_LIKELIHOOD_COLUMNS if c not in cols]
    if missing:
        raise RuntimeError(
            f"Likelihood dataset {likelihood_name!r} is missing required "
            f"column(s): {missing}. Available columns: {sorted(cols)}"
        )


def _validate_base_relation(ch: ClickHouseResource, base_sql: str) -> None:
    table = ch.query_arrow(f"SELECT * FROM ({base_sql}) LIMIT 0")
    got = set(table.column_names)
    expected = set(BASE_RELATION_COLUMNS)
    missing = expected - got
    extra = got - expected
    if missing:
        raise RuntimeError(
            f"build_base_relation() is missing expected column(s): "
            f"{sorted(missing)}."
        )
    if extra:
        raise RuntimeError(
            f"build_base_relation() exposes unexpected column(s): "
            f"{sorted(extra)}."
        )

def build_base_relation(
    ch: ClickHouseResource,
    likelihood_name: str = DEFAULT_LIKELIHOOD_NAME,
    features_name: str | None = DEFAULT_FEATURES_NAME,
) -> str:
    _validate_likelihood_schema(ch, likelihood_name)

    lsrc = _source(ch, likelihood_name)

    if features_name is None:
        return f"""
        SELECT
            record_id,
            start,
            end,
            toFloat64(supervised_position_count) AS sequence_length,
            toFloat64(supervised_position_count) AS effective_length,
            toFloat64(end - start) AS corpus_span,
            mean_log_prob,
            perplexity,
            min_token_logprob,
            argmin_position,
            per_token_logprob_std,
            CAST(NULL AS Nullable(Float64)) AS gc_content,
            CAST(NULL AS Nullable(Float64)) AS entropy,
            CAST(NULL AS Nullable(UInt8)) AS is_coding_region,
            
            if(per_token_logprob_std > 0, 
               (mean_log_prob - min_token_logprob) / per_token_logprob_std, 
               0) AS local_anomaly_z_score,
            
            if(argmin_position <= 50, 1, 0) AS is_burn_in_artifact,
            
            if(toFloat64(supervised_position_count) > 1,
               ((mean_log_prob * toFloat64(supervised_position_count)) - min_token_logprob) / (toFloat64(supervised_position_count) - 1),
               mean_log_prob) AS background_mean_log_prob,
               
            if(mean_log_prob < 0, 
               min_token_logprob / mean_log_prob, 
               CAST(NULL AS Nullable(Float64))) AS anomaly_depth_ratio

        FROM {lsrc}
        WHERE isFinite(mean_log_prob)
          AND isFinite(toFloat64(supervised_position_count))
        """

    fsrc = _source(ch, features_name)
    fc = discover_feature_columns(ch, features_name)

    seq_expr = f"toFloat64(l.supervised_position_count)"

    gc_expr = (
        f"toNullable(toFloat64(f.{_sql_ident(fc['gc_content'])}))"
        if fc["gc_content"]
        else "CAST(NULL AS Nullable(Float64))"
    )

    ent_expr = (
        f"toNullable(toFloat64(f.{_sql_ident(fc['entropy'])}))"
        if fc["entropy"]
        else "CAST(NULL AS Nullable(Float64))"
    )

    coding_expr = (
        f"toNullable(toUInt8(f.{_sql_ident(fc['is_coding_region'])}))"
        if fc["is_coding_region"]
        else "CAST(NULL AS Nullable(UInt8))"
    )

    return f"""
    SELECT
        l.record_id,
        l.start,
        l.end,
        {seq_expr} AS sequence_length,
        toFloat64(l.supervised_position_count) AS effective_length,
        toFloat64(l.end - l.start) AS corpus_span,
        l.mean_log_prob,
        l.perplexity,
        l.min_token_logprob,
        l.argmin_position,
        l.per_token_logprob_std,
        {gc_expr} AS gc_content,
        {ent_expr} AS entropy,
        {coding_expr} AS is_coding_region,
        
        if(l.per_token_logprob_std > 0, 
           (l.mean_log_prob - l.min_token_logprob) / l.per_token_logprob_std, 
           0) AS local_anomaly_z_score,
        
        if(l.argmin_position <= 50, 1, 0) AS is_burn_in_artifact,
        
        if(toFloat64(l.supervised_position_count) > 1,
           ((l.mean_log_prob * toFloat64(l.supervised_position_count)) - l.min_token_logprob) / (toFloat64(l.supervised_position_count) - 1),
           l.mean_log_prob) AS background_mean_log_prob,
           
        if(l.mean_log_prob < 0, 
           l.min_token_logprob / l.mean_log_prob, 
           CAST(NULL AS Nullable(Float64))) AS anomaly_depth_ratio

    FROM {lsrc} AS l
    LEFT JOIN {fsrc} AS f
        ON l.record_id = f.record_id
       AND l.start = f.start
       AND l.end = f.end
    WHERE isFinite(l.mean_log_prob)
      AND isFinite(toFloat64(l.supervised_position_count))
    """

# ---------------------------------------------------------------------------
# Layer 1 — cohort distributions
# ---------------------------------------------------------------------------

def layer1_cohort(
    ch: ClickHouseResource,
    base_sql: str,
    out_dir: Path,
) -> None:
    metrics = """
        sequence_length,
        effective_length,
        mean_log_prob,
        perplexity,
        -min_token_logprob AS token_surprise,
        per_token_logprob_std,
        gc_content,
        entropy,
        local_anomaly_z_score,
        is_burn_in_artifact,
        background_mean_log_prob,
        anomaly_depth_ratio
    """

    _query_write_csv(
        ch,
        f"""
        WITH base AS ({base_sql}),
        x AS (SELECT {metrics} FROM base)
        SELECT
            count() AS n,
            min(sequence_length) AS length_min,
            quantileExact(0.50)(sequence_length) AS length_p50,
            max(sequence_length) AS length_max,

            quantileExact(0.01)(mean_log_prob) AS mean_log_prob_p01,
            quantileExact(0.50)(mean_log_prob) AS mean_log_prob_p50,
            quantileExact(0.99)(mean_log_prob) AS mean_log_prob_p99,

            quantileExact(0.50)(perplexity) AS perplexity_p50,
            quantileExact(0.50)(token_surprise) AS token_surprise_p50,
            quantileExact(0.50)(per_token_logprob_std) AS heterogeneity_p50,

            avg(gc_content) AS gc_mean,
            quantileExact(0.50)(gc_content) AS gc_median,
            avg(entropy) AS entropy_mean,
            
            avg(local_anomaly_z_score) AS local_anomaly_z_mean,
            quantileExact(0.50)(local_anomaly_z_score) AS local_anomaly_z_median,
            
            avg(background_mean_log_prob) AS bg_mean_log_prob_mean,
            quantileExact(0.50)(background_mean_log_prob) AS bg_mean_log_prob_median,
            
            avg(anomaly_depth_ratio) AS anomaly_depth_ratio_mean,
            quantileExact(0.50)(anomaly_depth_ratio) AS anomaly_depth_ratio_median,
            
            sum(is_burn_in_artifact) AS burn_in_artifact_count
        FROM x
        """,
        out_dir / "layer1_cohort_summary.csv",
    )

    _query_write_csv(
        ch,
        f"""
        WITH base AS ({base_sql}),
        b AS
        (
            SELECT
                *,
                ntile(10) OVER (ORDER BY sequence_length ASC) AS length_decile
            FROM base
            WHERE sequence_length > 0
              AND isFinite(sequence_length)
        )
        SELECT
            length_decile,
            count() AS n,
            min(sequence_length) AS length_min,
            quantileExact(0.50)(sequence_length) AS length_median,
            max(sequence_length) AS length_max,
            
            avg(mean_log_prob) AS mean_log_prob_mean,
            quantileExact(0.50)(mean_log_prob) AS mean_log_prob_median,
            
            avg(perplexity) AS perplexity_mean,
            quantileExact(0.50)(per_token_logprob_std) AS heterogeneity_median,
            
            quantileExact(0.50)(local_anomaly_z_score) AS local_anomaly_z_median,
            quantileExact(0.50)(background_mean_log_prob) AS bg_mean_log_prob_median,
            quantileExact(0.50)(anomaly_depth_ratio) AS anomaly_depth_ratio_median,
            sum(is_burn_in_artifact) AS burn_in_artifacts_in_decile
        FROM b
        GROUP BY length_decile
        ORDER BY length_decile
        """,
        out_dir / "layer1_length_deciles.csv",
    )


# ---------------------------------------------------------------------------
# Layer 2 — conditional distributions / residual models
# ---------------------------------------------------------------------------

def _linear_coefficients(ch: ClickHouseResource, base_sql: str) -> tuple[float, float]:
    query = f"""
    SELECT
        avg(x * y) - avg(x) * avg(y) AS cov_xy,
        avg(x * x) - avg(x) * avg(x) AS var_x,
        avg(y) AS mean_y,
        avg(x) AS mean_x
    FROM
    (
        SELECT
            log1p(toFloat64(sequence_length)) AS x,
            toFloat64(mean_log_prob) AS y
        FROM ({base_sql})
        WHERE sequence_length > 0
          AND isFinite(mean_log_prob)
    )
    """

    result = ch.query_arrow(query)

    if result.num_rows == 0:
        raise RuntimeError("Length regression returned no usable rows.")

    row = result.slice(0, 1).to_pylist()[0]
    cov_xy, var_x, mean_x, mean_y = float(row["cov_xy"]), float(row["var_x"]), float(row["mean_x"]), float(row["mean_y"])

    if not math.isfinite(var_x) or var_x <= 0:
        raise RuntimeError("Cannot fit length model: sequence_length has zero variance.")

    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    return slope, intercept


def layer2_length_model(
    ch: ClickHouseResource,
    base_sql: str,
    out_dir: Path,
) -> tuple[float, float]:
    slope, intercept = _linear_coefficients(ch, base_sql)

    _query_write_csv(
        ch,
        f"""
        SELECT
            {slope:.17g} AS length_slope,
            {intercept:.17g} AS length_intercept
        """,
        out_dir / "layer2_length_model.csv",
    )

    _query_write_csv(
        ch,
        f"""
        WITH base AS ({base_sql}),
        scored AS
        (
            SELECT
                *,
                mean_log_prob -
                ({intercept:.17g} + {slope:.17g} * log1p(sequence_length))
                AS length_adjusted_likelihood
            FROM base
            WHERE sequence_length > 0
              AND isFinite(sequence_length)
              AND isFinite(mean_log_prob)
        ),
        ranked AS
        (
            SELECT
                *,
                ntile(10) OVER (ORDER BY sequence_length ASC) AS length_decile
            FROM scored
        )
        SELECT
            length_decile,
            count() AS n,
            min(sequence_length) AS length_min,
            quantileExact(0.50)(sequence_length) AS length_median,
            max(sequence_length) AS length_max,
            
            quantileExact(0.50)(mean_log_prob) AS raw_likelihood_median,
            quantileExact(0.50)(length_adjusted_likelihood) AS adjusted_likelihood_median,
            quantileExact(0.50)(background_mean_log_prob) AS bg_mean_log_prob_median,
            quantileExact(0.50)(local_anomaly_z_score) AS local_anomaly_z_median
        FROM ranked
        GROUP BY length_decile
        ORDER BY length_decile
        """,
        out_dir / "layer2_length_conditional.csv",
    )

    return slope, intercept


def layer2_composition_model(
    ch: ClickHouseResource,
    base_sql: str,
    out_dir: Path,
) -> tuple[float, float, float, float]:
    t = ch.query_arrow(
        f"""
        WITH base AS ({base_sql}),
        b AS
        (
            SELECT
                mean_log_prob AS y,
                log1p(sequence_length) AS x,
                gc_content AS g,
                entropy AS e
            FROM base
            WHERE isFinite(mean_log_prob)
              AND isFinite(sequence_length)
              AND isFinite(gc_content)
              AND isFinite(entropy)
        )
        SELECT
            count() AS n,
            avg(y) AS my, avg(x) AS mx, avg(g) AS mg, avg(e) AS me,
            avg(x*x) AS exx, avg(g*g) AS egg, avg(e*e) AS eee,
            avg(x*g) AS exg, avg(x*e) AS exe, avg(g*e) AS ege,
            avg(x*y) AS exy, avg(g*y) AS egy, avg(e*y) AS eey
        FROM b
        """
    )
    r = t.to_pydict()
    n = float(r["n"][0])

    if n == 0:
        raise RuntimeError("Composition model has no usable rows.")

    my, mx, mg, me = [float(r[k][0]) for k in ("my", "mx", "mg", "me")]
    exx, egg, eee = [float(r[k][0]) for k in ("exx", "egg", "eee")]
    exg, exe, ege = [float(r[k][0]) for k in ("exg", "exe", "ege")]
    exy, egy, eey = [float(r[k][0]) for k in ("exy", "egy", "eey")]

    sxx, sgg, see = exx - mx * mx, egg - mg * mg, eee - me * me
    sxg, sxe, sge = exg - mx * mg, exe - mx * me, ege - mg * me
    sxy, sgy, sey = exy - mx * my, egy - mg * my, eey - me * my

    import numpy as np
    X = np.array([[sxx, sxg, sxe], [sxg, sgg, sge], [sxe, sge, see]], dtype=float)
    b = np.array([sxy, sgy, sey], dtype=float)

    try:
        beta = np.linalg.solve(X, b)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Composition model is singular.") from exc

    bx, bg, be = map(float, beta)
    intercept = my - bx * mx - bg * mg - be * me

    _query_write_csv(
        ch,
        f"""
        SELECT
            {n:.17g} AS n,
            {intercept:.17g} AS intercept,
            {bx:.17g} AS log_length_beta,
            {bg:.17g} AS gc_beta,
            {be:.17g} AS entropy_beta
        """,
        out_dir / "layer2_composition_model.csv",
    )

    return intercept, bx, bg, be


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def layer3_record_metrics(
    ch: ClickHouseResource,
    base_sql: str,
    out_dir: Path,
    length_slope: float,
    length_intercept: float,
    comp_model: tuple[float, float, float, float] | None,
) -> Path:
    if comp_model is not None:
        ci, cl, cg, ce = comp_model
        composition_expr = f"""
            mean_log_prob -
            ({ci:.17g}
             + {cl:.17g} * log1p(sequence_length)
             + {cg:.17g} * gc_content
             + {ce:.17g} * entropy)
        """
    else:
        composition_expr = "CAST(NULL AS Nullable(Float64))"

    sql = f"""
    WITH base AS ({base_sql}),
    scored AS
    (
        SELECT
            *,
            mean_log_prob -
            ({length_intercept:.17g}
             + {length_slope:.17g} * log1p(sequence_length))
                AS length_adjusted_likelihood,

            {composition_expr}
                AS length_composition_adjusted_likelihood,

            -min_token_logprob AS token_surprise,
            argmin_position /
                greatest(effective_length, 1.0)
                AS relative_anomaly_position
        FROM base
    ),
    centers AS
    (
        SELECT
            avg(length_adjusted_likelihood) AS la_mean,
            stddevSamp(length_adjusted_likelihood) AS la_sd,
            quantileExact(0.50)(length_adjusted_likelihood) AS la_med,
            avg(token_surprise) AS ts_mean,
            stddevSamp(token_surprise) AS ts_sd,
            avg(per_token_logprob_std) AS h_mean,
            stddevSamp(per_token_logprob_std) AS h_sd,
            avg(length_composition_adjusted_likelihood) AS ca_mean,
            stddevSamp(length_composition_adjusted_likelihood) AS ca_sd
        FROM scored
    ),
    scales AS
    (
        SELECT
            c.*,
            quantileExact(0.50)(
                abs(s.length_adjusted_likelihood - c.la_med)
            ) AS la_mad
        FROM scored AS s
        CROSS JOIN centers AS c
        GROUP BY
            c.la_mean, c.la_sd, c.la_med,
            c.ts_mean, c.ts_sd,
            c.h_mean, c.h_sd,
            c.ca_mean, c.ca_sd
    ),
    thresholds AS
    (
        SELECT
            quantileExact(0.95)(length_adjusted_likelihood) AS la_p95,
            quantileExact(0.05)(length_adjusted_likelihood) AS la_p05,
            quantileExact(0.95)(length_composition_adjusted_likelihood) AS ca_p95,
            quantileExact(0.05)(length_composition_adjusted_likelihood) AS ca_p05
        FROM scored
    )
    SELECT
        s.record_id,
        s.start,
        s.end,
        s.sequence_length,
        s.effective_length,
        s.corpus_span,
        s.mean_log_prob,
        s.perplexity,
        s.min_token_logprob,
        s.argmin_position,
        s.per_token_logprob_std,
        s.gc_content,
        s.entropy,
        s.is_coding_region,
        
        s.local_anomaly_z_score,
        s.is_burn_in_artifact,
        s.background_mean_log_prob,
        s.anomaly_depth_ratio,

        s.token_surprise,
        s.relative_anomaly_position,

        s.length_adjusted_likelihood,
        if(sc.la_sd > 0, (s.length_adjusted_likelihood - sc.la_mean) / sc.la_sd, 0) AS length_adjusted_z,
        if(sc.la_mad > 0, (s.length_adjusted_likelihood - sc.la_med) / (1.4826 * sc.la_mad), 0) AS length_adjusted_robust_z,

        s.length_composition_adjusted_likelihood,
        if(isNotNull(s.length_composition_adjusted_likelihood) AND sc.ca_sd > 0, 
           (s.length_composition_adjusted_likelihood - sc.ca_mean) / sc.ca_sd, 
           CAST(NULL AS Nullable(Float64))) AS length_composition_adjusted_z,

        if(sc.ts_sd > 0, (s.token_surprise - sc.ts_mean) / sc.ts_sd, 0) AS token_surprise_z,
        if(sc.h_sd > 0, (s.per_token_logprob_std - sc.h_mean) / sc.h_sd, 0) AS heterogeneity_z,

        if(s.length_adjusted_likelihood >= th.la_p95, 1, 0) AS top_5pct_length_adjusted,
        if(s.length_adjusted_likelihood <= th.la_p05, 1, 0) AS bottom_5pct_length_adjusted,

        if(isNotNull(s.length_composition_adjusted_likelihood) AND s.length_composition_adjusted_likelihood >= th.ca_p95, 1, 0) AS top_5pct_length_composition_adjusted,
        if(isNotNull(s.length_composition_adjusted_likelihood) AND s.length_composition_adjusted_likelihood <= th.ca_p05, 1, 0) AS bottom_5pct_length_composition_adjusted

    FROM scored AS s
    CROSS JOIN scales AS sc
    CROSS JOIN thresholds AS th
    """

    path = out_dir / "layer3_record_metrics.csv"
    _query_write_csv(ch, sql, path)
    return path


def run_analysis(
    ch: ClickHouseResource,
    output_dir: str | Path,
    *,
    likelihood_name: str = DEFAULT_LIKELIHOOD_NAME,
    features_name: str | None = DEFAULT_FEATURES_NAME,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_sql = build_base_relation(
        ch,
        likelihood_name=likelihood_name,
        features_name=features_name,
    )
    _validate_base_relation(ch, base_sql)

    config = {
        "likelihood_dataset": likelihood_name,
        "features_dataset": features_name,
    }
    (out_dir / "analysis_config.json").write_text(
        __import__("json").dumps(config, indent=2)
    )

    print("[1/4] Layer 1 cohort distributions...")
    layer1_cohort(ch, base_sql, out_dir)

    print("[2/4] Layer 2 length model...")
    slope, intercept = layer2_length_model(ch, base_sql, out_dir)

    comp_model = None
    if features_name is not None:
        fc = discover_feature_columns(ch, features_name)
        if fc["gc_content"] and fc["entropy"]:
            print("[3/4] Layer 2 length + composition model...")
            comp_model = layer2_composition_model(ch, base_sql, out_dir)
        else:
            print("[3/4] Skipping composition model: GC and/or entropy column not available.")
    else:
        print("[3/4] Skipping composition model: no feature dataset.")

    print("[4/4] Layer 3 adjusted record metrics...")
    layer3_record_metrics(
        ch,
        base_sql,
        out_dir,
        slope,
        intercept,
        comp_model,
    )

    print(f"Done. Results written to: {out_dir}")


def _register_source(ch: ClickHouseResource, name: str, source: str) -> None:
    source_text = str(source)

    if source_text.startswith("https://huggingface.co/datasets/"):
        ch.register_hf_dataset(name, source_text)
        return

    source_path = Path(source_text).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")

    if source_path.is_dir() or source_path.suffix.lower() == ".parquet":
        ch.register_dataset(name, source_path)
        return

    raise ValueError("Local source must be a Parquet file or directory containing Parquet files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Length-aware likelihood distribution analysis."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Likelihood HF Parquet URL/glob or local Parquet file/directory.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional annotation/feature HF URL/glob or local Parquet file/directory; use 'none' to disable.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for CSV analysis results.",
    )

    args = parser.parse_args()
    features = None if args.features in (None, "none") else args.features

    ch = ClickHouseResource()
    _register_source(ch, DEFAULT_LIKELIHOOD_NAME, args.source)

    if features is not None:
        _register_source(ch, DEFAULT_FEATURES_NAME, features)

    run_analysis(
        ch,
        args.output,
        likelihood_name=DEFAULT_LIKELIHOOD_NAME,
        features_name=DEFAULT_FEATURES_NAME if features is not None else None,
    )


if __name__ == "__main__":
    main()