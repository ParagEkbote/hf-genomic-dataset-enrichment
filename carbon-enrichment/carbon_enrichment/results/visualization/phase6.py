#!/usr/bin/env python3
"""
Analyze heterogeneous candidate_evidence.csv for four taxonomy targets:
Colletotrichum, Aspergillus, Apocrita, and Brachycera.

Outputs:
  schema_audit.csv
  taxon_membership.csv
  taxon_summary.csv
  categorical_distributions.csv
  evidence_quality.csv
  pairwise_taxon_comparisons.csv
  pairwise_embedding_similarity.csv
  analysis_report.txt

Dependencies: pandas, numpy
Usage:
  python analyze_candidate_taxa.py candidate_evidence.csv --output-dir candidate_taxon_analysis
Optional:
  python analyze_candidate_taxa.py candidate_evidence.csv --output-dir out --embedding-pair-sample 500 --seed 42
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TAXA = ("Colletotrichum", "Aspergillus", "Apocrita", "Brachycera")

NUMERIC_CANDIDATES = [
    "start", "end", "begin_of_sequence", "end_of_sequence", "begin_of_gene",
    "end_of_gene", "gc_content", "gc_skew", "sequence_length", "gene_length",
    "shannon_entropy", "taxonomy_depth", "is_coding_region",
    "cpu_source_rows", "cpu_distinct_sequences", "likelihood_source_rows",
    "likelihood_duplicate_key", "mean_log_prob", "mean_log_prob_min",
    "mean_log_prob_max", "sum_log_prob", "sum_log_prob_min", "sum_log_prob_max",
    "perplexity", "perplexity_min", "perplexity_max",
    "supervised_position_count", "supervised_position_count_min",
    "supervised_position_count_max", "min_token_logprob",
    "min_token_logprob_min", "min_token_logprob_max", "argmin_position",
    "argmin_position_min", "argmin_position_max", "per_token_logprob_std",
    "per_token_logprob_std_min", "per_token_logprob_std_max",
    "embedding_source_rows", "embedding_distinct_vectors", "embedding_norm",
]
LIKELIHOOD_METRICS = [
    "mean_log_prob", "perplexity", "supervised_position_count",
    "min_token_logprob", "argmin_position", "per_token_logprob_std",
    "sum_log_prob",
]
SEQUENCE_METRICS = [
    "sequence_length", "gene_length", "gc_content", "gc_skew",
    "shannon_entropy", "taxonomy_depth",
]
CATEGORICAL_FIELDS = [
    "kingdom", "panel", "gene_type", "species_type", "strand", "molecule_type",
    "topology", "taxonomy_domain", "is_coding_region", "qc_flag",
    "candidate_definition",
]


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    """Find a column by case-insensitive exact normalized name."""
    lookup = {normalize_name(c): c for c in df.columns}
    for name in names:
        key = normalize_name(name)
        if key in lookup:
            return lookup[key]
    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def parse_vector(value: Any) -> np.ndarray | None:
    """Parse JSON/Python-list/array-like embedding cells; reject malformed vectors."""
    if is_missing(value):
        return None
    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=False).ravel()
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                break
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
        if parsed is None:
            # Accept a simple whitespace/comma-separated vector as a last resort.
            stripped = raw.strip("[]() ")
            try:
                arr = np.fromstring(stripped.replace(",", " "), sep=" ", dtype=float)
            except ValueError:
                return None
        else:
            try:
                arr = np.asarray(parsed, dtype=float).ravel()
            except (TypeError, ValueError):
                return None
    else:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    # Handles booleans and numeric strings while leaving invalid values as NaN.
    return pd.to_numeric(df[col], errors="coerce")


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else math.nan


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return str(value)


def normalized_sequence(value: Any) -> str | None:
    if is_missing(value):
        return None
    seq = re.sub(r"\s+", "", str(value)).upper()
    return seq or None


def sequence_identity(a: str, b: str) -> float:
    """Positional identity over the shared length; length difference reported separately."""
    n = min(len(a), len(b))
    if n == 0:
        return math.nan
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


def shared_taxonomy_depth(a: Any, b: Any) -> int:
    """Count shared leading taxonomy tokens, splitting common taxonomy delimiters."""
    if is_missing(a) or is_missing(b):
        return 0

    def tokens(value: Any) -> list[str]:
        return [
            re.sub(r"\s+", " ", x.strip()).lower()
            for x in re.split(r"\s*(?:;|\\|>|/)\s*", str(value))
            if x.strip()
        ]

    aa, bb = tokens(a), tokens(b)
    count = 0
    for x, y in zip(aa, bb):
        if x != y:
            break
        count += 1
    return count


def taxon_mask(df: pd.DataFrame, taxon: str, taxonomy_col: str | None,
               kingdom_col: str | None, fallback_cols: list[str]) -> tuple[pd.Series, str]:
    """
    Match taxon as a token in taxonomy first. If taxonomy is absent/empty for a row,
    use selected fallback taxonomy-adjacent columns. Match is case-insensitive and
    boundary-aware, so a genus name is not matched as a substring of another word.
    """
    escaped = re.escape(taxon)
    pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    if taxonomy_col:
        primary = df[taxonomy_col].fillna("").astype(str).map(lambda x: bool(pattern.search(x)))
        tax_empty = df[taxonomy_col].map(is_missing)
    else:
        primary = pd.Series(False, index=df.index)
        tax_empty = pd.Series(True, index=df.index)

    fallback = pd.Series(False, index=df.index)
    used = []
    for col in fallback_cols:
        if col and col != taxonomy_col:
            fallback |= df[col].fillna("").astype(str).map(lambda x: bool(pattern.search(x)))
            used.append(col)

    mask = primary | (tax_empty & fallback)
    rule = f"{taxonomy_col} exact-token match" if taxonomy_col else "taxonomy column unavailable"
    if used:
        rule += f"; fallback only when taxonomy missing: {', '.join(used)}"
    return mask, rule


def add_schema_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        nonmissing = s.notna() & s.astype(str).str.strip().ne("")
        sample_values = s[nonmissing].astype(str).head(3).tolist()
        numeric = pd.to_numeric(s, errors="coerce")
        numeric_valid = int(numeric.notna().sum())
        vector_valid = None
        if "embedding" in normalize_name(col):
            vector_valid = int(s.map(lambda x: parse_vector(x) is not None).sum())
        rows.append({
            "column": col,
            "pandas_dtype": str(s.dtype),
            "row_count": len(s),
            "nonmissing_count": int(nonmissing.sum()),
            "missing_count": int((~nonmissing).sum()),
            "missing_fraction": float((~nonmissing).mean()) if len(s) else math.nan,
            "distinct_nonmissing": int(s[nonmissing].nunique(dropna=True)),
            "numeric_parse_count": numeric_valid,
            "numeric_parse_fraction_of_nonmissing": (
                numeric_valid / int(nonmissing.sum()) if int(nonmissing.sum()) else math.nan
            ),
            "embedding_vector_parse_count": vector_valid,
            "sample_values": " | ".join(sample_values),
        })
    return pd.DataFrame(rows)


def summarize_taxon(df: pd.DataFrame, taxon: str, mask: pd.Series,
                    cols: dict[str, str | None], membership_rule: str) -> dict[str, Any]:
    sub = df.loc[mask].copy()
    out: dict[str, Any] = {
        "taxon": taxon,
        "membership_rule": membership_rule,
        "candidate_rows": int(len(sub)),
    }

    for label, names in [
        ("candidate_key", ("candidate_key",)),
        ("record_id", ("record_id",)),
        ("sequence", ("sequence", "cpu_sequence")),
        ("strand_normalized_sequence", ("strand_normalized_sequence", "cpu_strand_normalized_sequence")),
        ("taxonomy", ("taxonomy", "cpu_taxonomy")),
    ]:
        col = find_col(sub, *names)
        out[f"distinct_{label}"] = int(sub[col].dropna().nunique()) if col else math.nan

    # Report candidate and source integrity fields without assuming one row == one sequence.
    for col_name in [
        "sequence_conflict", "likelihood_duplicate_key", "embedding_conflict",
    ]:
        col = find_col(sub, col_name)
        if col:
            vals = sub[col].astype(str).str.strip().str.lower()
            truthy = vals.isin(["true", "1", "yes", "y"])
            out[f"{col_name}_true_rows"] = int(truthy.sum())
            out[f"{col_name}_nonmissing_rows"] = int((~sub[col].map(is_missing)).sum())
        else:
            out[f"{col_name}_true_rows"] = math.nan
            out[f"{col_name}_nonmissing_rows"] = 0

    for field, col in cols.items():
        if field in SEQUENCE_METRICS or field in LIKELIHOOD_METRICS or field in (
            "embedding_norm", "supervised_position_count", "min_token_logprob",
        ):
            s = numeric_series(sub, col)
            out[f"{field}_n"] = int(s.notna().sum())
            out[f"{field}_median"] = safe_median(s)
            out[f"{field}_mean"] = safe_mean(s)
            if field in ("mean_log_prob", "perplexity", "gc_content", "sequence_length"):
                out[f"{field}_q25"] = float(s.quantile(0.25)) if s.notna().any() else math.nan
                out[f"{field}_q75"] = float(s.quantile(0.75)) if s.notna().any() else math.nan

    coding_col = cols.get("is_coding_region")
    if coding_col:
        s = sub[coding_col].astype(str).str.strip().str.lower()
        yes = s.isin(["true", "1", "yes", "y"])
        no = s.isin(["false", "0", "no", "n"])
        out["coding_status_nonmissing_n"] = int((yes | no).sum())
        out["coding_fraction_among_recognized"] = float(yes.sum() / (yes | no).sum()) if (yes | no).any() else math.nan
    else:
        out["coding_status_nonmissing_n"] = 0
        out["coding_fraction_among_recognized"] = math.nan

    for field in CATEGORICAL_FIELDS:
        col = cols.get(field)
        if col:
            vals = sub[col].dropna().astype(str).str.strip()
            vals = vals[vals.ne("")]
            out[f"{field}_distinct_values"] = int(vals.nunique())
            out[f"{field}_nonmissing_n"] = int(len(vals))
        else:
            out[f"{field}_distinct_values"] = math.nan
            out[f"{field}_nonmissing_n"] = 0

    # Embedding validity / dimensions are assessed independently of the stored embedding_norm column.
    emb_col = cols.get("embedding")
    vectors = []
    dimensions = []
    if emb_col:
        for value in sub[emb_col]:
            vec = parse_vector(value)
            if vec is not None:
                vectors.append(vec)
                dimensions.append(int(vec.size))
    out["valid_embedding_vectors"] = len(vectors)
    out["embedding_dimensions_distinct"] = len(set(dimensions))
    out["embedding_dimension_mode"] = (pd.Series(dimensions).mode().iloc[0] if dimensions else math.nan)
    out["embedding_vector_norm_median_recomputed"] = (
        float(np.median([np.linalg.norm(v) for v in vectors])) if vectors else math.nan
    )

    # Sequence QC, calculated directly from strings where possible.
    seq_col = cols.get("strand_normalized_sequence") or cols.get("sequence")
    seqs = sub[seq_col].map(normalized_sequence) if seq_col else pd.Series(None, index=sub.index)
    valid_seqs = seqs.dropna()
    out["valid_sequence_strings"] = int(len(valid_seqs))
    out["sequence_length_median_recomputed"] = (
        float(valid_seqs.map(len).median()) if len(valid_seqs) else math.nan
    )
    out["sequence_alphabet_non_acgt_fraction"] = (
        float(valid_seqs.map(lambda s: any(c not in "ACGTN" for c in s)).mean())
        if len(valid_seqs) else math.nan
    )
    out["sequence_exact_duplicate_fraction"] = (
        float(1 - valid_seqs.nunique() / len(valid_seqs)) if len(valid_seqs) else math.nan
    )
    return out


def make_distribution(df: pd.DataFrame, masks: dict[str, pd.Series],
                      col: str | None, field_label: str) -> pd.DataFrame:
    if col is None:
        return pd.DataFrame(columns=["taxon", "field", "value", "count", "fraction"])
    rows = []
    for taxon, mask in masks.items():
        s = df.loc[mask, col].fillna("<MISSING>").astype(str).str.strip()
        s = s.replace("", "<MISSING>")
        counts = s.value_counts(dropna=False)
        denom = int(counts.sum())
        for value, count in counts.items():
            rows.append({
                "taxon": taxon, "field": field_label, "value": value,
                "count": int(count), "fraction": float(count / denom) if denom else math.nan,
            })
    return pd.DataFrame(rows)


def quality_table(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    key_col = find_col(df, "candidate_key")
    record_col = find_col(df, "record_id")
    seq_col = find_col(df, "strand_normalized_sequence", "sequence")
    for taxon, mask in masks.items():
        sub = df.loc[mask]
        row = {"taxon": taxon, "rows": len(sub)}
        for label, col in [
            ("candidate_key", key_col), ("record_id", record_col), ("sequence", seq_col),
        ]:
            if col:
                nonmissing = sub[col].dropna()
                row[f"{label}_nonmissing"] = int(len(nonmissing))
                row[f"{label}_distinct"] = int(nonmissing.nunique())
                row[f"{label}_duplicate_excess"] = int(len(nonmissing) - nonmissing.nunique())
            else:
                row[f"{label}_nonmissing"] = 0
                row[f"{label}_distinct"] = math.nan
                row[f"{label}_duplicate_excess"] = math.nan
        for field in ("sequence_conflict", "likelihood_duplicate_key", "embedding_conflict"):
            col = find_col(sub, field)
            if col:
                v = sub[col].astype(str).str.strip().str.lower()
                row[f"{field}_true_rows"] = int(v.isin(["true", "1", "yes", "y"]).sum())
            else:
                row[f"{field}_true_rows"] = math.nan
        for field in ("cpu_source_rows", "likelihood_source_rows", "embedding_source_rows",
                      "cpu_distinct_sequences", "likelihood_duplicate_key",
                      "embedding_distinct_vectors"):
            col = find_col(sub, field)
            if col:
                s = pd.to_numeric(sub[col], errors="coerce")
                row[f"{field}_sum"] = float(s.sum(min_count=1)) if s.notna().any() else math.nan
                row[f"{field}_median"] = safe_median(s)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_taxa_summary(df: pd.DataFrame, masks: dict[str, pd.Series],
                          taxonomy_col: str | None, score_col: str | None,
                          seq_col: str | None) -> pd.DataFrame:
    rows = []
    taxa = list(masks)
    for a, b in combinations(taxa, 2):
        aa, bb = df.loc[masks[a]], df.loc[masks[b]]
        score_a = numeric_series(aa, score_col).dropna() if score_col else pd.Series(dtype=float)
        score_b = numeric_series(bb, score_col).dropna() if score_col else pd.Series(dtype=float)
        seq_a = aa[seq_col].map(normalized_sequence).dropna() if seq_col else pd.Series(dtype=str)
        seq_b = bb[seq_col].map(normalized_sequence).dropna() if seq_col else pd.Series(dtype=str)
        rows.append({
            "taxon_a": a,
            "taxon_b": b,
            "rows_a": len(aa),
            "rows_b": len(bb),
            "median_mean_log_prob_a": safe_median(score_a),
            "median_mean_log_prob_b": safe_median(score_b),
            "median_mean_log_prob_difference_a_minus_b": (
                safe_median(score_a) - safe_median(score_b)
                if len(score_a) and len(score_b) else math.nan
            ),
            "exact_sequence_overlap_count": len(set(seq_a) & set(seq_b)),
            "taxonomy_column_used": taxonomy_col or "",
            "note": "Descriptive, unadjusted comparison; rows may be dependent and group sizes may differ.",
        })
    return pd.DataFrame(rows)


def embedding_pairwise(df: pd.DataFrame, masks: dict[str, pd.Series], emb_col: str | None,
                       max_per_taxon: int, seed: int) -> pd.DataFrame:
    """Compute sampled pairwise cosine similarity; cap each taxon for bounded runtime."""
    if not emb_col:
        return pd.DataFrame(columns=[
            "taxon_a", "taxon_b", "n_vectors_a", "n_vectors_b",
            "n_cosine_pairs", "cosine_similarity_mean", "cosine_similarity_median",
            "cosine_similarity_q05", "cosine_similarity_q95", "dimension",
        ])
    rng = np.random.default_rng(seed)
    parsed: dict[str, list[np.ndarray]] = {}
    for taxon, mask in masks.items():
        vecs = []
        sub = df.loc[mask, emb_col]
        for value in sub:
            v = parse_vector(value)
            if v is not None:
                vecs.append(v)
        # Keep only the modal vector dimension; mixed dimensions cannot be compared.
        if vecs:
            dims = pd.Series([v.size for v in vecs]).value_counts()
            mode_dim = int(dims.index[0])
            vecs = [v for v in vecs if v.size == mode_dim]
        if len(vecs) > max_per_taxon:
            chosen = rng.choice(len(vecs), size=max_per_taxon, replace=False)
            vecs = [vecs[int(i)] for i in chosen]
        parsed[taxon] = vecs

    rows = []
    for a, b in combinations(list(masks), 2):
        va, vb = parsed[a], parsed[b]
        sims = []
        if va and vb:
            dim_a = va[0].size
            dim_b = vb[0].size
            if dim_a == dim_b:
                A = np.vstack(va)
                B = np.vstack(vb)
                an = np.linalg.norm(A, axis=1)
                bn = np.linalg.norm(B, axis=1)
                valid_a = an > 0
                valid_b = bn > 0
                A = A[valid_a] / an[valid_a, None]
                B = B[valid_b] / bn[valid_b, None]
                if len(A) and len(B):
                    # Exact all-pairs for capped sample; chunks avoid one large allocation.
                    for start in range(0, len(A), 128):
                        sims.extend((A[start:start + 128] @ B.T).ravel().tolist())
                dimension = dim_a
            else:
                dimension = math.nan
        else:
            dimension = math.nan
        s = np.asarray(sims, dtype=float)
        rows.append({
            "taxon_a": a, "taxon_b": b,
            "n_vectors_a": len(va), "n_vectors_b": len(vb),
            "n_cosine_pairs": int(s.size),
            "cosine_similarity_mean": float(np.mean(s)) if s.size else math.nan,
            "cosine_similarity_median": float(np.median(s)) if s.size else math.nan,
            "cosine_similarity_q05": float(np.quantile(s, 0.05)) if s.size else math.nan,
            "cosine_similarity_q95": float(np.quantile(s, 0.95)) if s.size else math.nan,
            "dimension": dimension,
            "sampling_cap_per_taxon": max_per_taxon,
            "seed": seed,
            "note": "Cosine similarities are from a deterministic-size random sample when a group exceeds the cap; not inferential p-values.",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Path to candidate_evidence.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("candidate_taxon_analysis"))
    parser.add_argument("--taxa", nargs="+", default=list(DEFAULT_TAXA),
                        help="Taxon names to search (default: four requested taxa)")
    parser.add_argument("--embedding-pair-sample", type=int, default=500,
                        help="Maximum valid vectors sampled per taxon for pairwise cosine comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=0,
                        help="Reserved for future streaming; current analysis loads the CSV into memory")
    args = parser.parse_args()

    if not args.input_csv.exists():
        print(f"ERROR: input file does not exist: {args.input_csv}", file=sys.stderr)
        return 2
    if args.embedding_pair_sample < 1:
        parser.error("--embedding-pair-sample must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.input_csv} ...")
    # keep_default_na allows common missing markers to be interpreted as missing.
    df = pd.read_csv(args.input_csv, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns")

    schema = add_schema_audit(df)
    schema.to_csv(args.output_dir / "schema_audit.csv", index=False)

    taxonomy_col = find_col(df, "taxonomy", "cpu_taxonomy")
    kingdom_col = find_col(df, "kingdom")
    fallback_cols = [
        kingdom_col, find_col(df, "species_type"), find_col(df, "taxonomy_domain"),
        find_col(df, "panel"),
    ]
    masks: dict[str, pd.Series] = {}
    rules: dict[str, str] = {}
    for taxon in args.taxa:
        masks[taxon], rules[taxon] = taxon_mask(
            df, taxon, taxonomy_col, kingdom_col, fallback_cols
        )

    # Build a broad but explicit column map; aliases accommodate prefixed variants.
    cols = {
        "candidate_key": find_col(df, "candidate_key"),
        "record_id": find_col(df, "record_id"),
        "sequence": find_col(df, "sequence", "cpu_sequence"),
        "strand_normalized_sequence": find_col(df, "strand_normalized_sequence", "cpu_strand_normalized_sequence"),
        "taxonomy": taxonomy_col,
        "kingdom": kingdom_col,
        "panel": find_col(df, "panel"),
        "gene_type": find_col(df, "gene_type", "cpu_gene_type"),
        "species_type": find_col(df, "species_type", "cpu_species_type"),
        "strand": find_col(df, "strand", "cpu_strand"),
        "molecule_type": find_col(df, "molecule_type", "cpu_molecule_type"),
        "topology": find_col(df, "topology", "cpu_topology"),
        "taxonomy_domain": find_col(df, "taxonomy_domain", "cpu_taxonomy_domain"),
        "taxonomy_depth": find_col(df, "taxonomy_depth", "cpu_taxonomy_depth"),
        "is_coding_region": find_col(df, "is_coding_region", "cpu_is_coding_region"),
        "qc_flag": find_col(df, "qc_flag", "cpu_qc_flag"),
        "sequence_length": find_col(df, "sequence_length", "cpu_sequence_length"),
        "gene_length": find_col(df, "gene_length", "cpu_gene_length"),
        "gc_content": find_col(df, "gc_content", "cpu_gc_content"),
        "gc_skew": find_col(df, "gc_skew", "cpu_gc_skew"),
        "shannon_entropy": find_col(df, "shannon_entropy", "cpu_shannon_entropy"),
        "mean_log_prob": find_col(df, "mean_log_prob", "likelihood_mean_log_prob"),
        "perplexity": find_col(df, "perplexity", "likelihood_perplexity"),
        "supervised_position_count": find_col(df, "supervised_position_count", "likelihood_supervised_position_count"),
        "min_token_logprob": find_col(df, "min_token_logprob", "likelihood_min_token_logprob"),
        "argmin_position": find_col(df, "argmin_position", "likelihood_argmin_position"),
        "per_token_logprob_std": find_col(df, "per_token_logprob_std", "likelihood_per_token_logprob_std"),
        "sum_log_prob": find_col(df, "sum_log_prob", "likelihood_sum_log_prob"),
        "embedding": find_col(df, "embedding", "embedding_embedding"),
        "embedding_norm": find_col(df, "embedding_norm", "embedding_embedding_norm"),
    }

    membership_parts = []
    summary_rows = []
    for taxon, mask in masks.items():
        part = df.loc[mask].copy()
        part.insert(0, "matched_taxon", taxon)
        part.insert(1, "membership_rule", rules[taxon])
        membership_parts.append(part)
        summary_rows.append(summarize_taxon(df, taxon, mask, cols, rules[taxon]))

    membership = pd.concat(membership_parts, ignore_index=True) if membership_parts else pd.DataFrame()
    membership.to_csv(args.output_dir / "taxon_membership.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "taxon_summary.csv", index=False)

    dist_frames = []
    for field in CATEGORICAL_FIELDS:
        col = cols.get(field)
        if col:
            dist_frames.append(make_distribution(df, masks, col, field))
    if dist_frames:
        pd.concat(dist_frames, ignore_index=True).to_csv(
            args.output_dir / "categorical_distributions.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(args.output_dir / "categorical_distributions.csv", index=False)

    quality_table(df, masks).to_csv(args.output_dir / "evidence_quality.csv", index=False)
    pairwise_taxa_summary(
        df, masks, taxonomy_col, cols.get("mean_log_prob"),
        cols.get("strand_normalized_sequence") or cols.get("sequence")
    ).to_csv(args.output_dir / "pairwise_taxon_comparisons.csv", index=False)

    embedding_pairwise(
        df, masks, cols.get("embedding"), args.embedding_pair_sample, args.seed
    ).to_csv(args.output_dir / "pairwise_embedding_similarity.csv", index=False)

    # Global row-level integrity checks.
    lines = [
        "Candidate taxon evidence analysis",
        "=" * 32,
        f"Input: {args.input_csv.resolve()}",
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns):,}",
        f"Taxonomy column: {taxonomy_col or 'NOT FOUND'}",
        f"Embedding column: {cols.get('embedding') or 'NOT FOUND'}",
        "",
        "Taxon matches (row counts; matches may overlap if taxonomy strings contain multiple target names):",
    ]
    for taxon, mask in masks.items():
        lines.append(f"  {taxon}: {int(mask.sum()):,} rows")
        lines.append(f"    rule: {rules[taxon]}")
    lines += [
        "",
        "Interpretation cautions:",
        "- Taxon matching is a case-insensitive, boundary-aware search in taxonomy. Fallback columns are used only when taxonomy is missing.",
        "- Counts are candidate-evidence rows, not necessarily independent sequences, genomes, species, or biological replicates.",
        "- Summary statistics are descriptive. They do not establish functional significance or taxon-specific biological effects.",
        "- Mean log probability and perplexity depend on model/tokenization/scoring conventions; compare only if scoring settings are consistent.",
        "- GC, sequence length, coding status, taxonomic relatedness, record dependence, and duplicates can confound cross-taxon comparisons.",
        "- Embedding cosine similarities are sampled when needed and are descriptive, not p-values. Invalid vectors and mixed dimensions are excluded.",
        "- Sequence identity uses exact normalized strings for overlap; the pairwise identity measure is not an alignment-based homology estimate.",
        "- If a taxon has zero rows, verify the taxonomy encoding and target rank before interpreting it as biologically absent.",
        "",
        "Files written:",
    ]
    for name in [
        "schema_audit.csv", "taxon_membership.csv", "taxon_summary.csv",
        "categorical_distributions.csv", "evidence_quality.csv",
        "pairwise_taxon_comparisons.csv", "pairwise_embedding_similarity.csv",
    ]:
        lines.append(f"  {name}")
    (args.output_dir / "analysis_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nOutputs saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
