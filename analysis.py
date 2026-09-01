#!/usr/bin/env python3
"""
Case 4.5 KNN analysis

Analyzes candidate_screening.knn.csv without modifying the screening/KNN
pipeline.

Outputs:
  <prefix>.candidates.csv
      One row per candidate with uniqueness, taxonomy, distance, and
      taxonomic-discordance metrics.

  <prefix>.neighbors.csv
      One row per candidate-neighbor relationship with rank, distance,
      duplicate status, shared taxonomy prefix, and broad lineage flags.

  <prefix>.summary.txt
      Human-readable ranked summary.

Important taxonomy behavior
----------------------------
The source taxonomy strings are hierarchical semicolon-separated paths, but
they may contain intermediate clades and do not guarantee fixed positions for
kingdom/phylum/class/order/family/genus. Therefore this script does NOT infer
taxonomic ranks by position.

Instead:
  * "same genus" is detected conservatively using the final path component
    when it is genus-like.
  * "same kingdom" uses recognized broad lineage labels when present.
  * shared_prefix_depth measures exact shared hierarchy from the root and is
    rank-independent.
  * taxonomic_discordance is based on shared-prefix divergence, not guessed
    rank positions.

This makes Steps 1–3 safe with the taxonomy information currently present in
the KNN CSV. A later rank-aware taxonomy mapping can add exact family/order/
class/phylum fields without changing this script's other outputs.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path

import pandas as pd


BROAD_RANKS = {
    "kingdom": {
        "Bacteria",
        "Archaea",
        "Eukaryota",
        "Viruses",
        "Virus",
        "Viridiplantae",
        "Plantae",
        "Fungi",
        "Metazoa",
        "Animalia",
    },
    "phylum": {
        "Arthropoda",
        "Chordata",
        "Mollusca",
        "Cnidaria",
        "Echinodermata",
        "Porifera",
        "Nematoda",
        "Annelida",
        "Platyhelminthes",
        "Rotifera",
        "Tardigrada",
        "Bryozoa",
        "Brachiopoda",
        "Hemichordata",
        "Ascomycota",
        "Basidiomycota",
        "Mucoromycota",
        "Chytridiomycota",
        "Glomeromycota",
        "Oomycota",
        "Streptophyta",
        "Chlorophyta",
        "Rhodophyta",
    },
}


def clean_taxonomy(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    # The KNN CSV stores neighbor_taxonomies as a Python-like list string.
    # This function is for individual taxonomy strings.
    return [part.strip() for part in s.split(";") if part.strip()]


def parse_list_cell(value: object) -> list[object]:
    """Parse a serialized Python/CSV list safely."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    if isinstance(value, list):
        return value

    s = str(value).strip()
    if not s:
        return []

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (ValueError, SyntaxError):
        pass

    # Fallback for simple comma-separated values.
    return [x.strip() for x in s.split(",") if x.strip()]


def normalize_id(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def taxonomy_set(taxonomy: object) -> set[str]:
    return set(clean_taxonomy(taxonomy))


def broad_lineage(taxonomy: object, rank: str) -> str:
    parts = clean_taxonomy(taxonomy)

    # Prefer explicit recognized labels anywhere in the path.
    candidates = parts
    for value in candidates:
        if value in BROAD_RANKS.get(rank, set()):
            return value

    # Kingdom fallback: common root structure in the source.
    if rank == "kingdom":
        for value in ("Eukaryota", "Bacteria", "Archaea", "Viruses"):
            if value in parts:
                return value

    return ""


def genus_guess(taxonomy: object) -> str:
    """
    Conservative genus extraction.

    Taxonomy paths in this dataset are not guaranteed to encode ranks by fixed
    position. We therefore only use the final taxon-like component when it is
    clearly genus-shaped. Otherwise return empty rather than guessing.
    """
    parts = clean_taxonomy(taxonomy)
    if not parts:
        return ""

    # Common genus indicators in NCBI-like taxonomy strings.
    # A genus is generally a single alphabetic taxon token beginning uppercase.
    for part in reversed(parts):
        if (
            re.fullmatch(r"[A-Z][A-Za-z0-9_-]{2,}", part)
            and part not in {
                "Eukaryota",
                "Bacteria",
                "Archaea",
                "Metazoa",
                "Dikarya",
                "Opisthokonta",
                "Viridiplantae",
                "SAR",
                "Holozoa",
                "Ecdysozoa",
                "Spiralia",
                "Bilateria",
                "Amorphea",
                "Obazoa",
                "Fungi",
                "Animalia",
                "Plantae",
            }
        ):
            return part

    return ""


def shared_prefix_depth(a: object, b: object) -> int:
    pa = clean_taxonomy(a)
    pb = clean_taxonomy(b)

    depth = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        depth += 1
    return depth


def taxonomy_discordance(candidate_tax: object, neighbor_tax: object) -> float:
    """
    Rank-independent discordance score in [0, 1].

    0.0 = identical taxonomy path.
    1.0 = no shared taxonomy component at all.

    The score also accounts for the candidate's and neighbor's path lengths,
    so a divergence near the root scores more strongly than a divergence only
    at the terminal component.
    """
    a = clean_taxonomy(candidate_tax)
    b = clean_taxonomy(neighbor_tax)

    if not a or not b:
        return float("nan")

    depth = shared_prefix_depth(a, b)
    scale = max(len(a), len(b), 1)

    if depth == 0:
        return 1.0

    return min(1.0, max(0.0, 1.0 - depth / scale))


def parse_distances(value: object) -> list[float]:
    vals = parse_list_cell(value)
    result = []
    for v in vals:
        try:
            result.append(float(v))
        except (TypeError, ValueError):
            result.append(float("nan"))
    return result


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "candidate_record_id",
        "candidate_start",
        "candidate_end",
        "candidate_taxonomy",
        "k",
        "same_taxon_fraction",
        "neighbor_record_ids",
        "neighbor_taxonomies",
        "neighbor_distances",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "KNN CSV is missing required columns: "
            + ", ".join(missing)
        )


def analyze_candidate(row: pd.Series) -> tuple[dict, list[dict]]:
    candidate_id = normalize_id(row["candidate_record_id"])
    candidate_start = row["candidate_start"]
    candidate_end = row["candidate_end"]
    candidate_tax = normalize_id(row["candidate_taxonomy"])

    neighbor_ids = [normalize_id(x) for x in parse_list_cell(row["neighbor_record_ids"])]
    neighbor_tax = [normalize_id(x) for x in parse_list_cell(row["neighbor_taxonomies"])]
    distances = parse_distances(row["neighbor_distances"])

    n = max(len(neighbor_ids), len(neighbor_tax), len(distances))

    # Pad malformed rows rather than silently shifting fields.
    neighbor_ids += [""] * (n - len(neighbor_ids))
    neighbor_tax += [""] * (n - len(neighbor_tax))
    distances += [float("nan")] * (n - len(distances))

    seen: set[str] = set()
    neighbor_rows: list[dict] = []

    candidate_genus = genus_guess(candidate_tax)
    candidate_kingdom = broad_lineage(candidate_tax, "kingdom")

    same_genus = 0
    same_kingdom = 0
    discordances = []
    prefix_depths = []

    for i in range(n):
        nid = neighbor_ids[i]
        ntax = neighbor_tax[i]
        dist = distances[i]

        duplicate = bool(nid and nid in seen)
        if nid:
            seen.add(nid)

        ngenus = genus_guess(ntax)
        nkingdom = broad_lineage(ntax, "kingdom")

        is_same_genus = bool(candidate_genus and ngenus and candidate_genus == ngenus)
        is_same_kingdom = bool(
            candidate_kingdom and nkingdom and candidate_kingdom == nkingdom
        )

        if is_same_genus:
            same_genus += 1
        if is_same_kingdom:
            same_kingdom += 1

        prefix = shared_prefix_depth(candidate_tax, ntax)
        disc = taxonomy_discordance(candidate_tax, ntax)

        if not math.isnan(dist):
            pass
        if not math.isnan(disc):
            discordances.append(disc)
        prefix_depths.append(prefix)

        neighbor_rows.append(
            {
                "candidate_record_id": candidate_id,
                "candidate_start": candidate_start,
                "candidate_end": candidate_end,
                "candidate_taxonomy": candidate_tax,
                "neighbor_rank": i + 1,
                "neighbor_record_id": nid,
                "neighbor_taxonomy": ntax,
                "distance": dist,
                "is_duplicate_neighbor": duplicate,
                "candidate_genus": candidate_genus,
                "neighbor_genus": ngenus,
                "same_genus": is_same_genus,
                "candidate_kingdom": candidate_kingdom,
                "neighbor_kingdom": nkingdom,
                "same_kingdom": is_same_kingdom,
                "shared_prefix_depth": prefix,
                "taxonomic_discordance": disc,
            }
        )

    total = n
    unique_count = len({x for x in neighbor_ids if x})
    valid_distances = [x for x in distances if not math.isnan(x)]

    candidate = {
        "candidate_record_id": candidate_id,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "candidate_taxonomy": candidate_tax,
        "k_requested": int(row["k"]) if pd.notna(row["k"]) else total,
        "neighbor_rows": total,
        "unique_neighbor_count": unique_count,
        "unique_neighbor_fraction": unique_count / total if total else float("nan"),
        "same_taxon_fraction_original": row["same_taxon_fraction"],
        "candidate_genus": candidate_genus,
        "candidate_kingdom": candidate_kingdom,
        "same_genus_neighbors": same_genus,
        "same_genus_fraction": same_genus / total if total else float("nan"),
        "same_kingdom_neighbors": same_kingdom,
        "same_kingdom_fraction": same_kingdom / total if total else float("nan"),
        "min_distance": min(valid_distances) if valid_distances else float("nan"),
        "mean_distance": (
            sum(valid_distances) / len(valid_distances)
            if valid_distances else float("nan")
        ),
        "median_distance": (
            float(pd.Series(valid_distances).median())
            if valid_distances else float("nan")
        ),
        "mean_taxonomic_discordance": (
            sum(discordances) / len(discordances)
            if discordances else float("nan")
        ),
        "median_taxonomic_discordance": (
            float(pd.Series(discordances).median())
            if discordances else float("nan")
        ),
        "mean_shared_prefix_depth": (
            sum(prefix_depths) / len(prefix_depths)
            if prefix_depths else float("nan")
        ),
    }

    return candidate, neighbor_rows


def build_case45_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a transparent Case 4.5 ranking.

    Components:
      40% screening extremeness
      30% embedding similarity
      30% taxonomic discordance

    Screening extremeness is taken from candidate_score when available.
    Embedding similarity is inverted rank-normalized mean distance.
    Taxonomic discordance uses the candidate-level mean.

    The score is intentionally a ranking aid, not a statistical p-value.
    """
    out = df.copy()

    if "candidate_score" in out.columns:
        screen = pd.to_numeric(out["candidate_score"], errors="coerce")
    else:
        screen = pd.Series(float("nan"), index=out.index)

    out["screening_extremeness"] = screen

    # Convert lower mean distance into higher similarity.
    d = pd.to_numeric(out["mean_distance"], errors="coerce")
    valid_d = d.dropna()

    if len(valid_d):
        # percentile: 1 = smallest distance / strongest similarity
        out["embedding_similarity_percentile"] = 1.0 - d.rank(
            method="average", pct=True
        )
    else:
        out["embedding_similarity_percentile"] = float("nan")

    disc = pd.to_numeric(
        out["mean_taxonomic_discordance"], errors="coerce"
    )
    out["taxonomic_discordance_percentile"] = disc.rank(
        method="average", pct=True
    )

    # Candidate score may be on a very different numerical scale. Rank-normalize
    # it before combining.
    if screen.notna().any():
        out["screening_extremeness_percentile"] = screen.rank(
            method="average", pct=True
        )
    else:
        out["screening_extremeness_percentile"] = float("nan")

    out["case45_score"] = (
        0.40 * out["screening_extremeness_percentile"].fillna(0.0)
        + 0.30 * out["embedding_similarity_percentile"].fillna(0.0)
        + 0.30 * out["taxonomic_discordance_percentile"].fillna(0.0)
    )

    # Penalize duplicate-heavy neighborhoods so the ranking is based on unique
    # neighbors rather than repeated rows.
    out["case45_score_unique_adjusted"] = (
        out["case45_score"]
        * out["unique_neighbor_fraction"].fillna(0.0)
    )

    return out.sort_values(
        ["case45_score_unique_adjusted", "case45_score"],
        ascending=False,
        kind="stable",
    )


def write_summary(
    path: Path,
    candidates: pd.DataFrame,
    neighbors: pd.DataFrame,
    top_n: int,
) -> None:
    lines = [
        "Case 4.5 KNN analysis",
        "=" * 80,
        "",
        f"Candidates analyzed: {len(candidates)}",
        f"Neighbor rows: {len(neighbors)}",
        "",
        "Important:",
        "  family/order/class/phylum are NOT inferred from fixed taxonomy positions.",
        "  same_genus is conservative and may be blank when genus cannot be",
        "  established safely from the supplied taxonomy string.",
        "  taxonomic_discordance is rank-independent and based on shared hierarchy.",
        "",
        f"Top {min(top_n, len(candidates))} candidates",
        "-" * 80,
    ]

    display_cols = [
        "candidate_record_id",
        "candidate_start",
        "candidate_end",
        "candidate_taxonomy",
        "screening_extremeness",
        "min_distance",
        "mean_distance",
        "unique_neighbor_fraction",
        "same_genus_fraction",
        "same_kingdom_fraction",
        "mean_taxonomic_discordance",
        "case45_score_unique_adjusted",
    ]

    for _, row in candidates.head(top_n).iterrows():
        lines.append(
            f"{row['candidate_record_id']}:{row['candidate_start']}-{row['candidate_end']}"
        )
        lines.append(f"  taxonomy: {row['candidate_taxonomy']}")
        for col in display_cols[4:]:
            value = row.get(col, "")
            if isinstance(value, float):
                lines.append(f"  {col}: {value:.5f}" if not math.isnan(value) else f"  {col}: NA")
            else:
                lines.append(f"  {col}: {value}")
        lines.append("")

    # Compact neighbor composition for top candidates.
    lines.append("Top-candidate neighbor composition")
    lines.append("-" * 80)

    top_keys = {
        (
            str(r["candidate_record_id"]),
            r["candidate_start"],
            r["candidate_end"],
        )
        for _, r in candidates.head(top_n).iterrows()
    }

    for _, row in neighbors.iterrows():
        key = (
            str(row["candidate_record_id"]),
            row["candidate_start"],
            row["candidate_end"],
        )
        if key not in top_keys:
            continue

        lines.append(
            f"{row['candidate_record_id']}:{row['candidate_start']}-{row['candidate_end']} "
            f"#{row['neighbor_rank']} "
            f"distance={row['distance']:.6f} "
            f"duplicate={row['is_duplicate_neighbor']} "
            f"same_genus={row['same_genus']} "
            f"same_kingdom={row['same_kingdom']} "
            f"discordance={row['taxonomic_discordance']:.4f}"
        )
        lines.append(f"  neighbor: {row['neighbor_record_id']}")
        lines.append(f"  taxonomy: {row['neighbor_taxonomy']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Case 4.5 KNN candidates beyond exact taxonomy."
    )
    parser.add_argument(
        "--input",
        default="candidate_screening.knn.csv",
        type=Path,
        help="Existing KNN screening CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        default="case45_knn",
        type=Path,
        help="Output prefix.",
    )
    parser.add_argument(
        "--top-n",
        default=20,
        type=int,
        help="Number of candidates to show in the summary.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input KNN CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    validate_columns(df)

    candidates = []
    neighbor_rows = []

    for _, row in df.iterrows():
        candidate, neighbors = analyze_candidate(row)

        # Preserve screening columns when present.
        for col in [
            "candidate_score",
            "bio_extremeness",
            "likelihood_zscore",
            "candidate_taxonomy",
        ]:
            if col in row:
                candidate[col] = row[col]

        candidates.append(candidate)
        neighbor_rows.extend(neighbors)

    candidate_df = pd.DataFrame(candidates)
    neighbor_df = pd.DataFrame(neighbor_rows)

    candidate_df = build_case45_score(candidate_df)

    # Stable, explicit ordering for downstream review.
    candidate_df = candidate_df.reset_index(drop=True)
    candidate_df.insert(0, "case45_rank", range(1, len(candidate_df) + 1))

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    candidate_path = output_prefix.with_suffix(".candidates.csv")
    neighbor_path = output_prefix.with_suffix(".neighbors.csv")
    summary_path = output_prefix.with_suffix(".summary.txt")

    candidate_df.to_csv(candidate_path, index=False)
    neighbor_df.to_csv(neighbor_path, index=False)
    write_summary(summary_path, candidate_df, neighbor_df, args.top_n)

    print(f"Candidates analyzed : {len(candidate_df)}")
    print(f"Neighbor rows       : {len(neighbor_df)}")
    print(f"Candidate output    : {candidate_path}")
    print(f"Neighbor output     : {neighbor_path}")
    print(f"Summary output      : {summary_path}")
    print()
    print(candidate_df.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
