#!/usr/bin/env python3
"""Preflight gate for the biological annotation workflow.

This script deliberately stops before network access when required local
candidate/control covariates and rank-resolved taxonomy are absent. It never
scans datasets or attempts to work around the keyed retrieval policy.

Once a valid local keyed metadata table and frozen taxonomy snapshot are
available, extend this entry point to freeze the seeded case set, run the
footer-only dry run, and proceed only when all documented retrieval gates pass.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "metrics/derived/biological_visualization_analysis"
OUT = ROOT / "metrics/derived/biological_annotation"
REQUIRED = {
    "sequence_length",
    "gene_type",
    "is_coding_region",
    "strand",
    "sequence_hash",
}
KEY = {"record_id", "start", "end"}


def header(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return set(next(csv.reader(stream)))


def main() -> int:
    candidates = ANALYSIS / "umap_dbscan_memberships.csv"
    composition = ANALYSIS / "umap_candidate_composition_join.csv"
    likelihood = ANALYSIS / "umap_candidate_likelihood_join.csv"
    for path in (candidates, composition, likelihood):
        if not path.is_file():
            raise SystemExit(f"Required local artifact missing: {path.relative_to(ROOT)}")

    candidate_fields = header(candidates)
    composition_fields = header(composition)
    likelihood_fields = header(likelihood)
    available = candidate_fields | composition_fields | likelihood_fields
    missing = sorted(REQUIRED - available)
    taxonomy_is_ranked = all(
        rank in available
        for rank in ("kingdom", "phylum", "class", "order", "family", "genus", "species")
    )

    # Keep this rule explicit: raw UMAP DBSCAN eps=1.0, min_samples=20.
    # The specific cluster id is represented by the audit's raw_C2_e1_m20 name.
    if missing or not taxonomy_is_ranked:
        plan_path = OUT / "retrieval_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["status"] = "halted_before_remote_reads"
        plan["preflight_observed_missing_fields"] = missing
        plan["rank_annotated_taxonomy_available"] = taxonomy_is_ranked
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print("Preflight blocked; no case keys were selected and no remote reads were issued.")
        print(f"Missing per-key fields: {', '.join(missing) if missing else 'none'}")
        print(f"Rank-annotated taxonomy available: {taxonomy_is_ranked}")
        return 2

    print("Preflight fields exist. Case selection/retrieval requires a reviewed frozen plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
