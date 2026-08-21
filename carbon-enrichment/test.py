#!/usr/bin/env python3

"""
Carbon CPU-enriched dataset -> GPU-stage readiness audit.
Optimized for high-throughput streaming evaluation.
"""

import math
import time
from collections import Counter
from datasets import load_dataset

# ============================================================================
# Configuration
# ============================================================================
DATASET = "AINovice2005/carbon-cpu-enriched-sequences"
SPLIT = "train"

# Iteration bound: 2.5 million rows will take roughly 2 to 3 minutes.
MAX_ROWS = 2_500_000

VALIDATION_SAMPLE = 100_000
MAX_ERRORS = 10

# Lowered reporting threshold to see progress pings during a short run
REPORT_EVERY = 500_000

EXPECTED_KMER_DIM = 64
KMER_SUM_TOL = 1e-5

# Tuple is slightly faster to iterate over than a set
REQUIRED_COLUMNS = (
    "sequence", "strand", "strand_normalized_sequence", "sequence_length",
    "gene_length", "start", "end", "gc_content", "gc_skew", "shannon_entropy",
    "kmer_frequency_vector", "taxonomy", "taxonomy_depth", "gene_type",
    "species_type", "qc_flag"
)
COLUMNS = sorted(REQUIRED_COLUMNS)

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# ============================================================================
# Helpers
# ============================================================================
def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def pct(part, total):
    return 100.0 * part / total if total else 0.0

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# O(1) length binning mapping using log2
def get_length_bin_index(length: int) -> int:
    if length < 256:
        return 0
    # math.log2(256) = 8 -> index 1. log2(512) = 9 -> index 2.
    idx = int(math.log2(length)) - 7
    return min(idx, 11)

LENGTH_BINS = [
    0, 256, 512, 1024, 2048, 4096, 8192, 16384, 
    32768, 65536, 131072, math.inf
]

# ============================================================================
# Main streaming pass (Wrapped in function for faster local lookups)
# ============================================================================
def run_audit():
    print_header("CARBON CPU-ENRICHED → GPU READINESS AUDIT")
    print(f"Dataset : {DATASET}")
    print(f"Split   : {SPLIT}")
    print(f"Sample  : {VALIDATION_SAMPLE:,}")
    if MAX_ROWS:
        print(f"Max Rows: {MAX_ROWS:,} (Truncated Run)")

    ds = load_dataset(DATASET, split=SPLIT, streaming=True).select_columns(COLUMNS)

    rows = 0
    null_counts = Counter()
    taxonomy_counts = Counter()
    gene_type_counts = Counter()
    strand_counts = Counter()
    qc_counts = Counter()

    length_count = 0
    length_sum = 0
    length_min = math.inf
    length_max = 0
    length_bins_counts = Counter()

    sampled = 0
    strand_checked, strand_errors = 0, 0
    kmer_checked, kmer_errors = 0, 0
    numeric_errors = 0
    
    stop_strand = False
    stop_kmer = False
    stop_numeric = False

    start_time = time.time()

    for row in ds:
        if MAX_ROWS and rows >= MAX_ROWS:
            break
            
        rows += 1

        # 1. Null Audit
        for column in REQUIRED_COLUMNS:
            if row.get(column) is None:
                null_counts[column] += 1

        # 2. Cheap categorical statistics
        taxonomy = row.get("taxonomy")
        gene_type = row.get("gene_type")
        strand = row.get("strand")
        qc_flag = row.get("qc_flag")

        if taxonomy: taxonomy_counts[taxonomy] += 1
        if gene_type: gene_type_counts[gene_type] += 1
        if strand: strand_counts[strand] += 1
        if qc_flag: qc_counts[qc_flag] += 1

        # 3. Length statistics (O(1) Binning)
        sequence_length = row.get("sequence_length")
        if sequence_length is not None:
            length_count += 1
            length_sum += sequence_length
            if sequence_length < length_min: length_min = sequence_length
            if sequence_length > length_max: length_max = sequence_length
            length_bins_counts[get_length_bin_index(sequence_length)] += 1

        # 4. Sampled Expensive Checks
        if sampled < VALIDATION_SAMPLE:
            # Strand normalization
            if not stop_strand:
                sequence = row.get("sequence")
                normalized = row.get("strand_normalized_sequence")
                if sequence and normalized:
                    expected = None
                    if strand == "+":
                        expected = sequence
                    elif strand == "-":
                        expected = reverse_complement(sequence)

                    if expected:
                        strand_checked += 1
                        if normalized != expected:
                            strand_errors += 1
                            if strand_errors <= MAX_ERRORS:
                                print(f"\nSTRAND ERROR: {taxonomy} | Strand: {strand}")
                            if strand_errors == MAX_ERRORS:
                                stop_strand = True
                                print("\nToo many strand errors; stopping strand validation.")

            # K-mer vector validation (C-optimized built-ins)
            if not stop_kmer:
                vector = row.get("kmer_frequency_vector")
                if vector:
                    kmer_checked += 1
                    try:
                        if len(vector) != EXPECTED_KMER_DIM:
                            kmer_errors += 1
                            if kmer_errors <= MAX_ERRORS:
                                print(f"\nKMER DIMENSION ERROR: Expected {EXPECTED_KMER_DIM}, got {len(vector)}")
                        else:
                            total = sum(vector)
                            # C-level evaluation over sequence
                            bad_value = not all(math.isfinite(v) and v >= 0 for v in vector)
                            
                            if bad_value or abs(total - 1.0) > KMER_SUM_TOL:
                                kmer_errors += 1
                                if kmer_errors <= MAX_ERRORS:
                                    print(f"\nKMER NORMALIZATION ERROR: sum={total:.8f}, valid={not bad_value}")
                                if kmer_errors == MAX_ERRORS:
                                    stop_kmer = True
                    except Exception as exc:
                        kmer_errors += 1
                        if kmer_errors <= MAX_ERRORS:
                            print(f"\nKMER ERROR: {repr(exc)}")

            # Numeric sanity
            if not stop_numeric:
                numeric_values = (row.get("gc_content"), row.get("gc_skew"), row.get("shannon_entropy"))
                for value in numeric_values:
                    if value is not None and not math.isfinite(value):
                        numeric_errors += 1
                        if numeric_errors <= MAX_ERRORS:
                            print(f"\nNUMERIC ERROR: {value}")
                        if numeric_errors == MAX_ERRORS:
                            stop_numeric = True

            sampled += 1

        # 5. Progress
        if rows % REPORT_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"{rows:>12,} rows | {rows / elapsed:,.0f} rows/s | tax_count={len(taxonomy_counts)}")

    # ============================================================================
    # Final report (extracted variables)
    # ============================================================================
    elapsed = time.time() - start_time

    print_header("CORPUS")
    print(f"Rows processed       : {rows:,}")
    print(f"Runtime              : {elapsed / 60:.2f} min")
    print(f"Throughput           : {rows / elapsed:,.0f} rows/s")

    print_header("NULL AUDIT")
    null_total = sum(null_counts.values())
    if null_total == 0:
        print("PASS: No nulls in required GPU-stage columns.")
    else:
        for column in REQUIRED_COLUMNS:
            count = null_counts.get(column, 0)
            if count:
                print(f"{column:35s} {count:>12,} ({pct(count, rows):6.3f}%)")

    print_header("TAXONOMIC / BIOLOGICAL SANITY")
    print(f"Unique taxonomy values : {len(taxonomy_counts):,}")
    print(f"Gene types             : {len(gene_type_counts):,}")

    print("\nTop taxonomy values:")
    for tax, count in taxonomy_counts.most_common(5):
        print(f"{count:>12,} ({pct(count, rows):6.2f}%)  {tax}")

    print("\nGene types:")
    for value, count in gene_type_counts.most_common():
        print(f"{count:>12,} ({pct(count, rows):6.2f}%)  {value}")

    print("\nStrand:")
    for value, count in strand_counts.most_common():
        print(f"{count:>12,} ({pct(count, rows):6.2f}%)  {value}")

    print("\nQC:")
    for value, count in qc_counts.most_common():
        print(f"{count:>12,} ({pct(count, rows):6.2f}%)  {value}")

    print_header("SEQUENCE LENGTH / GPU WORKLOAD")
    if length_count:
        print(f"Min length       : {length_min:,}")
        print(f"Max length       : {length_max:,}")
        print(f"Mean length      : {length_sum / length_count:,.1f}")
        print("\nLength distribution:")
        for i in range(len(LENGTH_BINS) - 1):
            lower = LENGTH_BINS[i]
            upper = LENGTH_BINS[i + 1]
            count = length_bins_counts.get(i, 0)
            
            label = f">= {lower:,}" if upper == math.inf else f"{lower:,} - {upper - 1:,}"
            print(f"{label:20s} {count:>12,} ({pct(count, rows):6.2f}%)")

    print_header("SAMPLED VALIDATIONS")
    print(f"Rows sampled     : {sampled:,}")
    
    print(f"\n[STRAND] Checked : {strand_checked:,} | Errors: {strand_errors:,}")
    if strand_errors == 0: print("   PASS: strand normalization is consistent.")
    else: print("   FAIL: strand normalization errors detected.")

    print(f"\n[K-MER]  Checked : {kmer_checked:,} | Errors: {kmer_errors:,}")
    if kmer_errors == 0: print(f"   PASS: all vectors ({EXPECTED_KMER_DIM}-dim, finite, non-negative, normalized).")
    else: print("   FAIL: k-mer vector errors detected.")

    print(f"\n[NUMERIC] Errors : {numeric_errors:,}")
    if numeric_errors == 0: print("   PASS: no NaN/Inf detected in sampled features.")
    else: print("   FAIL: invalid numeric values detected.")

    print_header("GPU STAGE READINESS")
    checks = {
        "required columns / nulls": null_total == 0,
        "strand normalization": strand_errors == 0,
        "k-mer vectors": kmer_errors == 0,
        "numeric values": numeric_errors == 0,
        "sequence lengths": length_count == rows,
    }

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if all(checks.values()):
        print("\n" + "=" * 80 + "\nRESULT: GO\n" + "=" * 80)
        print("CPU-enriched dataset is suitable for proceeding to GPU enrichment.")
    else:
        print("\n" + "=" * 80 + "\nRESULT: HOLD\n" + "=" * 80)
        print("Resolve the failed validation(s) before GPU enrichment.")

if __name__ == "__main__":
    run_audit()