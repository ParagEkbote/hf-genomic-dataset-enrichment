"""
CPU-only wall-time estimator for the Carbon GPU enrichment stage.

What this does
---------------
1. Streams N sample rows from the raw HF corpus (no GPU, no model load).
2. Measures the real distribution of `sequence` lengths (chars) for those rows.
3. Converts char-length -> token-length using a configurable chars-per-token
   ratio (default assumes ~1 token per nucleotide char, which is the
   worst-case / most conservative assumption for a DNA tokenizer; override
   with --chars-per-token if you know your tokenizer's real ratio, e.g. from
   inspecting `token_length` in an existing tokenized_corpus shard).
4. Extrapolates avg-tokens/row across the sample to 30K / 1M / 3M / 6M rows.
5. Computes wall-time estimates for each row target given one or more
   measured throughput values (tokens/sec), so you can plug in your
   batch_size=4 baseline AND a new batch_size=64/128 measurement once you
   have it, and compare directly.

This intentionally does NOT touch the GPU, the tokenizer, or the model.
It only samples raw rows to get a length distribution, which is enough to
sanity-check the token-count assumption before committing to a multi-day
GPU run.

Usage
-----
    python estimate_wall_time.py --sample-size 2000 --throughput 843
    python estimate_wall_time.py --sample-size 2000 --throughput 843 6744

If you already have a measured avg tokens/row (e.g. from your 748-row log:
~0.24M tokens / 748 rows = ~321 tok/row) and don't want to re-stream the
corpus, pass --avg-tokens-per-row directly to skip sampling entirely.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

# Mirrors the shared data contract module (kept inline here so this script
# has zero dependency on the dagster package -- see carbon_enrichment's
# schema.py for the source of truth).
HF_DATASET_PATH = "HuggingFaceBio/carbon-pretraining-corpus"
HF_DATASET_CONFIG = "eukaryote_generator"
STREAMING_SPLIT = "train"

ROW_TARGETS = {
    "30K (dev_gpu_medium x1)": 30_000,
    "1M (integration)": 1_000_000,
    "3M (proposed run)": 3_000_000,
    "6M": 6_000_000,
}


MAX_NATIVE_CONTEXT_TOKENS = 32_768


@dataclass
class SampleStats:
    n_rows: int
    avg_chars: float
    median_chars: float
    p90_chars: float
    p99_chars: float
    max_chars: float
    stdev_chars: float
    avg_tokens_per_row: float
    max_tokens_per_row: float
    pct_over_context_limit: float


def sample_sequence_lengths(sample_size: int, chars_per_token: float) -> SampleStats:
    """Stream `sample_size` rows from the raw corpus and measure sequence length.

    Uses HF `datasets` streaming mode so it never materializes the full
    dataset -- consistent with the pipeline's IterableDataset approach.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "The `datasets` library is required for live sampling.\n"
            "Install it with: pip install datasets\n"
            "Or skip sampling entirely with --avg-tokens-per-row <value>.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Streaming {sample_size} rows from {HF_DATASET_PATH} "
        f"({HF_DATASET_CONFIG}/{STREAMING_SPLIT}) ...",
        file=sys.stderr,
    )

    ds = load_dataset(
        HF_DATASET_PATH,
        HF_DATASET_CONFIG,
        split=STREAMING_SPLIT,
        streaming=True,
    )

    lengths = []
    for i, row in enumerate(ds):
        if i >= sample_size:
            break
        seq = row.get("sequence", "") or ""
        lengths.append(len(seq))
        if (i + 1) % 500 == 0:
            print(f"  sampled {i + 1}/{sample_size} rows...", file=sys.stderr)

    if not lengths:
        print("No rows sampled -- check dataset access / network.", file=sys.stderr)
        sys.exit(1)

    avg_chars = statistics.mean(lengths)
    median_chars = statistics.median(lengths)
    stdev_chars = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    sorted_lengths = sorted(lengths)
    p90_chars = sorted_lengths[int(0.9 * len(sorted_lengths)) - 1]
    p99_chars = sorted_lengths[int(0.99 * len(sorted_lengths)) - 1]
    max_chars = sorted_lengths[-1]

    avg_tokens_per_row = avg_chars * chars_per_token
    max_tokens_per_row = max_chars * chars_per_token

    over_limit = sum(
        1 for L in lengths if L * chars_per_token > MAX_NATIVE_CONTEXT_TOKENS
    )
    pct_over_context_limit = 100.0 * over_limit / len(lengths)

    return SampleStats(
        n_rows=len(lengths),
        avg_chars=avg_chars,
        median_chars=median_chars,
        p90_chars=p90_chars,
        p99_chars=p99_chars,
        max_chars=max_chars,
        stdev_chars=stdev_chars,
        avg_tokens_per_row=avg_tokens_per_row,
        max_tokens_per_row=max_tokens_per_row,
        pct_over_context_limit=pct_over_context_limit,
    )


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.2f} hr"
    days = hours / 24
    return f"{days:.2f} days"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of rows to stream+measure for the length distribution (default: 2000)",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=1.0,
        help="Conversion ratio from sequence chars to tokens. "
        "Default 1.0 assumes ~1 token/nucleotide (conservative). "
        "If your tokenizer uses k-mers, this should be < 1.0 "
        "(e.g. 0.33 for 3-mers). Check an actual token_length "
        "column from carbon_tokenized_corpus to calibrate this.",
    )
    parser.add_argument(
        "--avg-tokens-per-row",
        type=float,
        default=None,
        help="Skip live sampling and use this avg tokens/row directly "
        "(e.g. 321, derived from your 748-row log: 0.24M/748).",
    )
    parser.add_argument(
        "--throughput",
        type=float,
        nargs="+",
        default=[843.0],
        help="One or more measured throughput values in tokens/sec "
        "to compare (e.g. --throughput 843 6744 for batch_size=4 "
        "vs a new batch_size=64 measurement)",
    )
    args = parser.parse_args()

    stats = None
    if args.avg_tokens_per_row is not None:
        avg_tokens_per_row = args.avg_tokens_per_row
        print(
            f"Using provided avg tokens/row: {avg_tokens_per_row:.1f} (no sampling performed)\n"
        )
    else:
        stats = sample_sequence_lengths(args.sample_size, args.chars_per_token)
        avg_tokens_per_row = stats.avg_tokens_per_row
        print()
        print(f"Sample size:             {stats.n_rows} rows")
        print(f"Avg sequence length:     {stats.avg_chars:.1f} chars")
        print(f"Median sequence length:  {stats.median_chars:.1f} chars")
        print(f"Stdev sequence length:   {stats.stdev_chars:.1f} chars")
        print(f"P90 sequence length:     {stats.p90_chars:.1f} chars")
        print(f"P99 sequence length:     {stats.p99_chars:.1f} chars")
        print(f"Max sequence length:     {stats.max_chars:.1f} chars")
        print(f"chars_per_token used:    {args.chars_per_token}")
        print(f"=> Avg tokens/row:       {avg_tokens_per_row:.1f}")
        print(f"=> Max tokens/row:       {stats.max_tokens_per_row:.1f}")
        print(
            f"=> % rows over {MAX_NATIVE_CONTEXT_TOKENS:,} tok context: {stats.pct_over_context_limit:.2f}%"
        )
        if stats.pct_over_context_limit > 0:
            print("   ^ These rows will need truncation/chunking -- they will NOT")
            print("     fit in a single forward pass at native context length.")
        print()

    print(f"{'Row target':<24}{'Total tokens':>16}", end="")
    for tp in args.throughput:
        print(f"{f'@ {tp:.0f} tok/s':>18}", end="")
    print()
    print("-" * (24 + 16 + 18 * len(args.throughput)))

    for label, rows in ROW_TARGETS.items():
        total_tokens = rows * avg_tokens_per_row
        print(f"{label:<24}{total_tokens / 1e6:>13.1f}M ", end="")
        for tp in args.throughput:
            wall_sec = total_tokens / tp
            print(f"{format_duration(wall_sec):>18}", end="")
        print()

    # Dedicated 30K scenario table -- this is the actual planned A100 run.
    # Since real throughput at batch_size > 4 is still unmeasured, show a
    # spread of plausible speedup multipliers rather than a single guess.
    thirty_k_tokens = 30_000 * avg_tokens_per_row
    base_throughput = args.throughput[0]
    speedup_multipliers = [1, 2, 4, 8, 12, 16]

    print()
    print("=" * 70)
    print(
        f"FOCUSED ESTIMATE: 30,000 rows @ {avg_tokens_per_row:.0f} avg tok/row "
        f"= {thirty_k_tokens / 1e6:.1f}M tokens"
    )
    print("=" * 70)
    print(f"{'Speedup vs base':<20}{'Assumed tok/s':>16}{'Wall time':>16}")
    print("-" * 52)
    for mult in speedup_multipliers:
        tp = base_throughput * mult
        wall_sec = thirty_k_tokens / tp
        label = f"{mult}x (batch=4)" if mult == 1 else f"{mult}x"
        print(f"{label:<20}{tp:>16.0f}{format_duration(wall_sec):>16}")
    print()
    print(
        f"(base throughput = {base_throughput:.0f} tok/s, from your batch_size=4 log)"
    )

    if stats is not None and stats.pct_over_context_limit > 0:
        print()
        print(
            f"WARNING: {stats.pct_over_context_limit:.2f}% of sampled rows exceed "
            f"{MAX_NATIVE_CONTEXT_TOKENS:,} tokens."
        )
        print("Batch size will be capped by your LONGEST row in a batch, not the")
        print("average -- consider length-bucketed batching so short/long sequences")
        print(
            "aren't padded together, which wastes compute and lowers effective tok/s."
        )

    print()
    print("Notes:")
    print("- These are linear extrapolations from the sample's avg tokens/row.")
    print("  Real corpora often have heavy-tailed sequence length distributions --")
    print("  check the P90/P99/max vs avg above; if they diverge a lot, consider a")
    print("  larger --sample-size for a more stable estimate.")
    print("- Throughput at batch_size=4 will NOT hold at batch_size=64/128.")
    print("  Re-run this script with --throughput <old> <new> once you've measured")
    print("  real tok/s from a dev_gpu_small (3K row) run at the new batch size --")
    print("  that single real measurement replaces the whole speedup-multiplier table.")


if __name__ == "__main__":
    main()
