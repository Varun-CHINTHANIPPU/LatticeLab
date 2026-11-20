"""Offline wall-clock timing analysis for matched LLL/BKZ trial records.

Descriptive quantiles and paired median ratios; no new lattice attacks.
Bootstrap intervals resample *instances*, not individual algorithm timings.
"""
import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.stats import load_trials, _write_csv


def quantile(values, fraction):
    """Linear-interpolated sample quantile (Hyndman-Fan type 7)."""
    if not values or not 0 <= fraction <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("nonempty finite values and fraction in [0,1] required")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    low = math.floor(pos)
    high = math.ceil(pos)
    return ordered[low] + (pos - low) * (ordered[high] - ordered[low])


def bootstrap_median_ratio(left, right, replicates=2000, seed=7):
    """95% percentile bootstrap CI for median(left/right), paired by instance.

    A ratio above 1 means right was faster. Each paired observation is sampled
    together; this is not a CI for ratio of separate group medians.
    """
    if len(left) != len(right) or not left or any(
        not math.isfinite(x) or x <= 0 for x in [*left, *right]
    ):
        raise ValueError("paired times must have equal nonzero lengths and be finite, positive")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 100:
        raise ValueError("bootstrap replicates must be >= 100")
    ratios = [a / b for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(ratios)
    samples = [statistics.median(ratios[rng.randrange(n)] for _ in range(n))
               for _ in range(replicates)]
    return statistics.median(ratios), quantile(samples, 0.025), quantile(samples, 0.975)


def analyze(paths, outdir="results/performance", bootstrap=2000, seed=7):
    """Summarize existing paired trial CSVs; do not execute any attacks."""
    # Shared validator checks fingerprints, missing algorithms, duplicates, etc.
    instances = load_trials(paths)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be nonnegative")
    times = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["seed"]), int(row["n"]), int(row["trial"]))
                label = row["reduction"]
                value = float(row["seconds"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{path}: timing must be strictly positive")
                times[(key, label)] = value
    dimensions = sorted({key[1] for key in instances})
    summary, per_seed, comparisons = [], [], []
    for n in dimensions:
        keys = sorted(key for key in instances if key[1] == n)
        labels = sorted(instances[keys[0]]["outcomes"],
                        key=lambda label: (label != "LLL", int(label.split("-")[-1]) if "-" in label else 0))
        seeds = sorted({key[0] for key in keys})
        for label in labels:
            values = [times[(key, label)] for key in keys]
            summary.append(_timing_row(n, label, values, len(seeds)))
            for current_seed in seeds:
                vals = [times[(key, label)] for key in keys if key[0] == current_seed]
                per_seed.append({"seed": current_seed, **_timing_row(n, label, vals, 1)})
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                a = [times[(key, left)] for key in keys]
                b = [times[(key, right)] for key in keys]
                ratio, low, high = bootstrap_median_ratio(a, b, replicates=bootstrap, seed=seed + n + i * 100 + labels.index(right))
                comparisons.append({"n": n, "left": left, "right": right, "pairs": len(keys),
                                    "median_paired_left_over_right": ratio,
                                    "bootstrap95_low": low, "bootstrap95_high": high,
                                    "right_faster_pairs": sum(y < x for x, y in zip(a, b)),
                                    "equal_pairs": sum(x == y for x, y in zip(a, b)),
                                    "left_faster_pairs": sum(x < y for x, y in zip(a, b))})
    out = Path(outdir)
    base_cols = ["n", "reduction", "seeds", "trials", "mean_seconds", "median_seconds",
                 "std_seconds", "min_seconds", "p05_seconds", "p25_seconds",
                 "p75_seconds", "p95_seconds", "max_seconds"]
    _write_csv(out / "timing_summary.csv", summary, base_cols)
    _write_csv(out / "timing_per_seed.csv", per_seed, ["seed", *base_cols])
    _write_csv(out / "timing_paired.csv", comparisons, ["n", "left", "right", "pairs",
        "median_paired_left_over_right", "bootstrap95_low", "bootstrap95_high",
        "right_faster_pairs", "equal_pairs", "left_faster_pairs"])
    for row in summary:
        print(f"n={row['n']} {row['reduction']}: median={row['median_seconds']:.3f}s "
              f"p95={row['p95_seconds']:.3f}s (N={row['trials']})")
    for row in comparisons:
        print(f"n={row['n']} {row['left']}/{row['right']}: paired median ratio="
              f"{row['median_paired_left_over_right']:.3f}, "
              f"bootstrap 95% [{row['bootstrap95_low']:.3f}, {row['bootstrap95_high']:.3f}]")
    print(f"Saved timing reports in {out}")
    return summary, per_seed, comparisons


def _timing_row(n, label, values, seeds):
    return {"n": n, "reduction": label, "seeds": seeds, "trials": len(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "std_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_seconds": min(values), "p05_seconds": quantile(values, .05),
            "p25_seconds": quantile(values, .25), "p75_seconds": quantile(values, .75),
            "p95_seconds": quantile(values, .95), "max_seconds": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_csvs", nargs="+")
    parser.add_argument("--outdir", default="results/performance")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7, help="bootstrap RNG seed, not LWE seed")
    args = parser.parse_args()
    try:
        analyze(args.trial_csvs, args.outdir, args.bootstrap, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
