"""Statistical analysis of paired LWE recovery experiments.

Wilson intervals summarize binomial recovery proportions. Exact two-sided
McNemar tests compare discordant outcomes on the SAME LWE instances.
P-values are unadjusted exploratory results, not security guarantees.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes, trials, z=1.959963984540054):
    """Two-sided 95% Wilson score interval (normal critical value)."""
    if not isinstance(trials, int) or trials <= 0 or not isinstance(successes, int) or not 0 <= successes <= trials:
        raise ValueError("require integer 0 <= successes <= trials, trials > 0")
    p = successes / trials
    z2 = z * z
    denom = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denom
    # Exact endpoints avoid tiny IEEE-754 residuals (e.g. 3.47e-18 at 0/100).
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == trials else min(1.0, center + half)
    return lower, upper


def mcnemar_exact(left_only, right_only):
    """Exact two-sided conditional binomial p-value for discordant pairs."""
    if any(not isinstance(x, int) or x < 0 for x in (left_only, right_only)):
        raise ValueError("discordant counts must be nonnegative integers")
    n = left_only + right_only
    if n == 0:
        return 1.0
    k = min(left_only, right_only)
    # Sum P[Bin(n, 1/2) <= k], using integer binomial coefficients.
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * (tail / (1 << n)))


def _write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_trials(paths):
    """Validate paired trials and configuration metadata when present.

    All-legacy CSVs remain supported, but their absent q/sigma metadata
    cannot be verified. Mixing old and new CSV formats is rejected.
    """
    if not paths:
        raise ValueError("at least one trial CSV is required")
    instances = {}
    algorithms_by_dimension = defaultdict(set)
    parameters_by_dimension = {}
    metadata_modes = set()
    metadata_fields = {"m", "q", "sigma", "float_type"}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"seed", "n", "trial", "instance_id", "reduction", "block_size", "success", "seconds"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path}: missing paired-trial CSV columns")
            present_metadata = metadata_fields.intersection(reader.fieldnames)
            if present_metadata and present_metadata != metadata_fields:
                raise ValueError(f"{path}: partial parameter metadata")
            metadata_modes.add(bool(present_metadata))
            count = 0
            for row in reader:
                count += 1
                try:
                    seed, n, trial = (int(row[k]) for k in ("seed", "n", "trial"))
                    beta, success = int(row["block_size"]), int(row["success"])
                    seconds = float(row["seconds"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}: invalid numeric trial record") from exc
                if seed < 0 or n < 1 or trial < 0 or beta < 0 or success not in (0, 1) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError(f"{path}: invalid trial values")
                label = "LLL" if beta == 0 else f"BKZ-{beta}"
                if (beta != 0 and beta < 2) or row["reduction"] != label or not row["instance_id"]:
                    raise ValueError(f"{path}: inconsistent reduction or instance ID")
                if present_metadata:
                    try:
                        m = int(row["m"])
                        q = int(row["q"])
                        sigma = float(row["sigma"])
                        float_type = row["float_type"]
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{path}: invalid experiment parameter metadata") from exc
                    if (m < n or q < 3 or q % 2 != 1 or not math.isfinite(sigma)
                            or sigma < 0 or sigma >= q or float_type not in ("d", "ld")):
                        raise ValueError(f"{path}: invalid experiment parameters")
                    parameters = (m, q, sigma, float_type)
                    previous = parameters_by_dimension.setdefault(n, parameters)
                    if previous != parameters:
                        raise ValueError(f"{path}: inconsistent experiment parameters at n={n}")
                key = (seed, n, trial)
                record = instances.setdefault(key, {"instance_id": row["instance_id"], "outcomes": {}})
                if record["instance_id"] != row["instance_id"]:
                    raise ValueError(f"{path}: unpaired instance fingerprint for {key}")
                if label in record["outcomes"]:
                    raise ValueError(f"{path}: duplicate outcome for {key}, {label}")
                record["outcomes"][label] = success
                algorithms_by_dimension[n].add(label)
            if not count:
                raise ValueError(f"{path}: empty trial CSV")
    if len(metadata_modes) > 1:
        raise ValueError("cannot mix legacy and new-format trial CSVs")
    for (seed, n, trial), record in instances.items():
        if set(record["outcomes"]) != algorithms_by_dimension[n]:
            raise ValueError(f"incomplete algorithm set at seed={seed}, n={n}, trial={trial}")
    # Require matching trial indices for every seed within each dimension.
    per_dimension = defaultdict(list)
    for seed, n, trial in instances:
        per_dimension[n].append((seed, trial))
    for n, values in per_dimension.items():
        seeds = {seed for seed, _ in values}
        trials_by_seed = {seed: {t for s, t in values if s == seed} for seed in seeds}
        if len({tuple(sorted(t)) for t in trials_by_seed.values()}) != 1:
            raise ValueError(f"dimension n={n}: seed runs have unequal trial indices")
    return instances


def summarize(paths, output_dir="results/stats"):
    instances = load_trials(paths)
    dimensions = sorted({n for _, n, _ in instances})
    summary, per_seed, comparisons = [], [], []
    for n in dimensions:
        keys = sorted(key for key in instances if key[1] == n)
        labels = sorted(instances[keys[0]]["outcomes"], key=lambda s: (0 if s == "LLL" else 1, int(s.split("-")[-1]) if "-" in s else 0))
        seeds = sorted({key[0] for key in keys})
        for label in labels:
            wins = sum(instances[key]["outcomes"][label] for key in keys)
            lower, upper = wilson_interval(wins, len(keys))
            summary.append({"n": n, "reduction": label, "seeds": len(seeds), "trials": len(keys),
                            "successes": wins, "rate": wins / len(keys),
                            "wilson95_low": lower, "wilson95_high": upper})
            for seed in seeds:
                seed_keys = [key for key in keys if key[0] == seed]
                s = sum(instances[key]["outcomes"][label] for key in seed_keys)
                lo, hi = wilson_interval(s, len(seed_keys))
                per_seed.append({"seed": seed, "n": n, "reduction": label,
                                 "trials": len(seed_keys), "successes": s, "rate": s / len(seed_keys),
                                 "wilson95_low": lo, "wilson95_high": hi})
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                left_only = sum(instances[key]["outcomes"][left] == 1 and instances[key]["outcomes"][right] == 0 for key in keys)
                right_only = sum(instances[key]["outcomes"][left] == 0 and instances[key]["outcomes"][right] == 1 for key in keys)
                both = sum(instances[key]["outcomes"][left] == 1 and instances[key]["outcomes"][right] == 1 for key in keys)
                neither = len(keys) - left_only - right_only - both
                comparisons.append({"n": n, "left": left, "right": right, "pairs": len(keys),
                                    "both_success": both, "left_only": left_only,
                                    "right_only": right_only, "neither_success": neither,
                                    "rate_difference_right_minus_left": (right_only - left_only) / len(keys),
                                    "mcnemar_exact_p": mcnemar_exact(left_only, right_only)})
    out = Path(output_dir)
    _write_csv(out / "summary.csv", summary, ["n", "reduction", "seeds", "trials", "successes", "rate", "wilson95_low", "wilson95_high"])
    _write_csv(out / "per_seed.csv", per_seed, ["seed", "n", "reduction", "trials", "successes", "rate", "wilson95_low", "wilson95_high"])
    _write_csv(out / "paired_tests.csv", comparisons, ["n", "left", "right", "pairs", "both_success", "left_only", "right_only", "neither_success", "rate_difference_right_minus_left", "mcnemar_exact_p"])
    for row in summary:
        print(f"n={row['n']} {row['reduction']}: {row['successes']}/{row['trials']} "
              f"({row['rate']:.1%}), Wilson 95% [{row['wilson95_low']:.1%}, {row['wilson95_high']:.1%}]")
    for row in comparisons:
        print(f"n={row['n']} {row['left']} vs {row['right']}: "
              f"discordant={row['left_only']}/{row['right_only']} "
              f"exact McNemar p={row['mcnemar_exact_p']:.4g} (unadjusted)")
    print(f"Saved statistical reports in {out}")
    return summary, per_seed, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_csvs", nargs="+", help="paired per-trial CSVs from experiments.scale")
    parser.add_argument("--outdir", default="results/stats")
    args = parser.parse_args()
    try:
        summarize(args.trial_csvs, output_dir=args.outdir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
