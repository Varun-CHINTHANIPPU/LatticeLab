"""Compare measured and Gaussian-approximated LWE decryption failure rates."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from lab.lwe import Params, keygen, encrypt, decrypt


def predicted_failure(q, m, sigma):
    """Across-key normal approximation for |<r,e>| >= q/4."""
    if sigma <= 0:
        return 0.0
    return math.erfc((q / 4) / (sigma * math.sqrt(m / 2) * math.sqrt(2)))


def per_key_prediction(q, errors):
    """Normal approximation conditional on the actual fixed error vector."""
    errors = np.asarray(errors, dtype=np.float64)
    mean = float(np.sum(errors)) / 2
    sd = float(np.linalg.norm(errors)) / 2
    threshold = q / 4
    if sd == 0:
        return float(abs(mean) >= threshold)
    return 0.5 * (math.erfc((threshold - mean) / (sd * math.sqrt(2)))
                  + math.erfc((threshold + mean) / (sd * math.sqrt(2))))


def run(sigmas=(8, 20, 30, 40, 50, 60, 80), keys=200, per_key=200,
        seed=2026, output="results/failrate.csv"):
    sigmas = tuple(sigmas)
    if not sigmas:
        raise ValueError("at least one sigma is required")
    if keys < 2 or per_key < 1:
        raise ValueError("need at least two keys and one encryption per key")
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in sigmas:
        p = Params(n=32, m=128, q=3329, sigma=float(sigma))
        failure_rates, predicted_keys = [], []
        for _ in range(keys):
            public_key, secret, errors = keygen(p, rng)
            failures = sum(
                decrypt(p, secret, encrypt(p, public_key, i % 2, rng)) != i % 2
                for i in range(per_key)
            )
            failure_rates.append(failures / per_key)
            predicted_keys.append(per_key_prediction(p.q, errors))
        measured = float(np.mean(failure_rates))
        ci_half = 1.96 * float(np.std(failure_rates, ddof=1)) / math.sqrt(keys)
        result = {
            "seed": seed, "n": p.n, "m": p.m, "q": p.q,
            "sigma": int(sigma) if float(sigma).is_integer() else float(sigma),
            "keys": keys, "encryptions_per_key": per_key,
            "measured": measured, "ci95_low": max(0.0, measured - ci_half),
            "ci95_high": min(1.0, measured + ci_half),
            "predicted": predicted_failure(p.q, p.m, sigma),
            "predicted_conditional_mean": float(np.mean(predicted_keys)),
            "worst_key": max(failure_rates),
        }
        rows.append(result)
        print(f"sigma={sigma:>5g} measured={measured:.4f} ±{ci_half:.4f} "
              f"predicted={result['predicted']:.4f} worst={result['worst_key']:.3f}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=200)
    parser.add_argument("--per-key", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[8, 20, 30, 40, 50, 60, 80])
    parser.add_argument("--output", default="results/failrate.csv")
    args = parser.parse_args()
    run(sigmas=args.sigmas, keys=args.keys, per_key=args.per_key, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
