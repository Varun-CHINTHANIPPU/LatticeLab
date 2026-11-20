"""Paired LLL/BKZ primal-attack study on identical weak LWE instances.

All reductions at a given n use the same (A, b, secret) for each trial.
Separate dimension-specific RNG streams make results independent of case order.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import statistics
import time

import numpy as np

from lab.lwe import Params, keygen
from lab.attacks import primal_attack


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _instance_id(A, b):
    """Fingerprint public LWE instance for auditing paired runs (not a secret)."""
    digest = hashlib.sha256()
    for value in (A, b):
        array = np.asarray(value, dtype="<i8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b":")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()[:24]


def run(cases=((20, 0), (40, 0), (60, 20), (70, 30)), trials=10,
        seed=7, sigma=3.0, q=3329, float_type="d", output="results/attack.csv",
        trial_output=None):
    """Run paired attacks and return aggregate rows.

    For each n, generate trials LWE instances once and apply every requested
    reduction to each instance. Each dimension uses a deterministic RNG stream
    derived from (seed, n), independent of case order and other dimensions.
    Ground truth is used ONLY for scoring, never supplied to primal_attack.
    """
    cases = tuple(tuple(case) for case in cases)
    if not cases:
        raise ValueError("at least one (n, block size) case is required")
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError("trials must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate (n, block size) configurations are not allowed")
    for n, beta in cases:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(beta, int) or isinstance(beta, bool) or (beta != 0 and beta < 2):
            raise ValueError("block size must be 0 (LLL) or >= 2 (BKZ)")

    by_n = {}
    for n, beta in cases:
        by_n.setdefault(n, []).append(beta)

    stats = {(n, beta): {"successes": 0, "durations": []} for n, beta in cases}
    trial_rows = []
    for n, betas in by_n.items():
        params = Params(n=n, m=n, q=q, sigma=sigma, secret="small")
        rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
        for trial in range(trials):
            (A, b), secret, _ = keygen(params, rng)
            fingerprint = _instance_id(A, b)
            for beta in betas:
                start = time.perf_counter()
                candidates = primal_attack(A.copy(), b.copy(), q, block_size=beta or None,
                                           float_type=float_type, sigma=sigma,
                                           return_candidates=True)
                duration = time.perf_counter() - start
                success = int(any(np.array_equal(candidate % q, secret % q)
                                  for candidate in candidates))
                stats[(n, beta)]["successes"] += success
                stats[(n, beta)]["durations"].append(duration)
                trial_rows.append({
                    "seed": seed, "n": n, "trial": trial, "instance_id": fingerprint,
                    "reduction": "LLL" if beta == 0 else f"BKZ-{beta}",
                    "block_size": beta, "success": success, "seconds": duration,
                    "m": n, "q": q, "sigma": sigma, "float_type": float_type,
                })

    rows = []
    for n, beta in cases:
        record = stats[(n, beta)]
        successes = record["successes"]
        row = {
            "seed": seed, "n": n, "m": n, "q": q, "sigma": sigma,
            "dimension": 2 * n + 1,
            "reduction": "LLL" if beta == 0 else f"BKZ-{beta}",
            "block_size": beta, "trials": trials, "successes": successes,
            "success_rate": successes / trials,
            "median_seconds": statistics.median(record["durations"]),
        }
        rows.append(row)
        print(f"n={n:3d} beta={beta:2d} success={successes}/{trials} "
              f"median={row['median_seconds']:.3f}s", flush=True)

    _write_csv(output, rows)
    if trial_output is not None:
        _write_csv(trial_output, trial_rows)
        print(f"Saved per-trial records: {trial_output}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["20:0", "40:0", "60:20", "70:30"],
                        help="n:blocksize (0 means plain LLL)")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--float-type", choices=["d", "ld"], default="d")
    parser.add_argument("--output", default="results/attack.csv")
    parser.add_argument("--trial-output", default=None,
                        help="optional CSV with per-instance identifiers and outcomes")
    args = parser.parse_args()
    try:
        cases = [tuple(map(int, case.split(":"))) for case in args.cases]
    except (ValueError, TypeError):
        parser.error("--cases entries must be n:blocksize integers")
    if any(len(case) != 2 for case in cases):
        parser.error("--cases entries must be n:blocksize integers")
    run(cases=cases, trials=args.trials, seed=args.seed, sigma=args.sigma,
        float_type=args.float_type, output=args.output, trial_output=args.trial_output)


if __name__ == "__main__":
    main()
