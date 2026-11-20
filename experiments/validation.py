"""Deterministic independent invariants and negative controls for toy LWE.

No expensive BKZ runs. All assertions are independent of attack success.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from lab.lwe import Params, keygen
from lab.attacks import solve_mod_q
from lab.lattice import lll, is_lll_reduced
from experiments.scale import _instance_id


def independent_residual(A, b, s, q):
    """Compute residual with Python integers, not NumPy matrix multiplication."""
    return [((int(b[i]) - sum(int(A[i, j]) * int(s[j])
             for j in range(A.shape[1]))) % q) for i in range(A.shape[0])]


def run(seed=2032, trials=24, n=8, m=12, q=3329, sigma=3.0):
    if not isinstance(seed, int) or seed < 0 or not isinstance(trials, int) or trials < 1:
        raise ValueError("seed must be nonnegative and trials positive")
    params = Params(n=n, m=m, q=q, sigma=sigma, secret="small")
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, m]))
    fingerprints = []
    noisy_rejections = 0
    tamper_rejections = 0
    for _ in range(trials):
        (A, b), secret, error = keygen(params, rng)
        residual = independent_residual(A, b, secret, q)
        assert residual == [int(x) % q for x in error], "LWE equation mismatch"
        fingerprints.append(_instance_id(A, b))
        # Explicitly constructed negative control: a modified public vector
        # is inconsistent with the original secret/error witness.
        corrupted = b.copy()
        corrupted[0] = (int(corrupted[0]) + 1) % q
        if independent_residual(A, corrupted, secret, q) != residual:
            tamper_rejections += 1
        # A noisy b generally differs from A*s; test the witness equation,
        # not whether another secret might also explain the same b.
        if any(x != 0 for x in residual):
            noisy_rejections += 1
    if tamper_rejections != trials:
        raise AssertionError("negative-control mutation was not detected")
    if len(set(fingerprints)) != trials:
        raise AssertionError("duplicate public-instance fingerprint in seeded run")
    # Independent algebraic oracle for noiseless elimination on an invertible
    # triangular matrix, and an inconsistent overdetermined negative control.
    A = np.array([[1, 1], [0, 1], [0, 0]], dtype=np.int64)
    s = np.array([2, 3], dtype=np.int64)
    b = (A @ s) % q
    solved = solve_mod_q(A, b, q)
    if solved is None or not np.array_equal(solved % q, s % q):
        raise AssertionError("noiseless modular elimination failed")
    inconsistent = b.copy()
    inconsistent[2] = 1
    if solve_mod_q(A, inconsistent, q) is not None:
        raise AssertionError("inconsistent system accepted")
    basis = [[4, 1, 3], [2, 1, 1], [1, 0, 2]]
    reduced = lll(basis)
    if not is_lll_reduced(reduced):
        raise AssertionError("own LLL output fails exact reducedness check")
    # Stable fingerprints do not depend on timings or platform floating-point
    # benchmarks, but NumPy RNG algorithms can change across versions.
    digest = hashlib.sha256("|".join(fingerprints).encode("ascii")).hexdigest()
    return {"status": "passed", "seed": seed, "trials": trials,
            "parameters": {"n": n, "m": m, "q": q, "sigma": sigma},
            "lwe_witness_checks": trials, "tamper_negative_controls": tamper_rejections,
            "nonzero_error_witnesses": noisy_rejections,
            "inconsistent_modular_system_rejected": True,
            "noiseless_modular_system_solved": True,
            "own_lll_exact_reducedness": True,
            "public_instance_digest": digest,
            "limitations": [
                "Tamper control checks a known witness, not uniqueness of LWE solutions.",
                "Noisy witness rejection is not a proof that modular elimination cannot find another solution.",
                "Exact LLL reducedness alone does not independently establish lattice equivalence.",
                "This lightweight suite does not rerun full-scale LLL/BKZ attacks.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--q", type=int, default=3329)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--output", default="results/validation/report.json")
    args = parser.parse_args()
    try:
        result = run(args.seed, args.trials, args.n, args.m, args.q, args.sigma)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (ValueError, AssertionError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
