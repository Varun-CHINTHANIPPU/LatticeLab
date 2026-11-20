"""Independent checks, mutation controls, and reproducibility."""
import hashlib
import json

import numpy as np
import pytest

from experiments.validation import independent_residual, run


def test_integer_oracle_handles_modular_wrap():
    A = np.array([[3328, 2], [1, 3328]], dtype=np.int64)
    s = np.array([-2, 3], dtype=np.int64)
    b = (A @ s + np.array([3, -4])) % 3329
    assert independent_residual(A, b, s, 3329) == [3, 3325]


def test_mutated_public_witness_rejected():
    A = np.array([[1, 2], [2, 3]], dtype=np.int64)
    s = np.array([1, 1], dtype=np.int64)
    b = np.array([3, 5], dtype=np.int64)
    baseline = independent_residual(A, b, s, 3329)
    b[0] += 1
    assert independent_residual(A, b, s, 3329) != baseline


def test_repeated_run_identical():
    first = run(seed=2028, trials=6, n=4, m=6)
    second = run(seed=2028, trials=6, n=4, m=6)
    assert first == second
    assert first["tamper_negative_controls"] == 6


def test_seed_changes_public_instances():
    assert run(seed=1, trials=5, n=4, m=6)["public_instance_digest"] != run(
        seed=2, trials=5, n=4, m=6)["public_instance_digest"]


def test_deterministic_serialization(tmp_path):
    data = run(seed=9, trials=4, n=3, m=5)
    first = json.dumps(data, sort_keys=True, indent=2) + "\n"
    second = json.dumps(run(seed=9, trials=4, n=3, m=5), sort_keys=True, indent=2) + "\n"
    assert hashlib.sha256(first.encode()).digest() == hashlib.sha256(second.encode()).digest()


@pytest.mark.parametrize("kwargs", [{"trials": 0}, {"seed": -1}, {"n": 0}, {"m": 2, "n": 3}, {"sigma": -1}])
def test_bad_validation_inputs(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)
