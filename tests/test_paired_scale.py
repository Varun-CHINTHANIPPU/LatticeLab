"""Regression tests for paired, reproducible LLL/BKZ experiments."""
import csv

import numpy as np
import pytest

from experiments import scale


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_same_instance_used_for_all_reductions(tmp_path, monkeypatch):
    generated = []
    attacked = []

    def fake_keygen(params, rng):
        tag = len(generated) + 1
        A = np.array([[tag, 0], [0, tag]], dtype=np.int64)
        b = np.array([tag, tag], dtype=np.int64)
        secret = np.array([1, 1], dtype=np.int64)
        generated.append((A.copy(), b.copy()))
        return (A, b), secret, np.zeros(2, dtype=np.int64)

    def fake_attack(A, b, q, **kwargs):
        attacked.append((A.copy(), b.copy(), kwargs["block_size"]))
        assert kwargs["return_candidates"] is True
        return [np.array([1, 1], dtype=np.int64)]

    monkeypatch.setattr(scale, "keygen", fake_keygen)
    monkeypatch.setattr(scale, "primal_attack", fake_attack)
    aggregate = tmp_path / "aggregate.csv"
    detail = tmp_path / "trials.csv"
    rows = scale.run(cases=[(2, 0), (2, 20), (2, 30)], trials=4,
                     output=aggregate, trial_output=detail)
    assert len(generated) == 4  # NOT 12 independently sampled instances
    assert len(attacked) == 12
    for i in range(4):
        for j, beta in enumerate((0, 20, 30)):
            A, b, actual_beta = attacked[3 * i + j]
            assert actual_beta == (beta or None)
            assert np.array_equal(A, generated[i][0])
            assert np.array_equal(b, generated[i][1])
    assert [row["successes"] for row in rows] == [4, 4, 4]
    details = _read(detail)
    assert len(details) == 12
    for i in range(4):
        assert len({r["instance_id"] for r in details[3 * i:3 * i + 3]}) == 1
    assert len({details[3 * i]["instance_id"] for i in range(4)}) == 4
    assert len(_read(aggregate)) == 3


def test_case_order_does_not_change_instances(tmp_path, monkeypatch):
    observed = []

    def fake_keygen(params, rng):
        value = int(rng.integers(1, 1_000_000))
        A = np.array([[value]], dtype=np.int64)
        b = np.array([value], dtype=np.int64)
        observed.append((params.n, value))
        return (A, b), np.array([0]), np.array([0])

    def fake_attack(A, b, q, **kwargs):
        return []

    monkeypatch.setattr(scale, "keygen", fake_keygen)
    monkeypatch.setattr(scale, "primal_attack", fake_attack)
    scale.run(cases=[(2, 0), (3, 0)], trials=3, seed=2027,
              output=tmp_path / "a.csv")
    first = list(observed)
    observed.clear()
    scale.run(cases=[(3, 20), (2, 20), (2, 0)], trials=3, seed=2027,
              output=tmp_path / "b.csv")
    second = list(observed)
    for n in (2, 3):
        assert [v for dim, v in first if dim == n] == [v for dim, v in second if dim == n]


def test_paired_success_is_scored_per_algorithm(tmp_path, monkeypatch):
    def fake_keygen(params, rng):
        return (np.eye(2, dtype=np.int64), np.ones(2, dtype=np.int64)), np.ones(2, dtype=np.int64), np.zeros(2)
    def fake_attack(A, b, q, **kwargs):
        return [np.ones(2, dtype=np.int64)] if kwargs["block_size"] == 20 else []
    monkeypatch.setattr(scale, "keygen", fake_keygen)
    monkeypatch.setattr(scale, "primal_attack", fake_attack)
    rows = scale.run(cases=[(2, 0), (2, 20)], trials=3,
                     output=tmp_path / "aggregate.csv", trial_output=tmp_path / "trials.csv")
    assert [r["successes"] for r in rows] == [0, 3]
    assert [int(r["success"]) for r in _read(tmp_path / "trials.csv")] == [0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("cases", [[], [(2, 0), (2, 0)], [(2, 1)], [(0, 0)]])
def test_invalid_cases_rejected(tmp_path, cases):
    with pytest.raises(ValueError):
        scale.run(cases=cases, output=tmp_path / "out.csv")
