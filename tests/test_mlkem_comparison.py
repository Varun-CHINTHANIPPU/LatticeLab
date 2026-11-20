"""FIPS 203 parameter regression and algebraic structural demonstrations."""
import csv
import json

import pytest

from experiments.mlkem_comparison import PARAMETERS, compare, negacyclic_multiply, write_reports


def test_fips203_parameters():
    assert [p.name for p in PARAMETERS] == ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]
    assert [p.category for p in PARAMETERS] == [1, 3, 5]
    assert [p.k for p in PARAMETERS] == [2, 3, 4]
    assert [p.eta1 for p in PARAMETERS] == [3, 2, 2]
    assert [p.eta2 for p in PARAMETERS] == [2, 2, 2]
    assert [p.du for p in PARAMETERS] == [10, 10, 11]
    assert [p.dv for p in PARAMETERS] == [4, 4, 5]
    assert [p.rbg_bits for p in PARAMETERS] == [128, 192, 256]
    assert [p.ek_bytes for p in PARAMETERS] == [800, 1184, 1568]
    assert [p.dk_bytes for p in PARAMETERS] == [1632, 2400, 3168]
    assert [p.ct_bytes for p in PARAMETERS] == [768, 1088, 1568]
    assert all(p.n == 256 and p.q == 3329 and p.shared_secret_bytes == 32 for p in PARAMETERS)


def test_ring_relation_xn_equals_minus_one():
    # x^(n-1) * x = x^n = -1 mod (x^n+1)
    assert negacyclic_multiply([0, 0, 0, 1], [0, 1, 0, 0], 3329) == [3328, 0, 0, 0]


def test_ring_unit_and_distributivity():
    a, b, c = [1, 2, 3, 4], [4, 0, 1, 2], [3, 1, 0, 1]
    assert negacyclic_multiply(a, [1, 0, 0, 0]) == a
    lhs = negacyclic_multiply(a, [(x+y) % 3329 for x, y in zip(b, c)])
    rhs = [(x+y) % 3329 for x, y in zip(negacyclic_multiply(a, b), negacyclic_multiply(a, c))]
    assert lhs == rhs


def test_invalid_ring_inputs():
    with pytest.raises(ValueError):
        negacyclic_multiply([1], [1, 2])
    with pytest.raises(ValueError):
        negacyclic_multiply([], [])


def test_no_equivalence_claim():
    data = compare()
    assert data["toy"]["problem"] == "ordinary_LWE"
    assert all(row["problem"] == "Module_LWE" for row in data["ml_kem"])
    assert all("no equivalent LWE security" in row["comparison"] for row in data["ml_kem"])
    assert not any("security_bits" in row for row in data["ml_kem"])


def test_report_roundtrip(tmp_path):
    paths = write_reports(compare(70, 70, 3329, 3.0), tmp_path)
    assert all(path.is_file() for path in paths)
    data = json.loads(paths[0].read_text())
    assert data["toy"]["n"] == 70
    with paths[1].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert rows[2]["ct_bytes"] == "1568"


@pytest.mark.parametrize("kwargs", [{"toy_n": 0}, {"toy_m": 0}, {"toy_q": 1}, {"toy_sigma": 0}, {"toy_sigma": float("nan")}])
def test_reject_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        compare(**kwargs)
