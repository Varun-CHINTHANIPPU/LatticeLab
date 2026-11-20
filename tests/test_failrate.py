import csv
import math
from experiments.failrate import predicted_failure, per_key_prediction, run


def test_prediction_increases_with_error():
    assert predicted_failure(3329, 128, 0) == 0
    assert 0 < predicted_failure(3329, 128, 40) < predicted_failure(3329, 128, 80) < 1
    assert per_key_prediction(3329, [0] * 10) == 0


def test_small_reproducible_sweep(tmp_path):
    target = tmp_path / "rates.csv"
    first = run(sigmas=(8, 40), keys=4, per_key=8, seed=77, output=target)
    second = run(sigmas=(8, 40), keys=4, per_key=8, seed=77, output=target)
    assert first == second
    with target.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert math.isfinite(first[0]["measured"])


def test_sigma_csv_spelling_matches_cli_floats(tmp_path):
    a = tmp_path / "integer-sigmas.csv"
    b = tmp_path / "float-sigmas.csv"
    run(sigmas=(8, 40), keys=3, per_key=4, seed=7, output=a)
    run(sigmas=(8.0, 40.0), keys=3, per_key=4, seed=7, output=b)
    assert a.read_bytes() == b.read_bytes()
    with b.open(newline="") as f:
        assert [row["sigma"] for row in csv.DictReader(f)] == ["8", "40"]


def test_empty_sigmas_raise_clear_error(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="at least one sigma"):
        run(sigmas=[], keys=3, per_key=4, output=tmp_path / "empty.csv")
