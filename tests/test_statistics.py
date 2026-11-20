"""Reference checks for Wilson intervals, exact McNemar, and paired CSV validation."""
import csv
import math
import pytest

from experiments.stats import wilson_interval, mcnemar_exact, load_trials, summarize


def _csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "n", "trial", "instance_id", "reduction", "block_size", "success", "seconds"])
        writer.writeheader()
        writer.writerows(rows)


def _row(seed, trial, beta, success, instance=None):
    return {"seed": seed, "n": 60, "trial": trial, "instance_id": instance or f"public_{seed}_{trial}",
            "reduction": "LLL" if beta == 0 else f"BKZ-{beta}", "block_size": beta,
            "success": int(success), "seconds": 0.1}


def test_wilson_known_values_and_edges():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038315303659956, abs=1e-9)
    assert hi == pytest.approx(0.5961684696340044, abs=1e-9)
    assert wilson_interval(0, 100)[0] == 0
    assert wilson_interval(100, 100)[1] == 1
    assert wilson_interval(0, 100)[1] > 0
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_mcnemar_exact_known_cases():
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(2, 2) == 1
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(1, 5) == pytest.approx(0.21875)
    assert mcnemar_exact(0, 100) == pytest.approx(2 ** -99)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_summary_exact_paired_contingency(tmp_path):
    path = tmp_path / "paired.csv"
    rows = []
    for trial, (a, b) in enumerate([(1, 1), (1, 0), (0, 1), (0, 1), (0, 0)]):
        rows.extend([_row(7, trial, 0, a), _row(7, trial, 20, b)])
    _csv(path, rows)
    summary, per_seed, comparisons = summarize([path], output_dir=tmp_path / "out")
    assert [(r["reduction"], r["successes"]) for r in summary] == [("LLL", 2), ("BKZ-20", 3)]
    assert len(per_seed) == 2
    comparison = comparisons[0]
    assert (comparison["both_success"], comparison["left_only"],
            comparison["right_only"], comparison["neither_success"]) == (1, 1, 2, 1)
    assert comparison["mcnemar_exact_p"] == 1.0
    assert (tmp_path / "out" / "paired_tests.csv").exists()


def test_multiple_seeds_pool_and_keep_separate(tmp_path):
    paths = []
    for seed in (10, 11):
        path = tmp_path / f"{seed}.csv"
        rows = []
        for trial in range(3):
            rows.extend([_row(seed, trial, 0, trial == 0), _row(seed, trial, 20, True)])
        _csv(path, rows)
        paths.append(path)
    summary, per_seed, comparisons = summarize(paths, tmp_path / "stats")
    assert len(per_seed) == 4
    assert summary[0]["trials"] == 6
    assert summary[0]["successes"] == 2
    assert summary[1]["successes"] == 6
    assert comparisons[0]["right_only"] == 4
    assert comparisons[0]["mcnemar_exact_p"] == pytest.approx(0.125)


@pytest.mark.parametrize("problem", ["fingerprint", "missing", "duplicate", "bad_success", "bad_time"])
def test_invalid_paired_data_rejected(tmp_path, problem):
    path = tmp_path / "invalid.csv"
    rows = [_row(1, 0, 0, 1), _row(1, 0, 20, 1)]
    if problem == "fingerprint":
        rows[1]["instance_id"] = "different"
    elif problem == "missing":
        rows.pop()
        rows.extend([_row(1, 1, 0, 1), _row(1, 1, 20, 1)])
    elif problem == "duplicate":
        rows.append(rows[0].copy())
    elif problem == "bad_success":
        rows[0]["success"] = 2
    elif problem == "bad_time":
        rows[0]["seconds"] = math.nan
    _csv(path, rows)
    if problem == "missing":
        # Another seed with incomplete algorithm set for one trial.
        rows.pop(1)
        _csv(path, rows)
    with pytest.raises(ValueError):
        load_trials([path])
