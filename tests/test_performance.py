"""Offline timing analysis: quantiles, paired bootstrap and input validation."""
import csv
import math

import pytest

from experiments.performance import analyze, bootstrap_median_ratio, quantile


FIELDS = ["seed", "n", "trial", "instance_id", "reduction", "block_size", "success", "seconds"]


def write_data(path, seed, n=60, count=5, zero=False):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for trial in range(count):
            for beta, seconds in ((0, trial + 1), (20, 2 * (trial + 1))):
                writer.writerow({"seed": seed, "n": n, "trial": trial,
                                 "instance_id": f"public_{seed}_{trial}",
                                 "reduction": "LLL" if beta == 0 else "BKZ-20",
                                 "block_size": beta, "success": 1,
                                 "seconds": 0 if zero and trial == 0 else seconds})


def test_quantile_interpolation():
    assert quantile([4, 1, 2, 3], 0.5) == 2.5
    assert quantile([4, 1, 2, 3], 0.25) == 1.75
    assert quantile([1], .95) == 1
    with pytest.raises(ValueError):
        quantile([], .5)


def test_paired_ratio_deterministic():
    a = [1., 2., 3., 4., 5.]
    b = [2., 4., 6., 8., 10.]
    result = bootstrap_median_ratio(a, b, replicates=200, seed=12)
    assert result == pytest.approx((0.5, 0.5, 0.5))
    assert result == bootstrap_median_ratio(a, b, replicates=200, seed=12)


def test_reports_and_multi_seed(tmp_path):
    paths = [tmp_path / f"seed_{s}.csv" for s in (1, 2)]
    for s, path in zip((1, 2), paths):
        write_data(path, s)
    summary, per_seed, comparisons = analyze(paths, tmp_path / "out", bootstrap=200, seed=3)
    assert len(summary) == 2
    assert len(per_seed) == 4
    assert summary[0]["trials"] == 10
    assert summary[0]["median_seconds"] == 3
    assert summary[0]["p95_seconds"] == pytest.approx(5)
    assert comparisons[0]["median_paired_left_over_right"] == pytest.approx(.5)
    assert comparisons[0]["left_faster_pairs"] == 10
    assert (tmp_path / "out" / "timing_summary.csv").exists()
    assert (tmp_path / "out" / "timing_per_seed.csv").exists()
    assert (tmp_path / "out" / "timing_paired.csv").exists()


def test_reject_nonpositive_time(tmp_path):
    path = tmp_path / "bad.csv"
    write_data(path, 1, zero=True)
    with pytest.raises(ValueError, match="strictly positive"):
        analyze([path], tmp_path / "out", bootstrap=200)


def test_reject_unpaired_data(tmp_path):
    path = tmp_path / "bad.csv"
    write_data(path, 1)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        analyze([path], tmp_path / "out", bootstrap=200)


def test_reject_invalid_bootstrap():
    with pytest.raises(ValueError):
        bootstrap_median_ratio([1], [0], 200)
    with pytest.raises(ValueError):
        bootstrap_median_ratio([1], [1], 99)
