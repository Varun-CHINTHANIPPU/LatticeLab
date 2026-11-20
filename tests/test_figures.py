"""Tests for figures and validation; no attacks executed."""
import csv
from pathlib import Path

import pytest

from experiments.figures import load_data, render


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path):
    stats = tmp_path / "stats.csv"
    times = tmp_path / "times.csv"
    s, t = [], []
    for n in (60, 70):
        for label in ("LLL", "BKZ-20", "BKZ-30"):
            rate = {"LLL": .1, "BKZ-20": .5, "BKZ-30": .9}[label]
            median = {"LLL": .2, "BKZ-20": .5, "BKZ-30": 1.1}[label]
            s.append({"n": n, "reduction": label, "seeds": 3, "trials": 300,
                      "rate": rate, "wilson95_low": rate - .05, "wilson95_high": rate + .05})
            t.append({"n": n, "reduction": label, "seeds": 3, "trials": 300,
                      "median_seconds": median, "p05_seconds": median * .8,
                      "p25_seconds": median * .9, "p75_seconds": median * 1.1,
                      "p95_seconds": median * 1.2})
    _write(stats, s)
    _write(times, t)
    return stats, times


def test_validate_data(tmp_path):
    s, t = _inputs(tmp_path)
    stats, times = load_data(s, t)
    assert len(stats) == len(times) == 6


def test_render_all_formats(tmp_path):
    pytest.importorskip("matplotlib")
    s, t = _inputs(tmp_path)
    outputs = render(s, t, tmp_path / "out", ("png", "pdf", "svg"))
    assert len(outputs) == 9
    assert all(path.is_file() and path.stat().st_size > 100 for path in outputs)
    assert (tmp_path / "out" / "recovery_wilson.pdf").read_bytes().startswith(b"%PDF")


def test_reject_incompatible_counts(tmp_path):
    s, t = _inputs(tmp_path)
    rows = list(csv.DictReader(t.open()))
    rows[0]["trials"] = 200
    _write(t, rows)
    with pytest.raises(ValueError, match="trial/seed counts differ"):
        load_data(s, t)


def test_reject_invalid_intervals(tmp_path):
    s, t = _inputs(tmp_path)
    rows = list(csv.DictReader(s.open()))
    rows[0]["wilson95_low"] = .9
    _write(s, rows)
    with pytest.raises(ValueError, match="invalid confidence interval"):
        load_data(s, t)


def test_reject_missing_configuration(tmp_path):
    s, t = _inputs(tmp_path)
    rows = list(csv.DictReader(t.open()))[:-1]
    _write(t, rows)
    with pytest.raises(ValueError, match="configuration sets differ"):
        load_data(s, t)
