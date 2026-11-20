"""CLI and API regression tests for experiment validation."""
import sys
import pytest
from experiments.scale import run as run_scale
from experiments.plots import main as plot_main


def test_scale_rejects_no_cases(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        run_scale(cases=[], trials=2, output=tmp_path / "attack.csv")


def test_plots_reports_missing_csvs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["plots", "--failrate", str(tmp_path / "missing1.csv"),
                                     "--attack", str(tmp_path / "missing2.csv")])
    with pytest.raises(SystemExit) as exc:
        plot_main()
    assert exc.value.code == 2
    assert "no input CSV found" in capsys.readouterr().err


def test_scale_counts_ground_truth_in_later_candidate(tmp_path, monkeypatch):
    import numpy as np
    from experiments import scale
    def fixed_keygen(params, rng):
        A = np.eye(2, dtype=np.int64)
        true_s = np.array([1, 1], dtype=np.int64)
        return (A, true_s.copy()), true_s, np.zeros(2, dtype=np.int64)
    def fake_attack(A, b, q, **kwargs):
        assert kwargs["return_candidates"] is True
        assert kwargs["sigma"] == 2.0
        return [np.array([0, 0]), np.array([1, 1])]
    monkeypatch.setattr(scale, "keygen", fixed_keygen)
    monkeypatch.setattr(scale, "primal_attack", fake_attack)
    rows = scale.run(cases=[(2, 0)], trials=1, sigma=2.0,
                     output=tmp_path / "attack.csv")
    assert rows[0]["successes"] == 1
