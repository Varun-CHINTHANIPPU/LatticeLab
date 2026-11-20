"""Paired experiment metadata regression checks."""
import csv
import pytest
from experiments.stats import load_trials
BASE = ("seed", "n", "trial", "instance_id", "reduction", "block_size", "success", "seconds")
META = ("m", "q", "sigma", "float_type")
def write(path, seed, *, q=3329, sigma=3, modern=True, partial=False):
    fields = list(BASE) + (list(META[:1] if partial else META) if modern else [])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for beta in (0, 20):
            row = {"seed": seed, "n": 60, "trial": 0, "instance_id": f"public_{seed}",
                   "reduction": "LLL" if beta == 0 else "BKZ-20", "block_size": beta,
                   "success": 1, "seconds": 0.1}
            if modern:
                row["m"] = 60
                if not partial:
                    row.update(q=q, sigma=sigma, float_type="d")
            writer.writerow(row)
def test_new_metadata_accepted(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, 1)
    write(b, 2)
    assert len(load_trials([a, b])) == 2
@pytest.mark.parametrize("field,value", [("q", 17), ("sigma", 4), ("float_type", "ld"), ("m", 61)])
def test_mismatched_metadata_rejected(tmp_path, field, value):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, 1)
    write(b, 2)
    with b.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row[field] = value
    with b.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="inconsistent experiment parameters"):
        load_trials([a, b])
def test_mixed_schema_rejected(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, 1, modern=False)
    write(b, 2)
    with pytest.raises(ValueError, match="cannot mix legacy"):
        load_trials([a, b])
def test_partial_metadata_rejected(tmp_path):
    a = tmp_path / "a.csv"
    write(a, 1, partial=True)
    with pytest.raises(ValueError, match="partial parameter metadata"):
        load_trials([a])
def test_all_legacy_supported(tmp_path):
    a = tmp_path / "a.csv"
    write(a, 1, modern=False)
    assert len(load_trials([a])) == 1
def test_snapshot_lists_raw_inputs(tmp_path):
    from experiments.report_snapshot import FILES, snapshot
    for seed in (2029, 2030, 2031):
        assert f"raw_trials_seed_{seed}" in FILES
        assert f"raw_trials_seed_{seed}" in snapshot(tmp_path)["missing"]
