"""Artifact provenance checks."""
import csv
import hashlib
import json

import pytest

from experiments.report_snapshot import FILES, snapshot


def test_missing_artifacts_explicit(tmp_path):
    result = snapshot(tmp_path)
    assert result["missing"] == sorted(FILES)
    assert all(item["status"] == "missing" for item in result["artifacts"].values())


def test_hash_and_csv_row_count(tmp_path):
    p = tmp_path / FILES["multi_seed_recovery"]
    p.parent.mkdir(parents=True)
    p.write_text("n,reduction,rate\n60,LLL,0.1\n70,BKZ-20,0.2\n")
    report = snapshot(tmp_path)
    row = report["artifacts"]["multi_seed_recovery"]
    assert row["rows"] == 2
    assert row["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert "multi_seed_recovery" not in report["missing"]


def test_json_metadata_without_claiming_validation(tmp_path):
    p = tmp_path / FILES["validation"]
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"status": "failed", "seed": 1}))
    report = snapshot(tmp_path)
    assert report["artifacts"]["validation"]["status"] == "present"
    assert report["artifacts"]["validation"]["keys"] == ["seed", "status"]


def test_malformed_json_rejected(tmp_path):
    p = tmp_path / FILES["validation"]
    p.parent.mkdir(parents=True)
    p.write_text("not json")
    with pytest.raises(ValueError):
        snapshot(tmp_path)


def test_empty_csv_header_rejected(tmp_path):
    p = tmp_path / FILES["multi_seed_runtime"]
    p.parent.mkdir(parents=True)
    p.write_text("")
    with pytest.raises(ValueError):
        snapshot(tmp_path)


def test_snapshot_repeatable(tmp_path):
    assert snapshot(tmp_path) == snapshot(tmp_path)
