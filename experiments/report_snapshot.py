"""Audit local research artifacts without fabricating missing measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FILES = {
    "raw_trials_seed_2029": "multiseed/seed_2029_trials.csv",
    "raw_trials_seed_2030": "multiseed/seed_2030_trials.csv",
    "raw_trials_seed_2031": "multiseed/seed_2031_trials.csv",
    "multi_seed_recovery": "multiseed/statistics/summary.csv",
    "multi_seed_paired_tests": "multiseed/statistics/paired_tests.csv",
    "multi_seed_runtime": "performance_multiseed/timing_summary.csv",
    "validation": "validation/report.json",
    "mlkem_reference": "mlkem_comparison/comparison.json",
    "sage_crosscheck": "sage_crosscheck.json",
    "toy_estimator": "estimator_toy.json",
}


def snapshot(root):
    root = Path(root)
    records = {}
    for label, relative in FILES.items():
        path = root / relative
        if not path.is_file():
            records[label] = {"status": "missing", "path": relative}
            continue
        raw = path.read_bytes()
        record = {"status": "present", "path": relative,
                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames:
                    raise ValueError(f"{path}: missing CSV header")
                record["rows"] = sum(1 for _ in reader)
                record["columns"] = reader.fieldnames
        else:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected JSON object")
            record["keys"] = sorted(data)
        records[label] = record
    return {"schema": 2, "source": "local files only",
            "artifacts": records,
            "missing": sorted(label for label, row in records.items()
                              if row["status"] == "missing")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results")
    parser.add_argument("--output", default="results/report_snapshot.json")
    args = parser.parse_args()
    try:
        data = snapshot(args.root)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Audited {len(data['artifacts'])} artifacts; missing: {len(data['missing'])}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
