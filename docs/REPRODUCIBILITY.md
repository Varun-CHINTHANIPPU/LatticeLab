# Reproducibility protocol

Run from the repository root. Commands are intentionally separated into a **fast smoke-test path** and a **costly full-attack path**.

## 1. Environment and provenance (1–5 minutes)

```bash
git rev-parse HEAD
python --version
python -m pip freeze > results_environment_pip_freeze.txt
uname -a
python -m pytest -q -rs
```

Optional primal attacks need `fpylll` and `cysignals`; SageMath and the upstream estimator are external. Save the Git commit, dependency versions, CPU details, and OS with every new experiment.

## 2. Fast validation and repeatability (15–60 seconds)

```bash
python -m experiments.validation --seed 2032 --trials 24 --n 8 --m 12 \
  --q 3329 --sigma 3 --output results/validation/report.json
sha256sum results/validation/report.json
python -m experiments.validation --seed 2032 --trials 24 --n 8 --m 12 \
  --q 3329 --sigma 3 --output results/validation/report.json
sha256sum results/validation/report.json
python -m experiments.mlkem_comparison --outdir results/mlkem_comparison
python -m experiments.report_snapshot --root results --output results/report_snapshot.json
```

The two validation hashes should match in the same environment. `report_snapshot` inventories the three raw seed trial CSVs and seven derived reports. It records presence, checksums and basic CSV/JSON parseability but does **not** certify the underlying measurements.

## 3. Optional external checks (1–3 minutes)

```bash
python -m integrations.sage_runner crosscheck --output results/sage_crosscheck.json
python -m integrations.sage_runner estimate \
  --estimator-path ~/cryptography-tools/lattice-estimator \
  --n 40 --m 40 --q 3329 --sigma 3 --secret-bound 1 \
  --output results/estimator_toy.json
git -C ~/cryptography-tools/lattice-estimator rev-parse HEAD
```

Estimator outputs are heuristic toy-LWE models, not ML-KEM security estimates.

## 4. Reanalyze existing paired data without rerunning attacks (under 2 minutes)

```bash
ls -lh results/multiseed/seed_{2029,2030,2031}_trials.csv
python -m experiments.stats \
  results/multiseed/seed_2029_trials.csv \
  results/multiseed/seed_2030_trials.csv \
  results/multiseed/seed_2031_trials.csv \
  --outdir results/multiseed/statistics
python -m experiments.performance \
  results/multiseed/seed_2029_trials.csv \
  results/multiseed/seed_2030_trials.csv \
  results/multiseed/seed_2031_trials.csv \
  --outdir results/performance_multiseed --bootstrap 2000 --seed 7
python -m experiments.figures \
  --statistics results/multiseed/statistics/summary.csv \
  --timing results/performance_multiseed/timing_summary.csv \
  --outdir results/figures_multiseed --formats png pdf svg
python -m experiments.report_snapshot --root results --output results/report_snapshot.json
```

The paired-trial loader validates instance fingerprints, duplicates and missing algorithm records. New trial CSVs now include `m,q,sigma,float_type`, and the loader rejects mismatched parameters at the same dimension. Original legacy seed CSVs do **not** contain these fields: their metadata cannot be checked retrospectively, so verify their q/sigma and reduction settings from the original run logs. Mixing legacy and new trial CSVs is rejected.

## 5. Full experiment rerun (potentially hours)

Only if you intentionally want new expensive attacks:

```bash
python -m experiments.multiseed --seeds 2029 2030 2031 --trials 100 \
  --cases 60:0 60:20 60:30 70:0 70:20 70:30 \
  --outdir results/multiseed
```

Expect runtime to depend heavily on CPU, BKZ settings, software, and thermal conditions. Preserve existing outputs before rerunning. Check `python -m experiments.multiseed --help` for installed CLI options.

## 6. Interpretation and publication checklist

- Confirm exact Git SHA, NumPy/fpylll/Sage versions, estimator SHA, CPU and OS.
- Archive raw per-trial files before treating aggregate claims as independently reproducible.
- Recompute paired discordant counts and unadjusted McNemar tests.
- Label Wilson confidence intervals separately from empirical timing quantiles.
- Never equate toy ordinary LWE with ML-KEM's Module-LWE or security categories.
- Never present attack failures as security guarantees.
- Avoid claims of publication-grade timing without randomized order, isolation, and repeated sessions.
