# LatticeLab — Learning With Errors, lattices, and attacks

A **reproducible educational cryptography lab** implementing exact Gram–Schmidt,
textbook LLL, Babai nearest-plane, Regev-style LWE bit encryption, and attacks on
deliberately weak instances.

> **Security warning:** This is textbook demonstration code. It is not constant-time,
> is not secure against active/CCA attackers, exposes error values to experiments,
> and MUST NOT be used for real encryption. Failed/timed-out attacks do not establish security.

## Quick start

```bash
git clone https://github.com/Varun-CHINTHANIPPU/LatticeLab
cd LatticeLab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# To enable the optional primal LLL/BKZ attacks:
python -m pip install -r requirements-attack.txt
make test
make demo
```

Dependencies: NumPy, pytest, matplotlib. **fpylll and cysignals
are optional** and are installed through `requirements-attack.txt`. The primal-attack demo and its tests skip
if fpylll is unavailable. On platforms without fpylll wheels, use conda-forge/SageMath.
`pytest` and `python -m pytest` both work from the repository root.

## Run experiments

```bash
python -m experiments.failrate --keys 200 --per-key 200 --seed 2026
python -m experiments.scale --trials 10 --cases 20:0 40:0 60:20 70:30
python -m experiments.plots
```

CSV files and PNGs are written under `results/`. The first measured sweep is committed as [`results/failrate.csv`](results/failrate.csv) and [`results/failrate.png`](results/failrate.png). The default failure-rate sweep
uses **200 independently sampled keys and 200 encryptions per key**, rather than
relying on a single atypical key. A 95% normal-approximation interval is formed
across per-key failure rates. The theoretical prediction is an *approximation*
for rounded Gaussian noise, not an exact finite-sample probability.

For the committed run, measured failure probabilities at sigma 40, 60 and 80 were
0.0083, 0.0772 and 0.1802, against normal-approximation predictions of
0.0093, 0.0829 and 0.1935 respectively. This is a repeatable experiment with
a fixed RNG seed, not independent evidence of cryptographic security.

`experiments.scale` uses a small secret, q=3329, m=n, and optional BKZ from
fpylll. The recovery heuristic requires the public experiment's `sigma` (or two explicit
norm bounds), enforces conservative Euclidean bounds on both error and small secret,
and ranks all plausible embedding rows instead of returning the first ±1 row.
**Bounds are heuristic, not proof that a candidate is the real secret.**
In particular for high noise and tiny square (m=n) instances, even uniform
random `b` may admit a comparably short explanation; no statistical filter
can guarantee rejection of all such inputs. The success metric examines **all**
plausible recovered rows and compares them against the planted secret (used
only for scoring experiments).
The reported successes, times and confidence depend on the configured trials,
platform, seed, and numerical implementation. Larger BKZ block sizes may need
fpylll's external pruning strategies.

## Project status and architecture

The core lab implements exact rational lattice arithmetic, toy Regev-style one-bit LWE, modular elimination, Babai decoding, and heuristic primal-embedding recovery with fpylll LLL/BKZ. The code is educational, not suitable for production encryption or ML-KEM cryptanalysis.

### End-to-end pipeline

1. Generate the public A,b together with secret s and noise e using lab/lwe.py.
2. Demonstrate one-bit encryption and analyze decryption failures with experiments/failrate.py.
3. Run modular elimination, tiny LLL/Babai decoding or a fpylll primal attack in lab/attacks.py.
4. Generate the same instance for LLL, BKZ-20 and BKZ-30 in experiments/scale.py and experiments/multiseed.py.
5. Record planted-secret recovery, public-instance hashes and wall-clock times in per-trial CSVs.
6. Reanalyze with experiments/stats.py and experiments/performance.py; visualize with experiments/figures.py.
7. Check a SageMath lattice invariant, estimate toy LWE costs, compare FIPS 203 structures and run negative controls.
8. Verify the availability of **raw trial data** and derived evidence with experiments/report_snapshot.py.

### Results

Multi-seed results (seeds 2029, 2030 and 2031); the raw trial CSVs are in `results/multiseed/`.

| Dimension (300 instances) | LLL | BKZ-20 | BKZ-30 |
|---|---:|---:|---:|
| n=60 | 10/300 (3.3%) | 297/300 (99.0%) | 300/300 (100.0%) |
| n=70 | 0/300 (0.0%) | 54/300 (18.0%) | 272/300 (90.7%) |

At n=70, BKZ-20 versus BKZ-30 produced paired discordances of 2 versus 220 and an **unadjusted exact McNemar p-value of approximately 7.345e-63**. At n=60, BKZ-20 versus BKZ-30 gave p=0.25. These are observations on deliberately weak ordinary LWE, not evidence about standardized ML-KEM security.

### Repository map

- lab/lattice.py — Fraction-based Gram–Schmidt, LLL, reducedness and Babai.
- lab/lwe.py — small LWE key generation and textbook one-bit encryption/decryption.
- lab/attacks.py — modular elimination, tiny decoding and primal-embedding recovery.
- experiments/scale.py and multiseed.py — paired attack trial generation.
- experiments/stats.py, performance.py, figures.py — offline inference and plots.
- experiments/validation.py and report_snapshot.py — controlled checks and evidence inventory.
- experiments/mlkem_comparison.py — NIST FIPS 203 structural parameters.
- integrations/ — optional SageMath and Lattice Estimator.
- tests/ — regression and negative-control tests.

See the [Reproducibility Guide](docs/REPRODUCIBILITY.md).

### Key limitations

- Ordinary LWE and the toy rounded-Gaussian secret/error model are not equivalent to Module-LWE or ML-KEM.
- Embedding candidates are heuristic. The planted secret is used to score success, not supplied to the attacker.
- Noisy square systems can admit short misleading explanations; attack failure is not proof of security.
- Sequential fixed-order timing measurements are descriptive and hardware-dependent.
- The SageMath check covers one small basis; the estimator uses an approximate mathematical model.
- New trial CSVs record m, q, sigma and float_type and reject cross-run mismatches. Historical seed CSVs lack these fields and remain accepted only as **all-legacy** inputs.

---
## Paired LLL/BKZ experiments

The scaling harness now generates each LWE instance **once per dimension and
trial** and runs every requested reduction on that same instance. Thus
`60:0 60:20 60:30` compares LLL, BKZ-20, and BKZ-30 on matched instances,
not separately generated datasets. Dimension-specific RNG streams derive from
`(seed, n)`, so adding or reordering configurations does not change the
instances for an existing dimension. Timings are sequential and are not
hardware-isolated; no speedup or security claim follows from a single run.

```bash
python -m experiments.scale --trials 100 --seed 2028 --sigma 3 \
  --cases 50:0 50:20 60:0 60:20 60:30 70:0 70:20 70:30 \
  --output results/paired_100.csv \
  --trial-output results/paired_100_trials.csv
python -m experiments.plots --attack results/paired_100.csv --outdir results/paired
```

The aggregate CSV retains the original column format. The optional trial CSV
records `seed,n,trial,instance_id,reduction,block_size,success,seconds`.
Newly generated trial CSVs also record `m,q,sigma,float_type` for checking cross-run consistency.
Rows with the same `(seed,n,trial)` have identical public-instance fingerprints
across algorithms. The fingerprint hashes only public `A,b`, not the secret.
Success means **any plausible candidate** matches the planted secret; the
planted secret is used for scoring only. The estimator/ML-KEM milestone is
separate, and these toy attacks do not assess real ML-KEM security.

## Statistical validation

Analyze the existing paired trial CSV **without rerunning attacks**:

```bash
python -m experiments.stats results/paired_100_trials.csv --outdir results/stats_2028
```

Outputs: `summary.csv` (pooled recovery rate and two-sided Wilson 95% CI),
`per_seed.csv` (individual seed rates and intervals), and
`paired_tests.csv` (paired contingency counts and exact two-sided McNemar
p-values). McNemar uses only discordant pairs, not independent-sample tests.
Reported p-values are **unadjusted for multiple comparisons**, exploratory,
and not evidence of cryptographic security. Wilson intervals describe sampling
uncertainty for each recovery rate; they do not correct systematic bias or
account for changes in attack implementation.

For an additional **multi-seed** study, run:

```bash
python -m experiments.multiseed --seeds 2029 2030 2031 --trials 100 \
  --cases 60:0 60:20 60:30 70:0 70:20 70:30 \
  --outdir results/multiseed
```

Each seed is a separate reproducible experiment. The script saves one aggregate
and one per-trial CSV per seed and analyzes the combined trial files. The
statistical loader rejects duplicate records, missing algorithms within a
paired instance, conflicting public fingerprints, and unequal trial indices
between seeds. Historical legacy CSVs omit parameter metadata, so confirm their q/sigma/settings
from their generation logs. Newly generated trial CSVs include m, q, sigma and
float_type; the loader rejects inconsistent modern inputs and mixed old/new schemas. Seed-level rows
are retained to inspect variability rather than interpreting pooled Wilson
intervals as accounting for all sources of heterogeneity.

## Offline runtime/performance analysis

**No new lattice attacks are required.** Analyze the existing per-trial files:

```bash
python -m experiments.performance results/paired_100_trials.csv \
  --outdir results/performance_2028
python -m experiments.performance \
  results/multiseed/seed_2029_trials.csv \
  results/multiseed/seed_2030_trials.csv \
  results/multiseed/seed_2031_trials.csv \
  --outdir results/performance_multiseed --bootstrap 2000 --seed 7
```

Produces `timing_summary.csv` (mean, median, sample standard deviation,
minimum, p05/p25/p75/p95, maximum), `timing_per_seed.csv`, and
`timing_paired.csv` (median of per-instance time ratios and a deterministic
95% percentile bootstrap interval using paired-instance resampling).
A ratio `left/right > 1` means the right algorithm was faster. These are
**wall-clock timings** from the original sequential fixed-order sweeps;
their confidence intervals describe instance-sampling variability only.
They do not correct CPU frequency scaling, background load, thermal throttling,
order effects, or correlations within a run. Do not interpret them as
hardware-independent speedups or rigorous causal benchmarks.

For a future controlled benchmark: pin the process to one core if appropriate,
record CPU governor and temperature, warm up the runtime, randomize algorithm
execution order *within each instance*, repeat across sessions, and document
software/hardware versions. Such controls require new measurements and are
separate from this offline analysis.

## Publication-style figures

Uses the existing statistics and timing CSVs. No attacks are rerun.

```bash
python -m experiments.figures \
  --statistics results/multiseed/statistics/summary.csv \
  --timing results/performance_multiseed/timing_summary.csv \
  --outdir results/figures_multiseed --formats png pdf svg
```

Outputs three plots in each requested format:
`recovery_wilson` (recovery fractions and Wilson 95% intervals),
`timing_quantiles` (median, IQR band, p05–p95 whiskers), and
`recovery_runtime` (recovery-versus-median-wall-clock trade-off).
PNG at 240 dpi, vector PDF and SVG are available. CSV validation ensures
matching configurations and seed/trial counts, finite values, valid intervals,
and ordered quantiles. Matplotlib uses a headless backend for CI.

Figure intervals reflect sampling of weak toy-LWE instances. Timing intervals
are empirical quantiles, **not** 95% confidence intervals; timing data came
from fixed-order, non-isolated runs. No security claims about ML-KEM follow.

## SageMath and Lattice Estimator

These are **optional external integrations**, not Python virtualenv dependencies.
Install SageMath 10.10 or a compatible version; clone the official
`https://github.com/malb/lattice-estimator` independently. Tested locally by
the project maintainer with SageMath 10.10 and estimator commit `7d59f86`
before integration; integration scripts require separate verification.

```bash
python -m integrations.sage_runner crosscheck --output results/sage_crosscheck.json
python -m integrations.sage_runner estimate \
  --estimator-path ~/cryptography-tools/lattice-estimator \
  --n 40 --m 40 --q 3329 --sigma 3 --secret-bound 1 \
  --output results/estimator_toy.json
python -m pytest -q -rs
```

The first command independently checks SageMath LLL lattice preservation for
an exact integer basis (unimodular change of basis). It does not independently
revalidate all our attack-recovery experiments. The second calls the upstream
Lattice Estimator using an explicit import path and writes machine-readable
log2 heuristic work factors; no `pip install estimator` is needed.

**Model distinction:** the toy experiment uses NumPy-rounded normal errors
and small secrets, whereas `ND.DiscreteGaussian` is an estimator model, not
an exact match to the sampled error law. Lattice Estimator outputs are
heuristic model estimates, **not** measured wall-clock times or a validation
of attack success. These parameters are ordinary toy LWE with `m=n`,
not ML-KEM module-LWE. Do not compare their estimated bit-security to ML-KEM
as if they represented the same cryptosystem. Pin the upstream estimator
commit when reproducing figures or reports.

The SageMath launcher uses `sage -c` because some Linux SageMath packages
reject options supplied after `.sage` filenames. Use
`python -m integrations.sage_runner` to forward arguments safely. Activate
the project virtual environment only if it exists; this wrapper works with the
Python interpreter used to run the rest of the project.

## FIPS 203 ML-KEM structural comparison

```bash
python -m experiments.mlkem_comparison --toy-n 40 --toy-m 40 \
  --toy-q 3329 --toy-sigma 3 --outdir results/mlkem_comparison
python -m pytest -q -rs
cat results/mlkem_comparison/comparison.json
cat results/mlkem_comparison/mlkem_parameters.csv
```

This is a **reproducible reference-data and algebra demonstration**, not an
ML-KEM implementation, benchmark, or ML-KEM security estimation. The
`comparison.json` report records toy ordinary-LWE parameters and three
standardized ML-KEM parameter sets from **NIST FIPS 203 Section 8 Tables 2–3**
(https://doi.org/10.6028/NIST.FIPS.203). The CSV includes module ranks,
centered-binomial distribution parameters, compression widths, key and
ciphertext sizes, and NIST security categories. Tests verify the standard's
tabulated values and the negacyclic relation `X^n = -1` in
`Z_q[X]/(X^n+1)` using small illustrative polynomials.

**Interpretation:** our experiments use unstructured ordinary LWE with
rounded continuous-Gaussian errors, small secrets, and `m=n`; ML-KEM uses
Module-LWE over a polynomial quotient ring, centered-binomial noise, and
KEM-specific algorithms. The count `256*k` is a coefficient count, not a
security-equivalent ordinary-LWE dimension. Security categories 1/3/5 are
standardized target categories, **not** independently calculated bit-security
figures. Do not compare our toy estimator's `log2_rop` or wall-clock
timings directly to ML-KEM categories or claim attacks on ML-KEM. No full
ML-KEM KEM implementation, side-channel analysis, or realistic security
estimation is attempted in this task.

## Independent invariants and negative controls

```bash
python -m experiments.validation --seed 2032 --trials 24 --n 8 --m 12 \
  --q 3329 --sigma 3 --output results/validation/report.json
python -m pytest -q -rs
sha256sum results/validation/report.json
python -m experiments.validation --seed 2032 --trials 24 --n 8 --m 12 \
  --q 3329 --sigma 3 --output results/validation/report.json
sha256sum results/validation/report.json
```

The independent residual oracle uses Python-integer summation instead of
NumPy matrix multiplication. Each trial checks `b - A*s = e (mod q)`,
rejects a one-coordinate public-vector mutation against the original witness,
and hashes the public instance. Separate controls verify an exact noiseless
modular system, reject an inconsistent overdetermined system, and check
our own LLL output against exact rational reducedness conditions.
Unit tests verify seed repeatability, seed sensitivity, tampering,
modular wraparound, and malformed inputs.

**Scope:** witness-based mutation rejection is not a cryptographic
authenticity test. The suite does not establish unique noisy-LWE solutions,
does not independently certify the primal attack's recovery success, and
does not rerun costly BKZ benchmarks. SageMath's independent LLL
unimodularity check is covered in the SageMath section; further cross-implementation
comparison of many randomized bases and adversarial negative controls
would be required for stronger external validation. Hashes should match
for repeat runs with the same software environment; pin NumPy versions
for cross-environment bit-for-bit claims.

## Provenance audit

See [the reproducibility protocol](docs/REPRODUCIBILITY.md). Audit the available evidence:

```bash
python -m experiments.report_snapshot --root results \
  --output results/report_snapshot.json
python -m pytest -q -rs
```

The snapshot includes three raw seed trial inputs and seven derived reports,
records checksums and row counts, and highlights missing sources. This is an educational experimental
report, not a new attack on ML-KEM or a cryptographic security proof.
