"""Run several independent seeded paired sweeps and analyze them."""
import argparse
from pathlib import Path

from experiments.scale import run
from experiments.stats import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2028, 2029, 2030])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--cases", nargs="+", default=["60:0", "60:20", "60:30", "70:0", "70:20", "70:30"])
    parser.add_argument("--float-type", choices=["d", "ld"], default="d")
    parser.add_argument("--outdir", default="results/multiseed")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    try:
        cases = [tuple(map(int, case.split(":"))) for case in args.cases]
    except ValueError:
        parser.error("--cases entries must be n:blocksize")
    if any(len(case) != 2 for case in cases):
        parser.error("--cases entries must be n:blocksize")
    out = Path(args.outdir)
    paths = []
    for seed in args.seeds:
        aggregate = out / f"seed_{seed}_aggregate.csv"
        trials = out / f"seed_{seed}_trials.csv"
        print(f"Starting seed={seed}", flush=True)
        run(cases=cases, trials=args.trials, seed=seed, sigma=args.sigma,
            float_type=args.float_type, output=aggregate, trial_output=trials)
        paths.append(trials)
    summarize(paths, output_dir=out / "statistics")


if __name__ == "__main__":
    main()
