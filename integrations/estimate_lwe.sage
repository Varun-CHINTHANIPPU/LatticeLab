"""SageMath LWE estimator integration: explicit toy parameters only.

Run: sage integrations/estimate_lwe.sage --estimator-path ~/cryptography-tools/lattice-estimator
"""
import argparse
import json
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--estimator-path", required=True, help="cloned malb/lattice-estimator directory")
parser.add_argument("--n", type=int, default=40)
parser.add_argument("--m", type=int, default=40)
parser.add_argument("--q", type=int, default=3329)
parser.add_argument("--sigma", type=float, default=3.0)
parser.add_argument("--secret-bound", type=int, default=1)
parser.add_argument("--output", default="results/estimator_toy.json")
args = parser.parse_args()
if args.n < 2 or args.m < 1 or args.q < 3 or args.secret_bound < 0 or not math.isfinite(args.sigma) or args.sigma <= 0:
    parser.error("require n>=2, m>=1, q>=3, sigma>0 finite, secret-bound>=0")
location = Path(args.estimator_path).expanduser().resolve()
if not (location / "estimator").is_dir():
    parser.error(f"estimator package not found under {location}")
sys.path.insert(0, str(location))
try:
    from estimator import LWE, ND
except ImportError as exc:
    parser.error(f"cannot import Lattice Estimator: {exc}")
params = LWE.Parameters(n=args.n, q=args.q, m=args.m,
                        Xs=ND.Uniform(-args.secret_bound, args.secret_bound),
                        Xe=ND.DiscreteGaussian(args.sigma), tag="LatticeLab-toy")
estimates = LWE.estimate.rough(params)
results = {}
for attack, cost in estimates.items():
    # Cost objects provide a mapping; rop is a heuristic work factor.
    rop = float(cost["rop"])
    if not math.isfinite(rop) or rop <= 0:
        raise ValueError(f"invalid rop for {attack}")
    results[str(attack)] = {"log2_rop": math.log2(rop)}
output = {"kind": "heuristic_toy_LWE_cost_estimate",
          "parameters": {"n": args.n, "m": args.m, "q": args.q,
                         "sigma": args.sigma, "secret_bound": args.secret_bound},
          "estimator_path": str(location), "attacks": results,
          "limitations": "Heuristic estimator model; not measured runtime, not ML-KEM security."}
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2, sort_keys=True))
