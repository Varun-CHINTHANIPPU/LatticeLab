"""Independent SageMath lattice sanity checks (no fpylll dependency).

Run: sage integrations/sage_crosscheck.sage --output results/sage_crosscheck.json
"""
import argparse
import json
from pathlib import Path
from sage.all import ZZ, matrix

def verify():
    basis = matrix(ZZ, [[4, 1, 3], [2, 1, 1], [1, 0, 2]])
    reduced = basis.LLL()
    original_det = abs(basis.det())
    reduced_det = abs(reduced.det())
    if original_det == 0 or original_det != reduced_det:
        raise AssertionError("LLL did not preserve lattice covolume")
    # For full-rank integral square bases, det(U)=±1 proves equality of lattices.
    transform = reduced * basis.inverse()
    if any(x.denominator() != 1 for x in transform.list()) or abs(transform.det()) != 1:
        raise AssertionError("Sage LLL output not unimodularly equivalent")
    return {"check": "sage_lll_unimodular_equivalence",
            "passed": True, "dimension": basis.nrows(),
            "absolute_determinant": int(original_det),
            "basis": [[int(x) for x in row] for row in basis.rows()],
            "reduced_basis": [[int(x) for x in row] for row in reduced.rows()]}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="results/sage_crosscheck.json")
args = parser.parse_args()
result = verify()
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
