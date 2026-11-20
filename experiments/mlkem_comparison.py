"""FIPS 203 ML-KEM parameter reference and toy-LWE structural comparison.

This module does NOT implement ML-KEM, estimate ML-KEM security, or claim
equivalence between ordinary LWE and Module-LWE. No expensive attacks run.
Source: NIST FIPS 203 (2024), Section 8, Tables 2 and 3.
https://doi.org/10.6028/NIST.FIPS.203
"""
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

FIPS203 = "https://doi.org/10.6028/NIST.FIPS.203"


@dataclass(frozen=True)
class MLKEM:
    name: str
    category: int
    n: int
    q: int
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int
    rbg_bits: int
    ek_bytes: int
    dk_bytes: int
    ct_bytes: int
    shared_secret_bytes: int = 32

    @property
    def coefficient_count(self):
        """Number of coefficients in a rank-k module vector, NOT LWE equivalence."""
        return self.n * self.k


PARAMETERS = (
    MLKEM("ML-KEM-512", 1, 256, 3329, 2, 3, 2, 10, 4, 128, 800, 1632, 768),
    MLKEM("ML-KEM-768", 3, 256, 3329, 3, 2, 2, 10, 4, 192, 1184, 2400, 1088),
    MLKEM("ML-KEM-1024", 5, 256, 3329, 4, 2, 2, 11, 5, 256, 1568, 3168, 1568),
)


def negacyclic_multiply(a, b, q=3329):
    """Exact coefficient convolution in Z_q[X]/(X^n+1), for small test vectors."""
    if len(a) != len(b) or not a or not isinstance(q, int) or q < 2:
        raise ValueError("equal nonempty polynomial lengths and q >= 2 required")
    n = len(a)
    out = [0] * n
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            degree = i + j
            if degree >= n:
                out[degree - n] -= x * y
            else:
                out[degree] += x * y
    return [value % q for value in out]


def compare(toy_n=40, toy_m=40, toy_q=3329, toy_sigma=3.0):
    if (not isinstance(toy_n, int) or toy_n < 1 or
        not isinstance(toy_m, int) or toy_m < 1 or
        not isinstance(toy_q, int) or toy_q < 2 or
        not 0 < toy_sigma < float("inf")):
        raise ValueError("invalid toy LWE parameters")
    toy = {"problem": "ordinary_LWE", "n": toy_n, "m": toy_m, "q": toy_q,
           "noise": "rounded continuous Gaussian (NumPy), sigma specified",
           "sigma": toy_sigma, "secret": "small integer coefficients",
           "structure": "unstructured matrix over Z_q",
           "scheme": "educational toy; not a KEM"}
    rows = []
    for p in PARAMETERS:
        record = asdict(p)
        record["coefficient_count"] = p.coefficient_count
        record["problem"] = "Module_LWE"
        record["ring"] = "Z_q[X]/(X^256+1)"
        record["noise"] = "centered binomial distribution; eta1/eta2 as specified"
        record["comparison"] = "structural parameters only; no equivalent LWE security inferred"
        rows.append(record)
    return {"source": FIPS203, "toy": toy, "ml_kem": rows,
            "limitations": [
                "NIST security categories are NOT independently estimated bit-security values.",
                "Module-LWE ring structure and centered-binomial noise differ from toy ordinary LWE.",
                "k*256 counts coefficients; it is not a security-equivalent ordinary LWE dimension.",
                "Toy estimator work factors and measured attack times cannot be compared numerically to ML-KEM categories.",
                "No ML-KEM key generation, encapsulation, decapsulation, or cryptanalysis is performed.",
            ]}


def write_reports(data, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "comparison.json"
    csv_path = out / "mlkem_parameters.csv"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data["ml_kem"][0]))
        writer.writeheader()
        writer.writerows(data["ml_kem"])
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-n", type=int, default=40)
    parser.add_argument("--toy-m", type=int, default=40)
    parser.add_argument("--toy-q", type=int, default=3329)
    parser.add_argument("--toy-sigma", type=float, default=3.0)
    parser.add_argument("--outdir", default="results/mlkem_comparison")
    args = parser.parse_args()
    try:
        data = compare(args.toy_n, args.toy_m, args.toy_q, args.toy_sigma)
        paths = write_reports(data, args.outdir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for row in data["ml_kem"]:
        print(f"{row['name']}: category={row['category']}, k={row['k']}, "
              f"coefficients={row['coefficient_count']}, ek={row['ek_bytes']}B, "
              f"ct={row['ct_bytes']}B")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
