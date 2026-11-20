"""Textbook Regev-style single-bit LWE public-key encryption; educational only.

NOT constant-time, authenticated, CCA-secure, or suitable for real data.
All experiments use small enough q and dimensions to avoid NumPy int64 overflow.
"""
from dataclasses import dataclass
from numbers import Real
import math
import numpy as np


@dataclass(frozen=True)
class Params:
    n: int
    m: int
    q: int
    sigma: float
    secret: str = "small"

    def __post_init__(self):
        for name in ("n", "m", "q"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if isinstance(self.sigma, (bool, np.bool_)) or not isinstance(self.sigma, Real) or not math.isfinite(self.sigma):
            raise TypeError("sigma must be a finite real number")
        if self.n < 1 or self.m < self.n or self.q < 3 or self.q % 2 != 1:
            raise ValueError("require n >= 1, m >= n, odd q >= 3")
        if not 0 <= self.sigma < self.q:
            raise ValueError("sigma must be nonnegative and smaller than q")
        if self.secret not in {"small", "uniform"}:
            raise ValueError("secret must be 'small' or 'uniform'")
        if self.q > 100000 or max(self.m, self.n) > 10000:
            raise ValueError("parameters exceed this educational int64 implementation's limits")


def sample_error(rng, sigma, size):
    return np.rint(rng.normal(0.0, sigma, size=size)).astype(np.int64)


def keygen(params: Params, rng):
    A = rng.integers(0, params.q, size=(params.m, params.n), dtype=np.int64)
    if params.secret == "uniform":
        secret = rng.integers(0, params.q, size=params.n, dtype=np.int64)
    elif params.sigma == 0:
        secret = rng.integers(-1, 2, size=params.n, dtype=np.int64)
    else:
        secret = sample_error(rng, params.sigma, params.n)
    error = sample_error(rng, params.sigma, params.m)
    public_b = (A @ secret + error) % params.q
    # Returning error is for experimental diagnostics, never for a production API.
    return (A, public_b), secret, error


def encrypt(params: Params, public_key, bit: int, rng):
    if bit not in (0, 1):
        raise ValueError("plaintext must be a bit (0 or 1)")
    A, b = public_key
    subset = rng.integers(0, 2, size=params.m, dtype=np.int64)
    return (subset @ A) % params.q, (int(subset @ b) + bit * (params.q // 2)) % params.q


def decrypt(params: Params, secret, ciphertext):
    c1, c2 = ciphertext
    phase = (int(c2) - int(c1 @ secret)) % params.q
    distance_to_zero = min(phase, params.q - phase)
    distance_to_one = abs(phase - params.q // 2)
    return int(distance_to_one < distance_to_zero)
