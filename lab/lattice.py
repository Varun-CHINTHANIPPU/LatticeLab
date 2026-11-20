"""Exact, educational row-basis Gram-Schmidt, LLL, and Babai algorithms.

We deliberately use Fraction and recompute Gram-Schmidt after each LLL step.
This is easy to audit but only practical for small dimensions (< 20).
"""
from fractions import Fraction


def dot(u, v):
    if len(u) != len(v):
        raise ValueError("vectors have different dimensions")
    return sum((a * b for a, b in zip(u, v)), 0)


def _validate_basis(basis):
    if not basis or not basis[0]:
        raise ValueError("basis cannot be empty")
    width = len(basis[0])
    if len(basis) > width or any(len(row) != width for row in basis):
        raise ValueError("basis rows must have the same width; rank cannot exceed dimension")
    if any(any(int(x) != x for x in row) for row in basis):
        raise ValueError("basis entries must be integers")


def gram_schmidt(basis):
    """Return (orthogonal_rows, mu), using exact rational arithmetic."""
    _validate_basis(basis)
    size = len(basis)
    orthogonal = []
    mu = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for i, row in enumerate(basis):
        v = [Fraction(x) for x in row]
        for j in range(i):
            denominator = dot(orthogonal[j], orthogonal[j])
            if not denominator:
                raise ValueError("linearly dependent basis")
            mu[i][j] = dot(row, orthogonal[j]) / denominator
            v = [a - mu[i][j] * b for a, b in zip(v, orthogonal[j])]
        if dot(v, v) == 0:
            raise ValueError("linearly dependent basis")
        orthogonal.append(v)
    return orthogonal, mu


def _validated_delta(delta):
    delta = Fraction(delta)
    if not Fraction(1, 4) < delta < 1:
        raise ValueError("delta must be strictly between 1/4 and 1")
    return delta


def lll(basis, delta=Fraction(3, 4)):
    """Return an LLL-reduced integer row basis; 1/4 < delta < 1."""
    delta = _validated_delta(delta)
    _validate_basis(basis)  # reject fractional entries BEFORE int() conversion
    rows = [[int(x) for x in row] for row in basis]
    orthogonal, mu = gram_schmidt(rows)
    k = 1
    while k < len(rows):
        for j in range(k - 1, -1, -1):
            coefficient = round(mu[k][j])
            if coefficient:
                rows[k] = [x - coefficient * y for x, y in zip(rows[k], rows[j])]
                orthogonal, mu = gram_schmidt(rows)
        lhs = dot(orthogonal[k], orthogonal[k])
        rhs = (delta - mu[k][k - 1] ** 2) * dot(orthogonal[k - 1], orthogonal[k - 1])
        if lhs >= rhs:
            k += 1
        else:
            rows[k], rows[k - 1] = rows[k - 1], rows[k]
            orthogonal, mu = gram_schmidt(rows)
            k = max(1, k - 1)
    return rows


def is_lll_reduced(basis, delta=Fraction(3, 4)):
    delta = _validated_delta(delta)
    orthogonal, mu = gram_schmidt(basis)
    return (
        all(abs(mu[i][j]) <= Fraction(1, 2) for i in range(len(basis)) for j in range(i))
        and all(
            dot(orthogonal[k], orthogonal[k])
            >= (delta - mu[k][k - 1] ** 2) * dot(orthogonal[k - 1], orthogonal[k - 1])
            for k in range(1, len(basis))
        )
    )


def babai_nearest_plane(basis, target):
    """Approximate closest lattice vector to target using a full-rank row basis."""
    orthogonal, _ = gram_schmidt(basis)
    if len(target) != len(basis[0]):
        raise ValueError("target dimension does not match basis")
    residual = [Fraction(x) for x in target]
    for i in range(len(basis) - 1, -1, -1):
        coefficient = round(dot(residual, orthogonal[i]) / dot(orthogonal[i], orthogonal[i]))
        residual = [x - coefficient * y for x, y in zip(residual, basis[i])]
    point = [Fraction(x) - r for x, r in zip(target, residual)]
    if any(x.denominator != 1 for x in point):
        raise ValueError("target or basis was not integral")
    return [int(x) for x in point]
