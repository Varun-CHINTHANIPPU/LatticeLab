"""Independent exact arithmetic checks for educational lattice routines."""
from fractions import Fraction
import random
import pytest
from lab.lattice import dot, gram_schmidt, lll, is_lll_reduced, babai_nearest_plane


def determinant(rows):
    """Exact elimination for lattice-volume invariance (test implementation)."""
    a = [[Fraction(x) for x in r] for r in rows]
    n = len(a)
    out = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col]), None)
        if pivot is None:
            return 0
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            out = -out
        value = a[col][col]
        out *= value
        for r in range(col + 1, n):
            factor = a[r][col] / value
            for j in range(col, n):
                a[r][j] -= factor * a[col][j]
    return out


def test_gs_orthogonality_and_reconstruction():
    basis = [[4, 1, 3], [2, 1, 0], [1, 2, 3]]
    orthogonal, mu = gram_schmidt(basis)
    for i in range(3):
        reconstructed = [orthogonal[i][j] + sum(mu[i][k] * orthogonal[k][j] for k in range(i)) for j in range(3)]
        assert reconstructed == basis[i]
        for j in range(i):
            assert dot(orthogonal[i], orthogonal[j]) == 0


def test_lll_random_bases_volume_and_reduced():
    generator = random.Random(2026)
    for n in (2, 3, 4, 5):
        for _ in range(12):
            while True:
                basis = [[generator.randrange(-12, 13) for _ in range(n)] for _ in range(n)]
                if determinant(basis):
                    break
            reduced = lll(basis)
            assert is_lll_reduced(reduced)
            assert abs(determinant(basis)) == abs(determinant(reduced))


def test_babai_planted_point():
    basis = [[25, 0, 0], [0, 25, 0], [0, 0, 25]]
    target = [26, 49, -23]
    assert babai_nearest_plane(lll(basis), target) == [25, 50, -25]


def test_degenerate_rejected():
    with pytest.raises(ValueError, match="dependent"):
        gram_schmidt([[2, 0], [4, 0]])
    with pytest.raises(ValueError, match="delta"):
        lll([[1, 0], [0, 1]], delta=1)


def test_fpylll_comparison():
    fpylll = pytest.importorskip("fpylll")
    generator = random.Random(124)
    for n in (3, 5, 8):
        for _ in range(10):
            while True:
                basis = [[generator.randrange(-40, 41) for _ in range(n)] for _ in range(n)]
                if determinant(basis):
                    break
            own = lll(basis)
            matrix = fpylll.IntegerMatrix.from_matrix(basis)
            fpylll.LLL.reduction(matrix, delta=0.75)
            other = [[int(matrix[i, j]) for j in range(n)] for i in range(n)]
            # LLL is not canonical: different valid reductions need not have
            # identical first-vector lengths, even with the same delta.
            assert is_lll_reduced(own)
            # fpylll's default eta is 0.51 rather than our exact 0.50,
            # so testing its output with our stricter checker is invalid.
            other_star, other_mu = gram_schmidt(other)
            assert all(abs(other_mu[i][j]) <= Fraction(51, 100)
                       for i in range(n) for j in range(i))
            assert all(
                dot(other_star[k], other_star[k]) >=
                (Fraction(3, 4) - other_mu[k][k - 1] ** 2) *
                dot(other_star[k - 1], other_star[k - 1])
                for k in range(1, n)
            )
            assert abs(determinant(own)) == abs(determinant(other)) == abs(determinant(basis))


def test_lll_rejects_non_integral_basis_before_conversion():
    non_integral = [[1.9, 0], [0, 1.9]]
    with pytest.raises(ValueError, match="integers"):
        lll(non_integral)
    with pytest.raises(ValueError, match="integers"):
        gram_schmidt(non_integral)


def test_reduced_checker_rejects_invalid_delta():
    for delta in (0.25, 1.0, -1.0):
        with pytest.raises(ValueError, match="delta"):
            is_lll_reduced([[1, 0], [0, 1]], delta=delta)
