"""Elementary secret-recovery attacks on deliberately weak LWE instances."""
import math
import operator
import numpy as np


def _require_prime(q):
    """Validate a small educational modulus before attempting inverses."""
    try:
        q = operator.index(q)
    except TypeError as exc:
        raise ValueError("q must be a prime integer") from exc
    if isinstance(q, bool) or q < 2:
        raise ValueError("q must be a prime integer")
    if q % 2 == 0:
        if q != 2:
            raise ValueError("q must be prime (composite modulus)")
        return q
    for factor in range(3, math.isqrt(q) + 1, 2):
        if q % factor == 0:
            raise ValueError("q must be prime (composite modulus)")
    return q


def solve_mod_q(A, b, q):
    """Solve A s = b (mod q) by Gauss-Jordan, assuming prime q and full column rank.

    Returns None if there is no solution, or if the columns are rank deficient.
    Does not pretend that noisy equations can be solved by ordinary elimination.
    """
    q = _require_prime(q)
    rows = [[int(x) % q for x in a] for a in A]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("A must be a nonempty rectangular matrix")
    if len(b) != len(rows):
        raise ValueError("b must contain one entry per row")
    m, n = len(rows), len(rows[0])
    matrix = [rows[i] + [int(b[i]) % q] for i in range(m)]
    pivot_row = 0
    for col in range(n):
        pivot = next((r for r in range(pivot_row, m) if matrix[r][col]), None)
        if pivot is None:
            return None
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][col], -1, q)
        matrix[pivot_row] = [(x * inverse) % q for x in matrix[pivot_row]]
        for r in range(m):
            if r != pivot_row and matrix[r][col]:
                multiplier = matrix[r][col]
                matrix[r] = [(x - multiplier * y) % q for x, y in zip(matrix[r], matrix[pivot_row])]
        pivot_row += 1
    if any(matrix[r][-1] for r in range(n, m)):
        return None
    return np.array([matrix[i][-1] for i in range(n)], dtype=np.int64)


def embedding_basis(A, b, q):
    """Kannan's (m+n+1)-dimensional row-basis embedding for small secrets.

    Its lattice contains (e, -s, 1) when b = A s + e (mod q).
    fpylll is an optional dependency and imported only when this is called.
    """
    from fpylll import IntegerMatrix

    m, n = A.shape
    if len(b) != m:
        raise ValueError("b dimension mismatch")
    basis = IntegerMatrix(m + n + 1, m + n + 1)
    for i in range(m):
        basis[i, i] = int(q)
    for j in range(n):
        for i in range(m):
            basis[m + j, i] = int(A[i, j])
        basis[m + j, m + j] = 1
    for i in range(m):
        basis[m + n, i] = int(b[i])
    basis[m + n, m + n] = 1
    return basis


def _gaussian_norm_bound(dim, sigma):
    """Conservative Gaussian-like tail bound; heuristic for rounded samples.

    Uses a chi-square concentration bound with tail parameter log(1e6).
    Not a statistical proof for the rounded-Gaussian distribution used here.
    """
    tail = math.log(1_000_000)
    effective_sigma = math.hypot(sigma, 1 / math.sqrt(12))
    return effective_sigma * math.sqrt(dim + 2 * math.sqrt(dim * tail) + 2 * tail)


def _rank_primal_candidates(rows, A, b, q, max_error_norm, max_secret_norm):
    """Choose the shortest *plausible* embedding candidate across all rows.

    Equation consistency is necessary but vacuous for an embedding vector with
    final coordinate +/-1. Norm bounds are the actual filter. Even a candidate
    that passes these heuristic checks is NOT a cryptographic proof of recovery.
    """
    m, n = A.shape
    q = int(q)
    error_limit_sq = max_error_norm ** 2
    secret_limit_sq = max_secret_norm ** 2
    target = tuple(int(value) % q for value in b)
    candidates = []
    for row in rows:
        row = [int(x) for x in row]
        if len(row) != m + n + 1 or abs(row[-1]) != 1:
            continue
        sign = row[-1]
        error = [sign * x for x in row[:m]]
        secret = [-sign * x for x in row[m:m + n]]
        error_sq = sum(x * x for x in error)
        secret_sq = sum(x * x for x in secret)
        if error_sq > error_limit_sq or secret_sq > secret_limit_sq:
            continue
        # Necessary identity check only; the bounds and ranking do real work.
        if any((sum(int(a) * x for a, x in zip(a_row, secret)) + e) % q != v
               for a_row, e, v in zip(A, error, target)):
            continue
        score = error_sq + secret_sq
        candidates.append((score, error_sq, tuple(secret)))
    # Deduplicate identical candidate secrets. Sorting across ALL rows keeps
    # multiple plausible recoveries available to experimental evaluation.
    ranked = sorted(set(candidates), key=lambda item: (item[0], item[1], item[2]))
    seen = set()
    result = []
    for _, _, secret in ranked:
        reduced = tuple(value % q for value in secret)
        if reduced not in seen:
            seen.add(reduced)
            result.append(np.array(secret, dtype=np.int64))
    return result


def _select_primal_candidate(rows, A, b, q, max_error_norm, max_secret_norm):
    """Return the most plausible extracted candidate, or None."""
    candidates = _rank_primal_candidates(rows, A, b, q, max_error_norm, max_secret_norm)
    return candidates[0] if candidates else None


def primal_attack(A, b, q, block_size=None, max_loops=8, float_type="d", *,
                  sigma=None, max_error_norm=None, max_secret_norm=None,
                  return_candidates=False):
    """Attempt small-secret recovery using LLL/BKZ with heuristic norm filtering.

    Pass the known rounded-Gaussian `sigma` to derive conservative norms, or
    provide BOTH explicit maximum Euclidean norms for error and small secret.
    Without these, any embedding row ending in +/-1 could appear 'verified'.
    We scan *all* eligible rows and rank by combined norm. With
    return_candidates=True, return every plausible secret in ranked order so
    experiments can determine whether the planted secret appeared in ANY row.
    This remains a heuristic: only a test with a known secret can measure success.
    """
    A = np.asarray(A, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if A.ndim != 2 or A.size == 0 or b.shape != (A.shape[0],):
        raise ValueError("A must be nonempty 2D and b must have m entries")
    if sigma is None and (max_error_norm is None or max_secret_norm is None):
        raise ValueError("provide sigma or both maximum error and secret norms")
    if sigma is not None:
        if not isinstance(sigma, (int, float, np.integer, np.floating)) or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
    m, n = A.shape
    if max_error_norm is None:
        max_error_norm = 0.0 if sigma == 0 else _gaussian_norm_bound(m, sigma)
    if max_secret_norm is None:
        max_secret_norm = math.sqrt(n) if sigma == 0 else _gaussian_norm_bound(n, sigma)
    if any(not math.isfinite(bound) or bound < 0 for bound in (max_error_norm, max_secret_norm)):
        raise ValueError("norm bounds must be finite and nonnegative")
    from fpylll import LLL, BKZ

    basis = embedding_basis(A, b, q)
    LLL.reduction(basis)
    if block_size is not None:
        if block_size < 2:
            raise ValueError("BKZ block size must be >= 2")
        params = BKZ.Param(block_size=block_size, max_loops=max_loops,
                           flags=BKZ.AUTO_ABORT | BKZ.MAX_LOOPS)
        BKZ.reduction(basis, params, float_type=float_type)
    rows = ([basis[i, j] for j in range(basis.ncols)] for i in range(basis.nrows))
    candidates = _rank_primal_candidates(rows, A, b, q, max_error_norm, max_secret_norm)
    if return_candidates:
        return candidates
    return candidates[0] if candidates else None


def decoding_attack(A, b, q, max_abs_error):
    """Tiny-dimension q-ary decoding with our own LLL and Babai (no fpylll).

    Constructs row basis [I C^T; 0 qI] for L = {A x (mod q) + q z},
    where C = A2 A1^-1 modulo q, after selecting n independent rows.
    Requires m > n: m=n makes this lattice all of Z^n (no error evidence).
    Returns None if decoding fails or the inferred error exceeds its bound.
    """
    from lab.lattice import lll, babai_nearest_plane

    q = _require_prime(q)
    A = np.asarray(A, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if A.ndim != 2 or A.size == 0 or b.shape != (A.shape[0],):
        raise ValueError("invalid decoding matrix or target")
    m, n = A.shape
    if m <= n:
        raise ValueError("decoding requires m > n to constrain the error")
    if max_abs_error < 0:
        raise ValueError("max_abs_error must be nonnegative")

    # Pivot on ANY available row in each column. Track original row indices
    # so A and b can be permuted together. This also handles an initially
    # singular top n-by-n submatrix when the full matrix has rank n.
    working = [[int(x) % q for x in row] for row in A]
    original_rows = list(range(m))
    rank = 0
    for column in range(n):
        pivot = next((i for i in range(rank, m) if working[i][column]), None)
        if pivot is None:
            break
        working[rank], working[pivot] = working[pivot], working[rank]
        original_rows[rank], original_rows[pivot] = original_rows[pivot], original_rows[rank]
        inv_pivot = pow(working[rank][column], -1, q)
        for i in range(rank + 1, m):
            multiplier = working[i][column] * inv_pivot % q
            if multiplier:
                working[i] = [(x - multiplier * y) % q for x, y in zip(working[i], working[rank])]
        rank += 1
    selected = original_rows[:rank]
    if len(selected) != n:
        return None
    selection = selected + [i for i in range(m) if i not in selected]
    ordered_A, ordered_b = A[selection], b[selection]
    inverse_columns = []
    for j in range(n):
        column = solve_mod_q(ordered_A[:n], [int(i == j) for i in range(n)], q)
        if column is None:
            return None
        inverse_columns.append(column)
    inv = np.column_stack(inverse_columns)
    C = (ordered_A[n:] @ inv) % q
    basis = []
    for j in range(n):
        basis.append([int(i == j) for i in range(n)] + [int(C[k, j]) for k in range(m - n)])
    for k in range(m - n):
        basis.append([0] * n + [q * int(i == k) for i in range(m - n)])
    reduced = lll(basis)
    closest = np.array(babai_nearest_plane(reduced, ordered_b.tolist()), dtype=np.int64)
    ordered_error = ordered_b - closest
    error = np.empty(m, dtype=np.int64)
    error[selection] = ordered_error
    if np.max(np.abs(error)) > max_abs_error:
        return None
    secret = solve_mod_q(A, (b - error) % q, q)
    if secret is None or not np.array_equal((A @ secret + error) % q, b % q):
        return None
    return secret
