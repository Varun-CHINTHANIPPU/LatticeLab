import numpy as np
import pytest
from lab.lwe import Params, keygen
from lab.attacks import solve_mod_q, decoding_attack, primal_attack


def test_linear_solver_recovers_noiseless_uniform_secret():
    p = Params(16, 32, 3329, 0, "uniform")
    (A, b), s, _ = keygen(p, np.random.default_rng(42))
    recovered = solve_mod_q(A, b, p.q)
    assert recovered is not None
    assert np.array_equal(recovered, s)


def test_noisy_system_inconsistent():
    p = Params(16, 32, 3329, 2, "uniform")
    (A, b), _, errors = keygen(p, np.random.default_rng(43))
    assert np.any(errors)
    assert solve_mod_q(A, b, p.q) is None


def test_solver_rejects_rank_deficiency():
    assert solve_mod_q([[1, 0], [2, 0]], [1, 2], 3329) is None


def test_own_decoding_tiny_instance():
    p = Params(3, 6, 3329, 0)
    (A, b), s, _ = keygen(p, np.random.default_rng(13))
    recovered = decoding_attack(A, b, p.q, max_abs_error=0)
    assert recovered is not None and np.array_equal(recovered % p.q, s % p.q)


def test_primal_lll_weak_instance():
    pytest.importorskip("fpylll")
    p = Params(8, 8, 3329, 2)
    (A, b), s, _ = keygen(p, np.random.default_rng(2))
    recovered = primal_attack(A, b, p.q, sigma=p.sigma)
    assert recovered is not None and np.array_equal(recovered % p.q, s % p.q)


def test_primal_candidate_scan_uses_norms_and_best_row():
    from lab.attacks import _select_primal_candidate
    A = np.diag([3, 3]).astype(np.int64)
    b = np.array([3, 3], dtype=np.int64)
    # Both rows satisfy the embedding identity. The first is a worse guess.
    wrong_first = [3, 3, 0, 0, 1]
    true_later = [0, 0, -1, -1, 1]
    recovered = _select_primal_candidate([wrong_first, true_later], A, b, 101, 10, 10)
    assert recovered is not None and np.array_equal(recovered, [1, 1])
    # The same modular identity is not sufficient without plausible norms.
    assert _select_primal_candidate([wrong_first], A, b, 101, 2, 10) is None


def test_primal_requires_sigma_or_two_explicit_norms():
    A = np.eye(2, dtype=np.int64)
    b = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="sigma or both"):
        primal_attack(A, b, 3329)


def test_public_b_candidates_are_not_equivalent_to_verified_secrets():
    from lab.attacks import _rank_primal_candidates
    A = np.diag([3, 3]).astype(np.int64)
    b = np.array([3, 3], dtype=np.int64)
    guesses = _rank_primal_candidates([[3, 3, 0, 0, 1], [0, 0, -1, -1, 1]],
                                     A, b, 101, 10, 10)
    assert len(guesses) == 2
    assert np.array_equal(guesses[0], [1, 1])
    # The second candidate is wrong but still equation-consistent and plausible:
    # only planted ground truth distinguishes them in this toy instance.
    assert np.array_equal(guesses[1], [0, 0])


def test_primal_lll_weak_instance_with_noise_bound():
    pytest.importorskip("fpylll")
    p = Params(8, 8, 3329, 2)
    (A, b), secret, _ = keygen(p, np.random.default_rng(2))
    found = primal_attack(A, b, p.q, sigma=p.sigma)
    assert found is not None and np.array_equal(found % p.q, secret % p.q)


def test_decoding_rejects_square_case():
    A = np.array([[2, 0], [0, 5]], dtype=np.int64)
    with pytest.raises(ValueError, match="m > n"):
        decoding_attack(A, [2, 5], 3329, max_abs_error=1)


def test_decoding_uses_independent_rows_not_only_first_n():
    A = np.array([[1, 0], [2, 0], [0, 1], [1, 1]], dtype=np.int64)
    secret = np.array([-1, 2], dtype=np.int64)
    b = A @ secret
    found = decoding_attack(A, b, 3329, max_abs_error=0)
    assert found is not None and np.array_equal(found % 3329, secret % 3329)


def test_decoding_rejects_genuinely_rank_deficient_matrix():
    A = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.int64)
    assert decoding_attack(A, [1, 2, 3], 3329, max_abs_error=0) is None


@pytest.mark.parametrize("modulus", [1, 4, 9, 15, 341])
def test_modular_solver_rejects_composite_moduli(modulus):
    with pytest.raises(ValueError, match="prime"):
        solve_mod_q([[1]], [1], modulus)
