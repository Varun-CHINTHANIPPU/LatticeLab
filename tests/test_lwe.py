import numpy as np
import pytest
from lab.lwe import Params, keygen, encrypt, decrypt


def test_10000_bits_round_trip():
    rng = np.random.default_rng(2026)
    p = Params(n=32, m=128, q=3329, sigma=2.0)
    pk, secret, _ = keygen(p, rng)
    for bit in range(10000):
        assert decrypt(p, secret, encrypt(p, pk, bit % 2, rng)) == bit % 2


def test_ciphertext_bounds_and_fresh_randomness():
    rng = np.random.default_rng(42)
    p = Params(n=12, m=48, q=3329, sigma=3.0)
    pk, _, _ = keygen(p, rng)
    cts = [encrypt(p, pk, 1, rng) for _ in range(5)]
    assert len({tuple(c[0]) + (c[1],) for c in cts}) == 5
    for c1, c2 in cts:
        assert all(0 <= x < p.q for x in c1)
        assert 0 <= c2 < p.q


def test_reproducible_seed():
    p = Params(8, 16, 3329, 2)
    a = keygen(p, np.random.default_rng(123))
    b = keygen(p, np.random.default_rng(123))
    assert all(np.array_equal(x, y) for x, y in zip((*a[0], *a[1:]), (*b[0], *b[1:])))


def test_bad_params_or_bit_rejected():
    with pytest.raises(ValueError):
        Params(8, 2, 3329, 2)
    p = Params(4, 8, 3329, 2)
    pk, _, _ = keygen(p, np.random.default_rng(1))
    with pytest.raises(ValueError):
        encrypt(p, pk, 2, np.random.default_rng(2))


@pytest.mark.parametrize("kwarg", [dict(n=2.5), dict(m=8.1), dict(q=3329.0), dict(n=True)])
def test_params_reject_non_integer_dimensions_and_modulus(kwarg):
    args = dict(n=4, m=8, q=3329, sigma=2.0)
    args.update(kwarg)
    with pytest.raises(TypeError):
        Params(**args)


@pytest.mark.parametrize("sigma", [float('nan'), float('inf'), '2', True])
def test_params_reject_invalid_sigma_type(sigma):
    with pytest.raises(TypeError, match="sigma"):
        Params(4, 8, 3329, sigma)
