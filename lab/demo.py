"""Quick, reproducible educational demonstration: python -m lab.demo."""
import time
import numpy as np
from lab.lwe import Params, keygen, encrypt, decrypt
from lab.attacks import solve_mod_q, primal_attack, decoding_attack


def main():
    rng = np.random.default_rng(2026)
    params = Params(n=32, m=128, q=3329, sigma=2.0)
    pk, secret, _ = keygen(params, rng)
    failures = sum(decrypt(params, secret, encrypt(params, pk, i % 2, rng)) != i % 2 for i in range(1000))
    print(f"1. Encryption: {1000 - failures}/1000 successful round trips")

    clean = Params(n=32, m=64, q=3329, sigma=0, secret="uniform")
    (A0, b0), s0, _ = keygen(clean, rng)
    start = time.perf_counter()
    recovered = solve_mod_q(A0, b0, clean.q)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"2. Noiseless elimination: recovered={np.array_equal(recovered, s0)}, {elapsed:.1f} ms")
    noisy = Params(n=32, m=64, q=3329, sigma=2.0, secret="uniform")
    (A1, b1), _, _ = keygen(noisy, rng)
    print(f"   Noisy elimination inconsistent: {solve_mod_q(A1, b1, noisy.q) is None}")

    tiny = Params(n=3, m=6, q=3329, sigma=0.5)
    (A2, b2), s2, _ = keygen(tiny, rng)
    decoded = decoding_attack(A2, b2, tiny.q, max_abs_error=4)
    print(f"3. Own-LLL/Babai decoding (n=3): recovered={decoded is not None and np.array_equal(decoded % tiny.q, s2 % tiny.q)}")

    try:
        weak = Params(n=12, m=12, q=3329, sigma=3.0)
        (A3, b3), s3, _ = keygen(weak, rng)
        begin = time.perf_counter()
        candidate = primal_attack(A3, b3, weak.q, sigma=weak.sigma)
        found = candidate is not None and np.array_equal(candidate % weak.q, s3 % weak.q)
        print(f"4. Primal embedding (LLL): recovered={found}, {time.perf_counter() - begin:.2f} s")
    except ImportError:
        print("4. Primal embedding skipped: install fpylll and cysignals")
    print("Educational code only. Attack failure is NOT proof of security.")


if __name__ == "__main__":
    main()
