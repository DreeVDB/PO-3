import numpy as np

from root_utils import encode_roots_ri, sort_roots_canonical


def sample_three_equal(rng: np.random.Generator, options=None):
    """Return one of three options with equal probability."""
    if options is None:
        return int(rng.integers(0, 3))
    opts = list(options)
    if len(opts) != 3:
        raise ValueError("options must have length 3")
    return opts[int(rng.integers(0, 3))]


def make_quintic_all_real(rng: np.random.Generator, low=-2.0, high=2.0):
    """Build a 5th-degree polynomial with five real roots."""
    roots = rng.uniform(low, high, size=5).astype(np.float64)
    coeffs = np.poly(roots)
    coeffs = np.real_if_close(coeffs, tol=1000).astype(np.float64)

    scale = rng.uniform(0.5, 2.0) * (1.0 if rng.random() < 0.5 else -1.0)
    coeffs *= scale

    return coeffs, sort_roots_canonical(roots)


def make_quintic_three_real_one_pair(rng: np.random.Generator, low=-2.0, high=2.0):
    """Build a 5th-degree polynomial with three real roots and one complex pair."""
    reals = rng.uniform(low, high, size=3)
    u = rng.uniform(low, high)
    v = rng.uniform(0.2, 2.0)

    roots = np.array(
        [reals[0], reals[1], reals[2], u + 1j * v, u - 1j * v],
        dtype=np.complex128,
    )

    coeffs = np.poly(roots)
    coeffs = np.real_if_close(coeffs, tol=1000).astype(np.float64)

    scale = rng.uniform(0.5, 2.0) * (1.0 if rng.random() < 0.5 else -1.0)
    coeffs *= scale

    return coeffs, sort_roots_canonical(roots)


def make_quintic_with_one_real_root(rng: np.random.Generator):
    """Build a 5th-degree polynomial with one real root and two complex pairs."""
    r = rng.uniform(-2.0, 2.0)

    u = rng.uniform(-2.0, 2.0)
    v = rng.uniform(0.2, 2.0)
    w = rng.uniform(-2.0, 2.0)
    z = rng.uniform(0.2, 2.0)

    roots = np.array(
        [r, u + 1j * v, u - 1j * v, w + 1j * z, w - 1j * z],
        dtype=np.complex128,
    )

    coeffs = np.poly(roots)
    coeffs = np.real_if_close(coeffs, tol=1000).astype(np.float64)

    scale = rng.uniform(0.5, 2.0) * (1.0 if rng.random() < 0.5 else -1.0)
    coeffs *= scale

    return coeffs, sort_roots_canonical(roots)


def build_dataset(n_samples: int):
    rng = np.random.default_rng()
    X = np.zeros((n_samples, 6), dtype=np.float64)
    # y stores all 5 roots as [re0..re4, im0..im4]
    y = np.zeros((n_samples, 10), dtype=np.float64)

    for i in range(n_samples):
        choice = sample_three_equal(rng)
        if choice == 0:
            coeffs, roots = make_quintic_all_real(rng)
        elif choice == 1:
            coeffs, roots = make_quintic_with_one_real_root(rng)
        else:
            coeffs, roots = make_quintic_three_real_one_pair(rng)

        X[i] = coeffs
        y[i] = encode_roots_ri(roots)

    return X, y


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    opts = ["A", "B", "C"]
    counts = {o: 0 for o in opts}
    for _ in range(3000):
        counts[sample_three_equal(rng, opts)] += 1
    print("Sample counts (should be ~equal):", counts)

    X, y = build_dataset(8)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print("voorbeeld y[0]:", y[0])
