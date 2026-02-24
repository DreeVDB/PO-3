import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def sample_three_equal(rng: np.random.Generator, options=None):
    """Return één van drie opties met gelijke kans.

    Args:
        rng: een numpy random Generator.
        options: optioneel een iterabele van lengte 3 met de opties. Als None wordt 0,1,2 geretourneerd.

    Returns:
        Eén van de drie opties (of een index 0/1/2 als `options` None).
    """
    if options is None:
        return int(rng.integers(0, 3))
    opts = list(options)
    if len(opts) != 3:
        raise ValueError("options must have length 3")
    return opts[int(rng.integers(0, 3))]


def make_quintic_all_real(rng: np.random.Generator, low=-2.0, high=2.0):
    """Bouw een 5e-graadspolynoom met vijf reële wortels.

    Keert terug: (coeffs, roots)
    """
    roots = rng.uniform(low, high, size=5).astype(np.float64)
    coeffs = np.poly(roots)
    coeffs = np.real_if_close(coeffs, tol=1000).astype(np.float64)

    scale = rng.uniform(0.5, 2.0) * (1.0 if rng.random() < 0.5 else -1.0)
    coeffs *= scale

    
    return coeffs, roots


def make_quintic_three_real_one_pair(rng: np.random.Generator, low=-2.0, high=2.0):
    """Bouw een 5e-graadspolynoom met 3 reële wortels en 1 complex geconjugeerd paar.

    Keert terug: (coeffs, roots)
    """
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

    return coeffs, roots


def make_quintic_with_one_real_root(rng: np.random.Generator):
    """
    Bouw een 5de-graadspolynoom met exact 1 reële wortel r
    en 2 complexe geconjugeerde paren.
    Return:
      coeffs: (6,) a5..a0 (float64)
      real_root: float
      
    """

    r = rng.uniform(-2.0, 2.0)

    u = rng.uniform(-2.0, 2.0)
    v = rng.uniform(0.2, 2.0)
    w = rng.uniform(-2.0, 2.0)
    z = rng.uniform(0.2, 2.0)

    roots = np.array(
        [r, u + 1j * v, u - 1j * v, w + 1j * z, w - 1j * z],
        dtype=np.complex128,
    )

    coeffs = np.poly(roots)  # monisch a5=1, a5..a0 (coeff leidende term = 1)
    coeffs = np.real_if_close(coeffs, tol=1000).astype(np.float64)

    # schaal zodat a5 niet altijd 1 is
    scale = rng.uniform(0.5, 2.0) * (1.0 if rng.random() < 0.5 else -1.0)
    coeffs *= scale

    return coeffs, roots


def build_dataset(n_samples: int):
    rng = np.random.default_rng()
    X = np.zeros((n_samples, 6), dtype=np.float64)
    # y bevat reële representatie van alle 5 wortels: eerst reële delen, daarna imaginaire delen
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
        # zet eerst de reële delen, daarna de imaginaire delen (shape 10)
        reals = np.real(roots).astype(np.float64)
        imags = np.imag(roots).astype(np.float64)
        y[i, :5] = reals
        y[i, 5:] = imags

    return X, y

### testfunctie
if __name__ == "__main__":
    # korte test van sample_three_equal: telt frequenties voor visuele controle
    rng = np.random.default_rng(42)
    opts = ["A", "B", "C"]
    counts = {o: 0 for o in opts}
    for _ in range(3000):
        counts[sample_three_equal(rng, opts)] += 1
    print("Sample counts (should be ~equal):", counts)

    # korte sanity-check van build_dataset
    X, y = build_dataset(8)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print("voorbeeld y[0]:", y[0])