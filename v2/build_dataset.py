import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

    return coeffs, float(r)


def build_dataset(n_samples: int):
    rng = np.random.default_rng()
    X = np.zeros((n_samples, 6), dtype=np.float64)
    y = np.zeros((n_samples, 1), dtype=np.float64)

    for i in range(n_samples):
        coeffs, r = make_quintic_with_one_real_root(rng)
        X[i] = coeffs
        y[i, 0] = r

    return X, y