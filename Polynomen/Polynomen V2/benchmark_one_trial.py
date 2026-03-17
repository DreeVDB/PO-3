import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from newton_raphson import newton_raphson_complex, poly_and_derivative
from root_utils import decode_roots_ri, sort_roots_canonical


def make_monic(coeffs: np.ndarray) -> np.ndarray:
    # monisch = coefficient at highest degree is 1
    a5 = coeffs[..., 0:1]
    return coeffs / a5


class Standardizer:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X: np.ndarray):
        self.mu = X.mean(axis=0, keepdims=True)
        self.sigma = X.std(axis=0, keepdims=True) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma


def _success_from_roots(roots_found: np.ndarray, roots_true: np.ndarray, atol=1e-5) -> bool:
    roots_found = sort_roots_canonical(roots_found)
    roots_true = sort_roots_canonical(roots_true)
    return bool(np.all(np.abs(roots_found - roots_true) < atol))


def benchmark_one_trial(model, scaler: Standardizer, X_test_raw, y_test, rand_range=(-3.0, 3.0)):
    rng = np.random.default_rng()

    X_nn = scaler.transform(make_monic(X_test_raw)).astype(np.float32)
    y_true_roots = np.array([sort_roots_canonical(decode_roots_ri(v)) for v in y_test])

    t0 = time.perf_counter()
    x0_nn_vec = model(X_nn, training=False).numpy()
    t_pred = time.perf_counter() - t0

    # Newton-Raphson met NN-start
    t0 = time.perf_counter()
    ok_nn = 0
    it_nn = []
    err_nn = []
    res_nn = []

    for coeffs, true_roots, x0_vec in zip(X_test_raw, y_true_roots, x0_nn_vec):
        x0_roots = decode_roots_ri(x0_vec)
        found_roots = []
        poly_iters = []
        poly_ok = True

        for x0 in x0_roots:
            x_hat, ok, it = newton_raphson_complex(coeffs, x0)
            found_roots.append(x_hat)
            poly_iters.append(it)
            poly_ok = poly_ok and ok

        found_roots = sort_roots_canonical(np.array(found_roots, dtype=np.complex128))
        poly_success = poly_ok and _success_from_roots(found_roots, true_roots)

        ok_nn += int(poly_success)
        it_nn.append(float(np.mean(poly_iters)))

        if poly_success:
            err_nn.extend(np.abs(found_roots - true_roots).tolist())
            for root_hat in found_roots:
                fx, _ = poly_and_derivative(coeffs, root_hat)
                res_nn.append(abs(fx))

    t_nn = time.perf_counter() - t0

    # Random start (1 trial per polynoom)
    t0 = time.perf_counter()
    ok_r = 0
    it_r = []
    err_r = []
    res_r = []

    lo, hi = rand_range
    for coeffs, true_roots in zip(X_test_raw, y_true_roots):
        found_roots = []
        poly_iters = []
        poly_ok = True

        for _ in range(5):
            x0 = rng.uniform(lo, hi) + 1j * rng.uniform(lo, hi)
            x_hat, ok, it = newton_raphson_complex(coeffs, x0)
            found_roots.append(x_hat)
            poly_iters.append(it)
            poly_ok = poly_ok and ok

        found_roots = sort_roots_canonical(np.array(found_roots, dtype=np.complex128))
        poly_success = poly_ok and _success_from_roots(found_roots, true_roots)

        ok_r += int(poly_success)
        it_r.append(float(np.mean(poly_iters)))

        if poly_success:
            err_r.extend(np.abs(found_roots - true_roots).tolist())
            for root_hat in found_roots:
                fx, _ = poly_and_derivative(coeffs, root_hat)
                res_r.append(abs(fx))

    t_r = time.perf_counter() - t0

    # Rapport
    n = len(X_test_raw)

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    print(f"\n=== RESULTATEN ===\n(Voor {n} polynomen)")
    print(f"NN predict tijd: {t_pred:.4f}s")

    print(
        f"Newton + NN pred x0: success= {ok_nn/n:.3f}, "
        f"gem iteraties= {np.mean(it_nn):.2f}, "
        f"time= {t_nn:.4f}s"
    )

    print(
        f"Newton + random x0: success= {ok_r/n:.3f}, "
        f"gem iteraties= {np.mean(it_r):.2f}, "
        f"time= {t_r:.4f}s"
    )

    print(
        f"\nRandom: {t_r:.4f}s - NN totaal: {t_pred+t_nn:.4f}s"
        f"\nWinst: {t_r-(t_pred+t_nn):.4f}s, {(t_r-t_pred+t_nn)/t_r*100:.4f}%\n"
    )
