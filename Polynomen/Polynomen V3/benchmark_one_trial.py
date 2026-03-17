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


def _run_newton_from_starts(coeffs: np.ndarray, true_roots: np.ndarray, start_roots: np.ndarray):
    found_roots = []
    poly_iters = []
    poly_ok = True

    for x0 in start_roots:
        x_hat, ok, it = newton_raphson_complex(coeffs, x0)
        found_roots.append(x_hat)
        poly_iters.append(it)
        poly_ok = poly_ok and ok

    found_roots = sort_roots_canonical(np.array(found_roots, dtype=np.complex128))
    poly_success = poly_ok and _success_from_roots(found_roots, true_roots)

    err = []
    res = []
    if poly_success:
        err.extend(np.abs(found_roots - true_roots).tolist())
        for root_hat in found_roots:
            fx, _ = poly_and_derivative(coeffs, root_hat)
            res.append(abs(fx))

    return {
        "success": int(poly_success),
        "mean_iter": float(np.mean(poly_iters)),
        "errors": err,
        "residuals": res,
    }


def benchmark_single_polynomial(
    model,
    scaler: Standardizer,
    coeffs,
    y_true,
    rand_range=(-3.0, 3.0),
    rng=None,
):
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(1, -1)
    true_roots = sort_roots_canonical(decode_roots_ri(np.asarray(y_true, dtype=np.float64)))
    coeffs_1d = coeffs[0]

    if rng is None:
        rng = np.random.default_rng()

    X_nn = scaler.transform(make_monic(coeffs)).astype(np.float32)

    t0 = time.perf_counter()
    x0_nn_vec = model(X_nn, training=False).numpy()[0]
    t_pred = time.perf_counter() - t0

    t0 = time.perf_counter()
    nn_result = _run_newton_from_starts(coeffs_1d, true_roots, decode_roots_ri(x0_nn_vec))
    t_nn = time.perf_counter() - t0

    lo, hi = rand_range
    random_starts = np.array(
        [rng.uniform(lo, hi) + 1j * rng.uniform(lo, hi) for _ in range(5)],
        dtype=np.complex128,
    )

    t0 = time.perf_counter()
    random_result = _run_newton_from_starts(coeffs_1d, true_roots, random_starts)
    t_r = time.perf_counter() - t0

    return {
        "t_pred": t_pred,
        "t_nn": t_nn,
        "t_r": t_r,
        "nn": nn_result,
        "random": random_result,
    }


def benchmark_one_trial(model, scaler: Standardizer, X_test_raw, y_test, rand_range=(-3.0, 3.0)):
    rng = np.random.default_rng()

    ok_nn = 0
    it_nn = []
    err_nn = []
    res_nn = []
    t_pred_total = 0.0
    t_nn_total = 0.0

    ok_r = 0
    it_r = []
    err_r = []
    res_r = []
    t_r_total = 0.0

    for coeffs, y_true in zip(X_test_raw, y_test):
        result = benchmark_single_polynomial(
            model,
            scaler,
            coeffs,
            y_true,
            rand_range=rand_range,
            rng=rng,
        )

        t_pred_total += result["t_pred"]
        t_nn_total += result["t_nn"]
        t_r_total += result["t_r"]

        ok_nn += result["nn"]["success"]
        it_nn.append(result["nn"]["mean_iter"])
        err_nn.extend(result["nn"]["errors"])
        res_nn.extend(result["nn"]["residuals"])

        ok_r += result["random"]["success"]
        it_r.append(result["random"]["mean_iter"])
        err_r.extend(result["random"]["errors"])
        res_r.extend(result["random"]["residuals"])

    n = len(X_test_raw)

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    nn_total = t_pred_total + t_nn_total
    winst = t_r_total - nn_total
    winst_pct = (winst / t_r_total * 100.0) if t_r_total > 0 else float("nan")

    print(f"\n=== RESULTATEN ===\n(Voor {n} polynomen, 1 per keer getimed)")
    print(
        f"NN predict tijd: {t_pred_total:.4f}s totaal, "
        f"{t_pred_total / n:.6f}s per polynoom"
    )

    print(
        f"Newton + NN pred x0: success= {ok_nn/n:.3f}, "
        f"gem iteraties= {safe_mean(it_nn):.2f}, "
        f"time= {t_nn_total:.4f}s totaal, {t_nn_total / n:.6f}s per polynoom"
    )

    print(
        f"Newton + random x0: success= {ok_r/n:.3f}, "
        f"gem iteraties= {safe_mean(it_r):.2f}, "
        f"time= {t_r_total:.4f}s totaal, {t_r_total / n:.6f}s per polynoom"
    )

    print(
        f"\nRandom: {t_r_total:.4f}s - NN totaal: {nn_total:.4f}s"
        f"\nWinst: {winst:.4f}s, {winst_pct:.4f}%\n"
    )
