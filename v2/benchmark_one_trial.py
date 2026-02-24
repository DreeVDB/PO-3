import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from newton_raphson_real import newton_raphson_real
from newton_raphson_real import poly_and_derivative

def make_monic(coeffs: np.ndarray) -> np.ndarray:
    # monisch = coëfficiënt bij de hoogste graad is 1
    a5 = coeffs[..., 0:1]
    return coeffs / a5

class Standardizer:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X: np.ndarray):
        self.mu = X.mean(axis=0, keepdims=True)
        self.sigma = X.std(axis=0, keepdims=True) + 1e-12 # Voorkomt delingen door nul
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma
    

def benchmark_one_trial(model, scaler: Standardizer, X_test_raw, y_test, rand_range=(-3.0, 3.0)):

    # X_test_raw = onbewerkte polynoomcoëfficiënten
    rng = np.random.default_rng()

    # NN voorspelling (1 startwaarde per polynoom)
    X_nn = scaler.transform(make_monic(X_test_raw)).astype(np.float32)

    t0 = time.perf_counter()
    x0_nn = model(X_nn, training=False).numpy().reshape(-1)
    t_pred = time.perf_counter() - t0

    # Newton-Raphson met NN-start
    t0 = time.perf_counter()
    ok_nn = 0
    it_nn = []
    err_nn = []
    res_nn = []

    for coeffs, target, x0 in zip(X_test_raw, y_test.reshape(-1), x0_nn):
        x_hat, ok, it = newton_raphson_real(coeffs, x0)
        fx, _ = poly_and_derivative(coeffs, x_hat)

        ok_nn += int(ok)
        it_nn.append(it)

        if ok:
            err_nn.append(abs(x_hat - target))
            res_nn.append(abs(fx))

    t_nn = time.perf_counter() - t0

    # Random start (1 trial per polynoom)
    t0 = time.perf_counter()
    ok_r = 0
    it_r = []
    err_r = []
    res_r = []

    lo, hi = rand_range
    for coeffs, target in zip(X_test_raw, y_test.reshape(-1)):
        x0 = rng.uniform(lo, hi)  # EXACT 1 random gok
        x_hat, ok, it = newton_raphson_real(coeffs, x0)
        fx, _ = poly_and_derivative(coeffs, x_hat)

        ok_r += int(ok)
        it_r.append(it)

        if ok:
            err_r.append(abs(x_hat - target))
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
        f"\nWinst: {t_r-t_pred+t_nn:.4f}s, {(t_r-t_pred+t_nn)/t_r*100:.4f}%\n"
    )