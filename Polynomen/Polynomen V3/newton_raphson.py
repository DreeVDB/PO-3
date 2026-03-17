import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def poly_and_derivative(coeffs: np.ndarray, x):
    a5, a4, a3, a2, a1, a0 = coeffs
    fx = (((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0)
    dfx = ((((5 * a5 * x + 4 * a4) * x + 3 * a3) * x + 2 * a2) * x + a1)
    return fx, dfx


def newton_raphson_real(coeffs: np.ndarray, x0: float, tol=1e-10, max_iter=50, dfx_eps=1e-14):
    x = float(x0)               # Reëel getal
    for it in range(1, max_iter + 1):
        fx, dfx = poly_and_derivative(coeffs, x)

        if abs(fx) < tol:
            return x, True, it

        if abs(dfx) < dfx_eps:
            return x, False, it

        x = x - fx / dfx

        if not np.isfinite(x):
            return x, False, it

    fx, _ = poly_and_derivative(coeffs, x)
    return x, abs(fx) < tol, max_iter


def newton_raphson_complex(coeffs: np.ndarray, x0, tol=1e-10, max_iter=50, dfx_eps=1e-14):
    """Newton-Raphson op complexe getallen, gebruikt voor complexe wortels."""
    x = np.complex128(x0)               # Numpyvoorstelling complex getal => automatisch NR opl
    for it in range(1, max_iter + 1):
        fx, dfx = poly_and_derivative(coeffs, x)

        if abs(fx) < tol:
            return x, True, it

        if abs(dfx) < dfx_eps:
            return x, False, it

        x = x - fx / dfx

        if (not np.isfinite(np.real(x))) or (not np.isfinite(np.imag(x))):
            return x, False, it

    fx, _ = poly_and_derivative(coeffs, x)
    return x, abs(fx) < tol, max_iter