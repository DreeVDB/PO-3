import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt



## 1) Data generatie: 5de graad met exact 1 reële wortel


def make_quintic_with_one_real_root(rng: np.random.Generator):
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

    scale = rng.uniform(0.5, 2.0) * (1 if rng.random() < 0.5 else -1)
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



## 2) Preprocessing: monisch maken + standaardiseren


def make_monic(coeffs: np.ndarray) -> np.ndarray:
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


# 3) Newton-Raphson implementatie


def poly_and_derivative(coeffs: np.ndarray, x: float):
    a5, a4, a3, a2, a1, a0 = coeffs
    fx = (((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0)
    dfx = ((((5 * a5 * x + 4 * a4) * x + 3 * a3) * x + 2 * a2) * x + a1)
    return fx, dfx


def newton_raphson_real(coeffs: np.ndarray, x0: float,
                        tol=1e-10, max_iter=50, dfx_eps=1e-14):
    x = float(x0)
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



# 4) NN met aantal lagen als parameter


def build_model(n_layers=2, width=64):
    model = keras.Sequential()
    model.add(layers.Input(shape=(6,)))

    for _ in range(n_layers):
        model.add(layers.Dense(width, activation="relu"))

    model.add(layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model



# 5) Test NN tijd en convergentie

def benchmark(model, scaler, X_raw, y, max_iter_NR):

    X_scaled = scaler.transform(make_monic(X_raw)).astype(np.float32)
    preds = model(X_scaled, training=False).numpy().reshape(-1)

    ok = 0
    iters = []

    t0 = time.perf_counter()

    for coeffs, target, x0 in zip(X_raw, y.reshape(-1), preds):
        x_hat, success, it = newton_raphson_real(
            coeffs, float(x0), max_iter=max_iter_NR
        )
        ok += int(success)
        iters.append(it)

    T = time.perf_counter() - t0

    return {
        "success_rate": ok / len(X_raw),
        "avg_iters": float(np.mean(iters)),
        "time": T,
    }





def run_experiments(
    X_train, y_train,
    X_test_raw, y_test,
    scaler,
    epoch_list=[10, 30, 50],
    layer_list=[1, 2, 3],
    max_iter_list=[20, 50, 100]
):
    results = {}

    for epochs in epoch_list:
        for nlayers in layer_list:
            model = build_model(n_layers=nlayers)

            model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=epochs,
                batch_size=256,
                verbose=0
                )
            for maxit in max_iter_list:
                print(f"\n=== epochs={epochs}, layers={nlayers}, NR_max_iter={maxit} ===")
                stats = benchmark(model, scaler, X_test_raw, y_test, maxit)
                results[(epochs, nlayers, maxit)] = stats

    return results



# 7) Plotten


def plot_results(results):
    labels = []
    success = []
    times = []

    for (ep, nl, mi), r in results.items():
        labels.append(f"ep={ep}, L={nl}, it={mi}")
        success.append(r["success_rate"])
        times.append(r["time"])

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].barh(labels, success)
    ax[0].set_title("Convergentiepercentage")
    ax[0].set_xlabel("Success rate")
    ax[0].invert_yaxis()

    ax[1].barh(labels, times)
    ax[1].set_title("Totale uitvoeringstijd")
    ax[1].set_xlabel("Tijd (s)")
    ax[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

#8) Rangschik resultaten
def rank_results(results,epoch_list, layer_list, max_iter_list):
    batches= []
    for epochs in epoch_list:
        for nlayers in layer_list:
            for maxit in max_iter_list:
                batches.append((epochs, nlayers, maxit))
    results_time=[]
    for batch in batches:
        results_time.append((results[batch]['time'], batch))
    results_time.sort(key=lambda x: x[0])  # Sorteer op tijd
    results_success=[]
    for batch in batches:
        results_success.append((results[batch]['success_rate'], batch))
    results_success.sort(key=lambda x: x[0], reverse=True)  # Sorteer op success rate, hoogste eerst
    return (results_time, results_success)

###9) Ranking printen

def print_ranking(results_time, results_success):
    print("=== RANKING OP TIJD (TOP 5) ===")
    for rank, (time, batch) in enumerate(results_time[:5], 1):
        print(f"{rank}. Tijd: {time:.4f}s - Parameters: epochs={batch[0]}, layers={batch[1]}, max_iter={batch[2]}")

    print("\n=== RANKING OP CONVERGENTIE (TOP 5) ===")
    for rank, (success, batch) in enumerate(results_success[:5], 1):
        print(f"{rank}. Convergentie: {success:.4f} - Parameters: epochs={batch[0]}, layers={batch[1]}, max_iter={batch[2]}")


###10) Main functie
def main():
    seed = 0
    tf.random.set_seed(seed)    
    X, y = build_dataset(20000)

    rng = np.random.default_rng(0)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n_train = int(0.8 * len(X))

    X_train_raw, y_train = X[:n_train], y[:n_train]
    X_test_raw,  y_test  = X[n_train:], y[n_train:]

    scaler = Standardizer().fit(make_monic(X_train_raw))
    X_train = scaler.transform(make_monic(X_train_raw)).astype(np.float32)
    epoch_list=[15]
    layer_list=[3]
    max_iter_list=[10*i for i in range(1, 21)]
    results = run_experiments(
        X_train, y_train,
        X_test_raw, y_test,
        scaler,
        epoch_list,
        layer_list,
        max_iter_list
    )

    plot_results(results)
    print_ranking(*rank_results(results, epoch_list, layer_list, max_iter_list))


    epoch_list=[15]
    layer_list=[1,2,3,4,5]
    max_iter_list=[100]
    results = run_experiments(
        X_train, y_train,
        X_test_raw, y_test,
        scaler,
        epoch_list,
        layer_list,
        max_iter_list
    )

    plot_results(results)
    print_ranking(*rank_results(results, epoch_list, layer_list, max_iter_list))


    epoch_list=[5*i for i in range(1, 11)]
    layer_list=[3]
    max_iter_list=[100]
    results = run_experiments(
        X_train, y_train,
        X_test_raw, y_test,
        scaler,
        epoch_list,
        layer_list,
        max_iter_list
    )

    plot_results(results)
    print_ranking(*rank_results(results, epoch_list, layer_list, max_iter_list))
    return


    


main()