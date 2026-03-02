import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


## 1) Data generatie: 5de graad met exact 1 reële wortel

# Stel getrained op [-2,2]: wat gebeurt er als je een root hierbuiten neemt.
# Uitbreiden van [-2,2] naar [-X,X]: onderzoek.

# Literatuurstudie: Bestaan er projecten die quasi op hetzelfde neerkomen?

# Trechterstructuur verslag: num methode, NN => samenbrengen = specifieker

# Elleboog voor BESTE epoch, max iter aanpassen , #layers,seed itvastzetten om te kunnen vergelijken met andere epoch en zelfde seed.

# Gokken op alle wortels


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

## 2) Preprocessing: monisch maken + standaardiseren


# Schalen geen invloed op wortels
# NN leert makkelijker op consistente data
def make_monic(coeffs: np.ndarray) -> np.ndarray:
    # monisch = coëfficiënt bij de hoogste graad is 1
    a5 = coeffs[..., 0:1]
    return coeffs / a5

# Zonder standardiseren kunnen grote coëfficiënten de learning domineren
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


## 3) Newton-raphson solver (reële wortel)

# fx en dfx berekenen voor de NR formule
def poly_and_derivative(coeffs: np.ndarray, x: float):
    a5, a4, a3, a2, a1, a0 = coeffs
    fx = (((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0)
    dfx = ((((5 * a5 * x + 4 * a4) * x + 3 * a3) * x + 2 * a2) * x + a1)
    return fx, dfx


def newton_raphson_real(coeffs: np.ndarray, x0: float, tol=1e-10, max_iter=50, dfx_eps=1e-14):

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

## 4) Klein Keras model

# Nieuwe functie, plot grootte NN en tijd

def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(6,)),   # 6 inputs (coëfficiënten)
            layers.Dense(64, activation="relu"),     # Relu: Max(0,x)
            layers.Dense(64, activation="relu"),
            layers.Dense(1),    # 1 output (reële wortel)
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        # model.compile bepaalt hoe het NN leert
        # Adam is een geavanceerde gradiënt descent
        # mse = mean squared error lossfunctie
    return model


## 5) Benchmark: 1 trial NN x0 vs 1 trial random x0


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

### plotten van functies epoch, maxiter en aantal layers

def build_model_nlayers(n_layers=2, width=64):
    """Bouw een simpel MLP met variabel aantal hidden layers."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(6,)))

    for _ in range(n_layers):
        model.add(layers.Dense(width, activation="relu"))

    model.add(layers.Dense(1))  # output
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def benchmark(model, scaler, X_raw, y, max_iter_NR):
    """Draai NR met NN-pred start & return: success rate, avg iters, total time."""
    X_nn = scaler.transform(make_monic(X_raw)).astype(np.float32)
    preds = model(X_nn, training=False).numpy().reshape(-1)

    ok = 0
    iters = []
    t0 = time.perf_counter()

    for coeffs, target, x0 in zip(X_raw, y.reshape(-1), preds):
        x_hat, succ, it = newton_raphson_real(
            coeffs, float(x0), max_iter=max_iter_NR
        )
        ok += succ
        iters.append(it)

    tot_time = time.perf_counter() - t0
    success_rate = ok / len(X_raw)
    avg_iters = np.mean(iters)
    return success_rate, avg_iters, tot_time


def run_experiments(
    X_train, y_train,
    X_test_raw, y_test,
    scaler,
    epoch_list=[10, 20, 40],
    layer_list=[1, 2, 3],
    max_iter_list=[20, 50, 100]
):

    results = {}

    for epochs in epoch_list:
        for nlayers in layer_list:
            for maxit in max_iter_list:

                print(f"\n=== RUN: epochs={epochs}, layers={nlayers}, max_iter={maxit} ===")

                # Model bouwen
                model = build_model_nlayers(n_layers=nlayers)

                # Training
                model.fit(
                    X_train, y_train,
                    validation_split=0.1,
                    epochs=epochs,
                    batch_size=256,
                    verbose=0
                )

                # Benchmark
                sr, ai, t_tot = benchmark(model, scaler, X_test_raw, y_test, maxit)

                results[(epochs, nlayers, maxit)] = {
                    "success": sr,
                    "avg_iters": ai,
                    "time": t_tot,
                }

    return results


def plot_results(results):
    """Plot convergentie & tijd als functie van parameters."""
    # Voor eenvoud: 2 grafieken naast elkaar
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    epochs = []
    succ = []
    times = []

    for (ep, nl, mi), r in results.items():
        label = f"ep={ep}, L={nl}, it={mi}"
        epochs.append(label)
        succ.append(r["success"])
        times.append(r["time"])

    # Convergentie
    axes[0].barh(epochs, succ)
    axes[0].set_title("Convergentie-percentage")
    axes[0].set_xlabel("Success rate")
    axes[0].invert_yaxis()

    # Tijd
    axes[1].barh(epochs, times)
    axes[1].set_title("Uitvoeringstijd (NR + NN-start)")
    axes[1].set_xlabel("Tijd [s]")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()



def main():
    # Data
    X, y = build_dataset(n_samples=20000)

    # shuffle + split
    rng = np.random.default_rng(0)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n_train = int(0.8 * len(X))
    X_train_raw, y_train = X[:n_train], y[:n_train]
    X_test_raw, y_test = X[n_train:], y[n_train:]

    # Preprocessing
    scaler = Standardizer().fit(make_monic(X_train_raw))
    X_train = scaler.transform(make_monic(X_train_raw)).astype(np.float32)
    X_test = scaler.transform(make_monic(X_test_raw)).astype(np.float32)

    # Model
    model = build_model()
    model.summary()

    # Train
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs = 50,
        batch_size=256,
        verbose=2
    )

    # Save
    model.save("quintic_root_model.keras")
    np.save("scaler_mu.npy", scaler.mu)
    np.save("scaler_sigma.npy", scaler.sigma)

    # Benchmark (1 trial voor beide)
    benchmark_one_trial(model, scaler, X_test_raw, y_test, rand_range=(-3.0, 3.0))


if __name__ == "__main__":
    main()