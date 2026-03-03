import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


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