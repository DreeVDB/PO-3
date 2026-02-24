import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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