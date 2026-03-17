import numpy as np

from build_dataset import build_dataset
from build_model import build_model
from main import Standardizer, make_monic
from newton_raphson import newton_raphson_complex, poly_and_derivative
from root_utils import decode_roots_ri, sort_roots_canonical

### Illustratie van hoe 5 polynomen gegenereerd, gepreprocessed, aangepast door NN etc.
# Zo kunnen we de pipeline van V2 gestructureerd zien.


def train_with_main_v2_settings():
    # Zelfde instellingen als v2/main.py
    X, y = build_dataset(n_samples=20000)

    rng = np.random.default_rng(0)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n_train = int(0.8 * len(X))
    X_train_raw, y_train = X[:n_train], y[:n_train]
    X_test_raw, y_test = X[n_train:], y[n_train:]

    scaler = Standardizer().fit(make_monic(X_train_raw))
    X_train = scaler.transform(make_monic(X_train_raw)).astype(np.float32)

    model = build_model()
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=256,
        verbose=2,
    )

    return model, scaler, X_test_raw, y_test


def run_procedure_for_5_polynomials():
    model, scaler, X_test_raw, y_test = train_with_main_v2_settings()

    print("\nProcedure op 5 testpolynomen (na training met main.py instellingen):\n")

    X_nn = scaler.transform(make_monic(X_test_raw[:5])).astype(np.float32)
    y_pred = model(X_nn, training=False).numpy()

    for i in range(5):
        coeffs = X_test_raw[i]
        true_roots = sort_roots_canonical(decode_roots_ri(y_test[i]))
        nn_starts = decode_roots_ri(y_pred[i])

        found = []
        ok_flags = []
        iters = []
        residuals = []

        for x0 in nn_starts:
            x_hat, ok, it = newton_raphson_complex(coeffs, x0)
            fx, _ = poly_and_derivative(coeffs, x_hat)
            found.append(x_hat)
            ok_flags.append(ok)
            iters.append(it)
            residuals.append(abs(fx))

        found = sort_roots_canonical(np.array(found, dtype=np.complex128))
        success = bool(np.all(np.abs(found - true_roots) < 1e-5) and np.all(ok_flags))
        # True als alle 5 de wortels onder de NR tolerantie zitten

        print(f"Polynoom {i + 1}")
        print(f"  coeffs a5..a0: {np.array2string(coeffs, precision=4)}")
        print(f"  echte wortels: {np.array2string(true_roots, precision=4)}")
        print(f"  NN start (5x): {np.array2string(nn_starts, precision=4)}")
        print(f"  Newton output: {np.array2string(found, precision=4)}")
        print(f"  iteraties per wortel: {iters}")
        print(f"  |p(r)| per wortel: {np.array2string(np.array(residuals), precision=3)}")
        print(f"  alle 5 correct: {success}\n")


if __name__ == "__main__":
    run_procedure_for_5_polynomials()

# WE VINDEN SOMS MEERDERE KEREN DEZELFDE WORTEL!