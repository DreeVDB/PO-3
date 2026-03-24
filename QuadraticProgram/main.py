import numpy as np
from qpsolvers import solve_qp
import keras
from keras import layers
import numpy as np

from QuadraticProgram.QPGeneration import Generate_QP_dataset
from QuadraticProgram.NeuraalNetwerk import build_model


def flatten_sample(Q, c, A, b, Aeq, beq):
    # Het netwerk verwacht één 1D inputvector per sample.
    # Matrices worden rij-voor-rij platgemaakt (row-major, standaard in numpy),
    # daarna worden alle delen achter elkaar geplakt in dezelfde volgorde
    # als de input_size berekening in build_model.
    return np.concatenate([Q.flatten(), c, A.flatten(), b, Aeq.flatten(), beq])


def build_dataset(samples, n, m, k):
    raw_data = Generate_QP_dataset(samples, n, m, k)
    X = np.array([flatten_sample(*(p)) for (p, x) in raw_data])
    y = np.array([x for (p, x) in raw_data])
    return X, y


def main():
    # Hyperparameters
    n = 2
    m = 6
    k = 3
    samples = 1000
    val_split = 0.2
    epochs = 100
    batch_size = 32

    print("Genereer dataset...")
    X, y = build_dataset(samples, n, m, k)

    model = build_model(n, m, k)
    model.summary()

    print("Train model...")
    model.fit(X, y, validation_split=val_split, epochs=epochs, batch_size=batch_size)

    model_path = "quadratic_model.h5"
    model.save(model_path)
    print(f"Model opgeslagen naar {model_path}")

    # Eenvoudige evaluatie
    loss = model.evaluate(X[int(samples * (1 - val_split)):], y[int(samples * (1 - val_split)):], verbose=0)
    print(f"Eindverlies op validatie subset: {loss}")


if __name__ == "__main__":
    main()