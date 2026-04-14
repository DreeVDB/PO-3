"""
Stap 2: Laad de dataset, train het neuraal netwerk en sla het model op.
Het aantal lagen wordt meegegeven als parameter en opgenomen in de bestandsnaam.
"""

from pathlib import Path
import numpy as np

try:
    from NeuraalNetwerk import build_model
except ModuleNotFoundError:
    from NeuraalNetwerk import build_model


def layers_to_str(layers_sizes):
    """Zet een lijst van laaggroottes om naar een string, bv. [128, 64] -> '128-64'."""
    return "-".join(str(s) for s in layers_sizes)


def train_warm_start_model(X, y, n, m, k, layers_sizes, epochs=15, batch_size=64):
    model = build_model(n, m, k, layers_sizes=layers_sizes)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return model


def main(k=1, layers_sizes=None):
    n = 500
    m = 500
    epochs = 15
    batch_size = 64

    if layers_sizes is None:
        layers_sizes = [128, 128, 64]

    if k not in (0, 1):
        raise ValueError("k moet 0 of 1 zijn.")

    dataset_dir = (
        Path.home()
        / "OneDrive - KU Leuven"
        / "Bestanden van Dré Vandenbroeke - P&O3"
        / "dataset"
    )

    X_path = dataset_dir / f"X_n{n}_m{m}_k{k}.npy"
    y_path = dataset_dir / f"y_n{n}_m{m}_k{k}.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Dataset niet gevonden in {dataset_dir}. "
            "Voer eerst 1_generate_dataset.py uit."
        )

    print(f"Laad dataset van {dataset_dir}...")
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"  X.shape = {X.shape}, y.shape = {y.shape}")

    layers_str = layers_to_str(layers_sizes)
    print(f"Train neuraal netwerk met lagen {layers_sizes} (epochs={epochs}, batch_size={batch_size})...")
    model = train_warm_start_model(X, y, n, m, k, layers_sizes=layers_sizes, epochs=epochs, batch_size=batch_size)

    model_dir = (
        Path.home()
        / "OneDrive - KU Leuven"
        / "Bestanden van Dré Vandenbroeke - P&O3"
        / "quadratic models"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"quadratic_model_n{n}_m{m}_k{k}_layers{layers_str}.keras"
    model.save(model_path)
    print(f"Model opgeslagen naar {model_path}")
    return model_path


if __name__ == "__main__":
    main(k=1, layers_sizes=[128, 128, 64])
