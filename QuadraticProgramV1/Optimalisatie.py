from itertools import product
from time import perf_counter
from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import tensorflow as tf

from main import flatten_sample
from NeuraalNetwerk import build_model
from QPGeneration import Generate_QP_dataset
from SolveQPCasInt import SolveQPCasInt
from SolveQPCasOases import SolveQPCasOases

from main import flatten_sample
from NeuraalNetwerk import build_model

from NeuraalNetwerk import build_model
from main2 import build_benchmark_dataset
from main2 import benchmark_interior
import QuadraticProgram



def benchmark_nn_hyperparameters(
    *,
    samples=100,
    n=4,
    m=4,
    k=1,
    seed=7,
    generation_tolerance=1e-10,
    oases_tolerance=1e-5,
    epoch_grid=(5, 10, 20),
    batch_size_grid=(32, 64),
    layer_grid=((128,), (256,), (256, 256)),
    save_csv=True,
):
    """
    Voert een volledige benchmark uit om te testen welke NN-configuratie
    het snelst qpOASES oplost via warm start.

    Test-parameters:
      - epochs
      - batch size
      - hidden layers

    Returns:
      Gesorteerde lijst van resultaten (snelste eerst)
    """

    print(
        f"Genereer {samples} QP-problemen (n={n}, m={m}, k={k})..."
    )

    problems, X, y = build_benchmark_dataset(
        samples,
        n,
        m,
        k,
        seed=seed,
        generation_tolerance=generation_tolerance,
    )

    results = []

    for epochs, batch_size, hidden_layers in product(
        epoch_grid, batch_size_grid, layer_grid
    ):
        print(
            f"\n▶ Test: epochs={epochs}, "
            f"batch={batch_size}, "
            f"layers={hidden_layers}"
        )

        # -------- Train NN --------
        train_start = perf_counter()
        model = build_model(n, m, k, hidden_layers=hidden_layers)
        model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )
        train_time = perf_counter() - train_start

        # -------- Benchmark qpOASES --------
        nn_stats = benchmark_oases_with_model(
            problems,
            model,
            tolerance=oases_tolerance,
        )

        summary = summarize_stats(
            name="qpOASES + NN warm start",
            stats_list=nn_stats,
        )

        summary.update(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_layers": hidden_layers,
                "train_time_seconds": train_time,
            }
        )

        results.append(summary)

        print(
            f"  → avg solve: {summary['avg_time']:.4f}s | "
            f"avg iter: {summary['avg_iter']:.1f}"
        )

    # -------- Sorteer op snelste solve --------
    results_sorted = sorted(results, key=lambda r: r["avg_time"])

    # -------- Print top resultaten --------
    print("\n=== Beste configuraties (snelste eerst) ===\n")
    print(
        f"{'Layers':<20}"
        f"{'Epochs':>8}"
        f"{'Batch':>8}"
        f"{'Train (s)':>12}"
        f"{'Avg solve (s)':>14}"
        f"{'Avg iter':>10}"
    )
    print("-" * 74)

    for r in results_sorted[:10]:
        print(
            f"{str(r['hidden_layers']):<20}"
            f"{r['epochs']:>8}"
            f"{r['batch_size']:>8}"
            f"{r['train_time_seconds']:>12.2f}"
            f"{r['avg_time']:>14.4f}"
            f"{r['avg_iter']:>10.2f}"
        )

    # -------- Optioneel opslaan --------
    if save_csv and results:
        csv_path = (
            Path(__file__).parent
            / f"benchmark_nn_n{n}_m{m}_k{k}.csv"
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=results[0].keys()
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResultaten opgeslagen in {csv_path}")

    return results_sorted

def main():
    benchmark_nn_hyperparameters(
        samples=100,
        n=4,
        m=4,
        k=1,
        epoch_grid=(3, 5, 10, 20),
        batch_size_grid=(32, 64, 128),
        layer_grid=((128,), (256,), (256, 256)),
    )


main()