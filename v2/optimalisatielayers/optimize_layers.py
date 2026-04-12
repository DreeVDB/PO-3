"""
Optimalisatie: vergelijk NN-architecturen met verschillende aantallen lagen.

Workflow:
  1. Genereer de dataset (of sla over als die al bestaat)
  2. Train een model per architectuur
  3. Benchmark: interior point, qpOASES random, qpOASES NN (per architectuur)
  4. Overzichtstabel + elleboogplot met baselines
"""

from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    from main import flatten_sample
    from QPGeneration import Generate_QP_dataset
    from NeuraalNetwerk import build_model
    from SolveQPCasInt import SolveQPCasInt
    from SolveQPCasOases import SolveQPCasOases
except ModuleNotFoundError:
    from main import flatten_sample
    from QPGeneration import Generate_QP_dataset
    from NeuraalNetwerk import build_model
    from SolveQPCasInt import SolveQPCasInt
    from SolveQPCasOases import SolveQPCasOases

from train_model import layers_to_str, train_warm_start_model


# ---------------------------------------------------------------------------
# Configuratie — pas hier aan
# ---------------------------------------------------------------------------
N = 200
M = 150
K = 1
SAMPLES = 500
EPOCHS = 20
BATCH_SIZE = 64
SEED = 7
GENERATION_TOLERANCE = 1e-10
BENCHMARK_TOLERANCE = 1e-5

# Elke entry is één te testen architectuur: lijst van neuronen per verborgen laag.
# Vaste breedte (128) maar variabel aantal lagen: 1 t/m 6 lagen.
LAYER_CONFIGS = [
    [256, 256, 256],          # huidige winnaar, als referentie erin houden
    [256, 256, 256, 256],     # nog een laag dieper
    [512, 512, 512],          # breder maar uniform
    [512, 256, 256],          # breed begin, dan uniform
    [256, 256, 128],          # iets smaller uitlopen
    [384, 384, 384],          # tussenin
    [512, 512, 256, 256],     # breed + uniform uiteinde
]

BASE_DIR = (
    Path.home()
    / "OneDrive - KU Leuven"
    / "Bestanden van Dré Vandenbroeke - P&O3"
)
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR  = BASE_DIR / "quadratic models"
PLOT_PATH  = BASE_DIR / "elbow_plot_layers.png"
# ---------------------------------------------------------------------------


def ensure_dataset():
    """Genereer de dataset als die nog niet bestaat."""
    X_path = DATASET_DIR / f"X_n{N}_m{M}_k{K}.npy"
    y_path = DATASET_DIR / f"y_n{N}_m{M}_k{K}.npy"
    problems_dir = DATASET_DIR / f"problems_n{N}_m{M}_k{K}"

    if X_path.exists() and y_path.exists() and problems_dir.exists():
        print("Dataset al aanwezig, generatie overgeslagen.")
        return

    print(f"Genereer {SAMPLES} QP-problemen met n={N}, m={M}, k={K}...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    raw_data = Generate_QP_dataset(
        samples=SAMPLES,
        n=N,
        ineq=M,
        eq=K,
        solver="interior",
        rng_instance=rng,
        solver_tolerance=GENERATION_TOLERANCE,
    )

    problems = [problem for (problem, _) in raw_data]
    X = np.array([flatten_sample(*problem) for (problem, _) in raw_data])
    y = np.array([solution for (_, solution) in raw_data])

    np.save(X_path, X)
    np.save(y_path, y)

    problems_dir.mkdir(parents=True, exist_ok=True)
    for i, (Q, c, A, b, Aeq, beq) in enumerate(problems):
        np.savez(problems_dir / f"problem_{i:04d}.npz", Q=Q, c=c, A=A, b=b, Aeq=Aeq, beq=beq)

    print(f"Dataset opgeslagen naar {DATASET_DIR}")


def load_problems():
    problems_dir = DATASET_DIR / f"problems_n{N}_m{M}_k{K}"
    problems = []
    for i in range(SAMPLES):
        data = np.load(problems_dir / f"problem_{i:04d}.npz")
        problems.append((data["Q"], data["c"], data["A"], data["b"], data["Aeq"], data["beq"]))
    return problems


def ensure_model(layers_sizes):
    """Train en sla het model op als het nog niet bestaat, geef het pad terug."""
    layers_str = layers_to_str(layers_sizes)
    model_path = MODEL_DIR / f"quadratic_model_n{N}_m{M}_k{K}_layers{layers_str}.keras"

    if model_path.exists():
        print(f"  Model voor lagen {layers_sizes} al aanwezig, training overgeslagen.")
        return model_path

    X = np.load(DATASET_DIR / f"X_n{N}_m{M}_k{K}.npy")
    y = np.load(DATASET_DIR / f"y_n{N}_m{M}_k{K}.npy")

    print(f"  Train model met lagen {layers_sizes}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = train_warm_start_model(
        X, y, N, M, K,
        layers_sizes=layers_sizes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    model.save(model_path)
    print(f"  Opgeslagen naar {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Benchmark-functies
# ---------------------------------------------------------------------------

def benchmark_interior(problems):
    """Één keer uitvoeren: architectuur-onafhankelijke baseline."""
    wall_times = []
    for problem in problems:
        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasInt(
            Q, c, A, b, Aeq, beq,
            return_stats=True,
            tolerance=BENCHMARK_TOLERANCE,
        )
        wall_times.append(stats["solve_time_seconds"])
    return mean(wall_times)


def benchmark_random(problems):
    """Één keer uitvoeren: architectuur-onafhankelijke baseline."""
    rng = np.random.default_rng(SEED + 1)
    random_starts = rng.uniform(-100, 100, size=(len(problems), N))
    wall_times = []
    for problem, x0 in zip(problems, random_starts):
        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasOases(
            Q, c, A, b, Aeq, beq,
            x0=x0,
            return_stats=True,
            tolerance=BENCHMARK_TOLERANCE,
        )
        wall_times.append(stats["solve_time_seconds"])
    return mean(wall_times)


def benchmark_nn(problems, model_path):
    """Benchmark qpOASES met NN warm start voor één architectuur."""
    model = tf.keras.models.load_model(model_path)

    @tf.function(reduce_retracing=True)
    def fast_predict(x):
        return model(x, training=False)

    # Warm-up
    dummy = tf.constant(flatten_sample(*problems[0]).reshape(1, -1).astype(np.float32))
    fast_predict(dummy)

    wall_times = []
    for problem in problems:
        sample = tf.constant(flatten_sample(*problem).reshape(1, -1).astype(np.float32))

        t0 = perf_counter()
        x0 = fast_predict(sample).numpy()[0]
        predict_time = perf_counter() - t0

        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasOases(
            Q, c, A, b, Aeq, beq,
            x0=x0,
            return_stats=True,
            tolerance=BENCHMARK_TOLERANCE,
        )
        wall_times.append(predict_time + stats["solve_time_seconds"])

    return mean(wall_times)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_elbow_plot(nn_results, avg_interior, avg_random):
    """
    nn_results : lijst van (layers_sizes, avg_nn_time)
    avg_interior: scalar gem. tijd interior point
    avg_random  : scalar gem. tijd random warm start
    """
    num_layers = [len(ls) for ls, _ in nn_results]
    nn_times   = [t for _, t in nn_results]
    labels     = [layers_to_str(ls) for ls, _ in nn_results]

    fig, ax = plt.subplots(figsize=(9, 5))

    # NN-lijn
    ax.plot(num_layers, nn_times, marker="o", linewidth=2, markersize=8,
            label="qpOASES NN warm start")
    for x, y, lbl in zip(num_layers, nn_times, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Baselines als horizontale stippellijnen
    ax.axhline(avg_interior, linestyle="--", linewidth=1.5, color="tab:orange",
               label=f"Interior point (IPOPT)  {avg_interior:.4f} s")
    ax.axhline(avg_random, linestyle="--", linewidth=1.5, color="tab:green",
               label=f"qpOASES random  {avg_random:.4f} s")

    ax.set_xlabel("Aantal verborgen lagen", fontsize=12)
    ax.set_ylabel("Gemiddelde wall-time per QP (s)", fontsize=12)
    ax.set_title("Elleboogplot: aantal lagen vs. oplostijd", fontsize=13)
    ax.set_xticks(num_layers)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Plot opgeslagen naar {PLOT_PATH}")
    plt.show()


# ---------------------------------------------------------------------------
# Overzichtstabel
# ---------------------------------------------------------------------------

def print_results_table(nn_results, avg_interior, avg_random):
    col_arch   = 30
    col_layers = 14
    col_time   = 16

    header = (
        f"{'Methode':<{col_arch}}"
        f"{'Aantal lagen':>{col_layers}}"
        f"{'Gem. tijd (s)':>{col_time}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    # Baselines bovenaan
    print(f"{'Interior point (IPOPT)':<{col_arch}}{'-':>{col_layers}}{avg_interior:>{col_time}.4f}")
    print(f"{'qpOASES random warm start':<{col_arch}}{'-':>{col_layers}}{avg_random:>{col_time}.4f}")
    print(sep)

    # NN-architecturen
    for layers_sizes, avg_time in nn_results:
        arch = f"NN layers {layers_to_str(layers_sizes)}"
        print(f"{arch:<{col_arch}}{len(layers_sizes):>{col_layers}}{avg_time:>{col_time}.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Stap 1: dataset
    ensure_dataset()

    # Stap 2: train alle modellen
    print("\n=== Training ===")
    model_paths = {}
    for layers_sizes in LAYER_CONFIGS:
        model_paths[tuple(layers_sizes)] = ensure_model(layers_sizes)

    # Stap 3: laad problemen
    print("\n=== Benchmark ===")
    problems = load_problems()

    # Baselines (architectuur-onafhankelijk, één keer)
    print("Benchmark interior point (IPOPT)...")
    avg_interior = benchmark_interior(problems)
    print(f"  Gem. wall-time: {avg_interior:.4f} s")

    print("Benchmark qpOASES met random warm start...")
    avg_random = benchmark_random(problems)
    print(f"  Gem. wall-time: {avg_random:.4f} s")

    # NN per architectuur
    nn_results = []
    for layers_sizes in LAYER_CONFIGS:
        key = tuple(layers_sizes)
        print(f"Benchmark qpOASES NN warm start — lagen {layers_sizes}...")
        avg_time = benchmark_nn(problems, model_paths[key])
        nn_results.append((layers_sizes, avg_time))
        print(f"  Gem. wall-time: {avg_time:.4f} s")

    # Stap 4: tabel + plot
    print()
    print_results_table(nn_results, avg_interior, avg_random)
    print()
    make_elbow_plot(nn_results, avg_interior, avg_random)


if __name__ == "__main__":
    main()
