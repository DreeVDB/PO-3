from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import tensorflow as tf

try:
    from main import flatten_sample
    from NeuraalNetwerk import build_model
    from QPGeneration import Generate_QP_dataset
    from SolveQPCasInt import SolveQPCasInt
    from SolveQPCasOases import SolveQPCasOases
except ModuleNotFoundError:
    from main import flatten_sample
    from NeuraalNetwerk import build_model
    from QPGeneration import Generate_QP_dataset
    from SolveQPCasInt import SolveQPCasInt
    from SolveQPCasOases import SolveQPCasOases


def build_benchmark_dataset(samples, n, m, k, seed=None, generation_tolerance=1e-10):
    rng = np.random.default_rng(seed)
    raw_data = Generate_QP_dataset(
        samples=samples,
        n=n,
        ineq=m,
        eq=k,
        solver="interior",
        rng_instance=rng,
        solver_tolerance=generation_tolerance,
    )

    problems = [problem for (problem, _) in raw_data]
    X = np.array([flatten_sample(*problem) for (problem, _) in raw_data])
    y = np.array([solution for (_, solution) in raw_data])
    return problems, X, y


def benchmark_interior(problems, tolerance):
    stats_list = []
    for problem in problems:
        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasInt(Q, c, A, b, Aeq, beq, return_stats=True, tolerance=tolerance)
        stats["wall_time_seconds"] = stats["solve_time_seconds"]
        stats_list.append(stats)
    return stats_list


def benchmark_oases(problems, initial_guesses, tolerance):
    stats_list = []
    for problem, x0 in zip(problems, initial_guesses):
        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasOases(Q, c, A, b, Aeq, beq, x0=x0, return_stats=True, tolerance=tolerance)
        stats["wall_time_seconds"] = stats["solve_time_seconds"]
        stats_list.append(stats)
    return stats_list


def benchmark_oases_with_model(problems, model, tolerance):
    @tf.function(reduce_retracing=True)
    def fast_predict(x):
        return model(x, training=False)

    # Warm-up: eerste call triggert tf.function compilatie, niet meten
    dummy = flatten_sample(*problems[0]).reshape(1, -1).astype(np.float32)
    fast_predict(tf.constant(dummy))

    stats_list = []
    for problem in problems:
        sample = tf.constant(flatten_sample(*problem).reshape(1, -1).astype(np.float32))

        predict_start = perf_counter()
        x0 = fast_predict(sample).numpy()[0]
        predict_time = perf_counter() - predict_start

        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasOases(Q, c, A, b, Aeq, beq, x0=x0, return_stats=True, tolerance=tolerance)
        stats["predict_time_seconds"] = predict_time
        stats["wall_time_seconds"] = predict_time + stats["solve_time_seconds"]
        stats_list.append(stats)
    return stats_list


def summarize_stats(name, stats_list):
    total_time = sum(stat.get("wall_time_seconds", stat["solve_time_seconds"]) for stat in stats_list)
    avg_time = mean(stat.get("wall_time_seconds", stat["solve_time_seconds"]) for stat in stats_list)
    avg_iter = mean(stat["iter_count"] for stat in stats_list)
    success_rate = 100.0 * mean(1.0 if stat["success"] else 0.0 for stat in stats_list)

    return {
        "name": name,
        "total_time": total_time,
        "avg_time": avg_time,
        "avg_iter": avg_iter,
        "success_rate": success_rate,
    }


def print_summary_table(summaries):
    header = f"{'Methode':<28}{'Totale tijd (s)':>18}{'Gem. tijd (s)':>18}{'Gem. iteraties':>18}{'Succes (%)':>14}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary['name']:<28}"
            f"{summary['total_time']:>18.3f}"
            f"{summary['avg_time']:>18.3f}"
            f"{summary['avg_iter']:>18.2f}"
            f"{summary['success_rate']:>14.2f}"
        )


def train_warm_start_model(X, y, n, m, k, epochs=12, batch_size=64):
    model = build_model(n, m, k)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    return model


def main(k=1):
    samples = 100
    n = 100
    m = 50
    epochs = 15
    batch_size = 64
    seed = 7
    generation_tolerance = 1e-10
    interior_comparison_tolerance = 1e-5
    oases_comparison_tolerance = 1e-5

    if k not in (0, 1):
        raise ValueError("k moet 0 of 1 zijn.")

    print(f"Genereer {samples} QP-problemen met n={n}, m={m}, k={k}...")
    problems, X, y = build_benchmark_dataset(
        samples,
        n,
        m,
        k,
        seed=seed,
        generation_tolerance=generation_tolerance,
    )

    print("Benchmark interior point (IPOPT)...")
    interior_stats = benchmark_interior(problems, tolerance=interior_comparison_tolerance)

    print("Train neuraal netwerk voor warm start...")
    model = train_warm_start_model(X, y, n, m, k, epochs=epochs, batch_size=batch_size)

    rng = np.random.default_rng(seed + 1)
    random_warm_starts = rng.uniform(-100, 100, size=(samples, n))

    print("Benchmark qpOASES met random startgok...")
    random_stats = benchmark_oases(problems, random_warm_starts, tolerance=oases_comparison_tolerance)

    print("Benchmark qpOASES met neural network warm start, 1 QP per predict-call...")
    nn_stats = benchmark_oases_with_model(problems, model, tolerance=oases_comparison_tolerance)

    summaries = [
        summarize_stats("Interior point (IPOPT)", interior_stats),
        summarize_stats("qpOASES random warm start", random_stats),
        summarize_stats("qpOASES NN warm start", nn_stats),
    ]

    print()
    print_summary_table(summaries)

    model_path = Path.home() / "OneDrive - KU Leuven" / "Bestanden van Dré Vandenbroeke - P&O3" / "quadratic models" / f"quadratic_model_n{n}_m{m}_k{k}.keras"
    model.save(model_path)
    print()
    print(f"Model opgeslagen naar {model_path}")


if __name__ == "__main__":
    main(k=1)