from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

import numpy as np
import tensorflow as tf


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from main import active_set_from_solution, flatten_sample
from NeuraalNetwerk import build_model
from QPGeneration import Generate_QP_dataset
from SolveQPCasInt import SolveQPCasInt
from SolveQPCasOases import SolveQPCasOases


def build_benchmark_dataset(samples, n, m, k, seed=None, generation_tolerance=1e-10, active_tolerance=1e-5):
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
    X = np.array([flatten_sample(*problem) for (problem, _) in raw_data], dtype=np.float32)
    y = np.array(
        [
            active_set_from_solution(problem[2], problem[3], solution, tolerance=active_tolerance)
            for (problem, solution) in raw_data
        ],
        dtype=np.float32,
    )
    return problems, X, y


def split_dataset(problems, X, y, validation_split, seed):
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split moet strikt tussen 0 en 1 liggen.")

    sample_count = len(problems)
    if sample_count < 2:
        raise ValueError("Er zijn minstens 2 samples nodig om train/validatie te splitsen.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)

    val_count = max(1, int(round(sample_count * validation_split)))
    val_count = min(val_count, sample_count - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_problems = [problems[i] for i in train_indices]
    val_problems = [problems[i] for i in val_indices]

    return (
        train_problems,
        X[train_indices],
        y[train_indices],
        val_problems,
        X[val_indices],
        y[val_indices],
    )


def compute_positive_weight(y):
    positive_rate = float(np.mean(y))
    if positive_rate <= 1e-8:
        return 1.0

    negative_rate = 1.0 - positive_rate
    return float(np.clip(negative_rate / positive_rate, 1.0, 25.0))


def create_input_normalizer(X_train):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    return normalizer


def benchmark_interior(problems, tolerance):
    stats_list = []
    for problem in problems:
        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasInt(Q, c, A, b, Aeq, beq, return_stats=True, tolerance=tolerance)
        stats["wall_time_seconds"] = stats["solve_time_seconds"]
        stats_list.append(stats)
    return stats_list


def benchmark_oases_cold(problems, tolerance):
    stats_list = []
    for problem in problems:
        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasOases(Q, c, A, b, Aeq, beq, return_stats=True, tolerance=tolerance)
        stats["wall_time_seconds"] = stats["solve_time_seconds"]
        stats_list.append(stats)
    return stats_list


def solve_kkt_warm_start(Q, c, A, b, Aeq, beq, active_mask, ridge):
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    Aeq = np.array(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)

    n = Q.shape[0]
    active_mask = np.array(active_mask, dtype=bool)
    active_rows = A[active_mask] if A.shape[0] else np.zeros((0, n))
    active_rhs = b[active_mask] if b.shape[0] else np.zeros(0)

    if Aeq.shape[0] > 0:
        constraint_matrix = np.vstack([active_rows, Aeq]) if active_rows.shape[0] > 0 else Aeq.copy()
        constraint_rhs = np.concatenate([active_rhs, beq]) if active_rows.shape[0] > 0 else beq.copy()
    else:
        constraint_matrix = active_rows
        constraint_rhs = active_rhs

    regularized_Q = Q + ridge * np.eye(n)

    if constraint_matrix.shape[0] == 0:
        return np.linalg.lstsq(regularized_Q, -c, rcond=None)[0]

    zero_block = np.zeros((constraint_matrix.shape[0], constraint_matrix.shape[0]))
    kkt_matrix = np.block(
        [
            [regularized_Q, constraint_matrix.T],
            [constraint_matrix, zero_block],
        ]
    )
    rhs = np.concatenate([-c, constraint_rhs])
    solution = np.linalg.lstsq(kkt_matrix, rhs, rcond=None)[0]
    return solution[:n]


def benchmark_oases_with_active_probabilities(
    problems,
    active_probabilities,
    tolerance,
    threshold,
    kkt_ridge,
    include_prediction_time=False,
    prediction_times=None,
):
    stats_list = []
    for idx, (problem, probabilities) in enumerate(zip(problems, active_probabilities)):
        warm_start_start = perf_counter()
        active_mask = probabilities >= threshold
        x0 = solve_kkt_warm_start(*problem, active_mask=active_mask, ridge=kkt_ridge)
        warm_start_time = perf_counter() - warm_start_start

        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasOases(Q, c, A, b, Aeq, beq, x0=x0, return_stats=True, tolerance=tolerance)

        predict_time = 0.0
        if include_prediction_time and prediction_times is not None:
            predict_time = float(prediction_times[idx])

        stats["predict_time_seconds"] = predict_time + warm_start_time
        stats["wall_time_seconds"] = stats["predict_time_seconds"] + stats["solve_time_seconds"]
        stats["predicted_active_count"] = int(np.count_nonzero(active_mask))
        stats_list.append(stats)
    return stats_list


def predict_active_probabilities(model, X, batch_size):
    start_time = perf_counter()
    probabilities = model.predict(X, batch_size=batch_size, verbose=0)
    total_predict_time = perf_counter() - start_time
    average_predict_time = total_predict_time / max(len(X), 1)
    prediction_times = np.full(len(X), average_predict_time, dtype=np.float64)
    return probabilities, prediction_times


def compute_active_set_diagnostics(y_true, active_probabilities, threshold):
    y_true_bool = np.array(y_true >= 0.5, dtype=bool)
    y_pred_bool = np.array(active_probabilities >= threshold, dtype=bool)

    tp = int(np.logical_and(y_true_bool, y_pred_bool).sum())
    fp = int(np.logical_and(~y_true_bool, y_pred_bool).sum())
    fn = int(np.logical_and(y_true_bool, ~y_pred_bool).sum())
    tn = int(np.logical_and(~y_true_bool, ~y_pred_bool).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    exact_match = float(np.mean(np.all(y_true_bool == y_pred_bool, axis=1)))

    true_active_counts = np.sum(y_true_bool, axis=1)
    pred_active_counts = np.sum(y_pred_bool, axis=1)
    intersections = np.logical_and(y_true_bool, y_pred_bool).sum(axis=1)
    unions = np.logical_or(y_true_bool, y_pred_bool).sum(axis=1)
    jaccard_per_sample = np.ones_like(unions, dtype=np.float64)
    np.divide(intersections, unions, out=jaccard_per_sample, where=unions > 0)

    positive_probs = active_probabilities[y_true_bool]
    negative_probs = active_probabilities[~y_true_bool]

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "mean_jaccard": float(np.mean(jaccard_per_sample)),
        "avg_true_active": float(np.mean(true_active_counts)),
        "avg_pred_active": float(np.mean(pred_active_counts)),
        "avg_count_error": float(np.mean(np.abs(pred_active_counts - true_active_counts))),
        "mean_positive_probability": float(np.mean(positive_probs)) if positive_probs.size else 0.0,
        "mean_negative_probability": float(np.mean(negative_probs)) if negative_probs.size else 0.0,
    }


def collect_diagnostic_examples(y_true, active_probabilities, threshold, limit=5):
    y_true_bool = np.array(y_true >= 0.5, dtype=bool)
    y_pred_bool = np.array(active_probabilities >= threshold, dtype=bool)

    mismatches = np.not_equal(y_true_bool, y_pred_bool)
    mismatch_counts = mismatches.sum(axis=1)
    order = np.argsort(-mismatch_counts)

    examples = []
    for idx in order[:limit]:
        tp = int(np.logical_and(y_true_bool[idx], y_pred_bool[idx]).sum())
        fp = int(np.logical_and(~y_true_bool[idx], y_pred_bool[idx]).sum())
        fn = int(np.logical_and(y_true_bool[idx], ~y_pred_bool[idx]).sum())
        examples.append(
            {
                "sample_index": int(idx),
                "true_active": int(y_true_bool[idx].sum()),
                "pred_active": int(y_pred_bool[idx].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_probability": float(np.mean(active_probabilities[idx])),
                "max_probability": float(np.max(active_probabilities[idx])),
            }
        )
    return examples


def print_active_set_diagnostics(name, diagnostics, examples=None):
    print(f"Diagnostiek {name}:")
    print(
        f"  precision={100.0 * diagnostics['precision']:.2f} %, "
        f"recall={100.0 * diagnostics['recall']:.2f} %, "
        f"F1={diagnostics['f1']:.4f}, "
        f"exact-match={100.0 * diagnostics['exact_match']:.2f} %, "
        f"gem. Jaccard={diagnostics['mean_jaccard']:.4f}"
    )
    print(
        f"  gem. true active={diagnostics['avg_true_active']:.2f}, "
        f"gem. predicted active={diagnostics['avg_pred_active']:.2f}, "
        f"gem. count error={diagnostics['avg_count_error']:.2f}"
    )
    print(
        f"  gem. kans op echte active={diagnostics['mean_positive_probability']:.4f}, "
        f"gem. kans op echte inactive={diagnostics['mean_negative_probability']:.4f}"
    )
    print(
        f"  confusion totals: TP={diagnostics['tp']}, FP={diagnostics['fp']}, "
        f"FN={diagnostics['fn']}, TN={diagnostics['tn']}"
    )

    if examples:
        print("  Moeilijkste voorbeelden op validatie:")
        for example in examples:
            print(
                f"    sample {example['sample_index']}: "
                f"true={example['true_active']}, pred={example['pred_active']}, "
                f"TP={example['tp']}, FP={example['fp']}, FN={example['fn']}, "
                f"gem.p={example['mean_probability']:.4f}, max.p={example['max_probability']:.4f}"
            )


def tune_active_set_hyperparameters(
    val_problems,
    y_val,
    val_probabilities,
    tolerance,
    threshold_candidates,
    ridge_candidates,
):
    best_result = None
    tuning_rows = []

    for threshold in threshold_candidates:
        diagnostics = compute_active_set_diagnostics(y_val, val_probabilities, threshold)
        for ridge in ridge_candidates:
            stats_list = benchmark_oases_with_active_probabilities(
                val_problems,
                val_probabilities,
                tolerance=tolerance,
                threshold=threshold,
                kkt_ridge=ridge,
                include_prediction_time=False,
            )
            avg_wall_time = mean(stat["wall_time_seconds"] for stat in stats_list)
            success_rate = mean(1.0 if stat["success"] else 0.0 for stat in stats_list)
            avg_iter = mean(stat["iter_count"] for stat in stats_list)
            avg_active = mean(stat["predicted_active_count"] for stat in stats_list)

            row = {
                "threshold": threshold,
                "ridge": ridge,
                "avg_wall_time": avg_wall_time,
                "success_rate": success_rate,
                "avg_iter": avg_iter,
                "avg_active": avg_active,
                "precision": diagnostics["precision"],
                "recall": diagnostics["recall"],
                "f1": diagnostics["f1"],
                "mean_jaccard": diagnostics["mean_jaccard"],
            }
            tuning_rows.append(row)

            if best_result is None:
                best_result = row
                continue

            if row["success_rate"] > best_result["success_rate"]:
                best_result = row
                continue

            if (
                abs(row["success_rate"] - best_result["success_rate"]) <= 1e-12
                and row["avg_wall_time"] < best_result["avg_wall_time"]
            ):
                best_result = row

    return best_result, tuning_rows


def summarize_stats(name, stats_list):
    total_time = sum(stat.get("wall_time_seconds", stat["solve_time_seconds"]) for stat in stats_list)
    avg_time = mean(stat.get("wall_time_seconds", stat["solve_time_seconds"]) for stat in stats_list)
    avg_iter = mean(stat["iter_count"] for stat in stats_list)
    avg_nn_time = mean(stat.get("predict_time_seconds", 0.0) for stat in stats_list)
    avg_solver_time = mean(stat["solve_time_seconds"] for stat in stats_list)
    success_rate = 100.0 * mean(1.0 if stat["success"] else 0.0 for stat in stats_list)
    avg_predicted_active = mean(stat.get("predicted_active_count", 0) for stat in stats_list)

    return {
        "name": name,
        "total_time": total_time,
        "avg_time": avg_time,
        "avg_nn_time": avg_nn_time,
        "avg_solver_time": avg_solver_time,
        "avg_iter": avg_iter,
        "success_rate": success_rate,
        "avg_predicted_active": avg_predicted_active,
    }


def print_summary_table(summaries):
    header = (
        f"{'Methode':<34}"
        f"{'Totale tijd (s)':>17}"
        f"{'Gem. tijd (s)':>17}"
        f"{'Gem. NN tijd (s)':>18}"
        f"{'Gem. solver tijd (s)':>22}"
        f"{'Gem. iteraties':>16}"
        f"{'Gem. active':>14}"
        f"{'Succes (%)':>12}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary['name']:<34}"
            f"{summary['total_time']:>17.5f}"
            f"{summary['avg_time']:>17.5f}"
            f"{summary['avg_nn_time']:>18.5f}"
            f"{summary['avg_solver_time']:>22.5f}"
            f"{summary['avg_iter']:>16.2f}"
            f"{summary['avg_predicted_active']:>14.2f}"
            f"{summary['success_rate']:>12.2f}"
        )


def print_tuning_summary(best_result, tuning_rows):
    print("Validatie voor active-set warm start:")
    for row in tuning_rows:
        print(
            f"  threshold={row['threshold']:.2f}, ridge={row['ridge']:.0e}, "
            f"gem. warm-start+tijd={row['avg_wall_time']:.5f} s, "
            f"succes={100.0 * row['success_rate']:.2f} %, "
            f"gem. iter={row['avg_iter']:.2f}, "
            f"gem. active={row['avg_active']:.2f}, "
            f"precision={100.0 * row['precision']:.2f} %, "
            f"recall={100.0 * row['recall']:.2f} %, "
            f"F1={row['f1']:.4f}, "
            f"Jaccard={row['mean_jaccard']:.4f}"
        )
    print(
        f"Gekozen validatie-instelling: threshold={best_result['threshold']:.2f}, "
        f"ridge={best_result['ridge']:.0e}"
    )


def train_active_set_model(
    X_train,
    y_train,
    X_val,
    y_val,
    n,
    m,
    k,
    hidden_layers,
    dropout_rate,
    l2_weight,
    learning_rate,
    epochs,
    batch_size,
    early_stopping_patience,
    reduce_lr_patience,
):
    positive_weight = compute_positive_weight(y_train)
    normalizer = create_input_normalizer(X_train)

    model = build_model(
        n,
        m,
        k,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight,
        learning_rate=learning_rate,
        positive_weight=positive_weight,
        normalizer=normalizer,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history, positive_weight


def main(k=1):
    # Aanpasbare instellingen
    samples = 300
    n = 200
    m = 150
    epochs = 40
    batch_size = 32
    validation_split = 0.2
    seed = 7
    hidden_layers = [384, 256, 128]
    dropout_rate = 0.10
    l2_weight = 1e-6
    learning_rate = 5e-4
    early_stopping_patience = 6
    reduce_lr_patience = 3
    generation_tolerance = 1e-10
    active_tolerance = 1e-5
    interior_comparison_tolerance = 1e-5
    oases_comparison_tolerance = 1e-5
    threshold_candidates = [0.30, 0.40, 0.50, 0.60, 0.70]
    ridge_candidates = [1e-10, 1e-8, 1e-6]

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
        active_tolerance=active_tolerance,
    )

    (
        train_problems,
        X_train,
        y_train,
        val_problems,
        X_val,
        y_val,
    ) = split_dataset(problems, X, y, validation_split=validation_split, seed=seed)

    avg_active_train = float(np.mean(np.sum(y_train, axis=1)))
    positive_rate_train = float(np.mean(y_train))
    print(
        f"Train/validatie split: {len(train_problems)} / {len(val_problems)} samples, "
        f"gem. actieve constraints in train = {avg_active_train:.2f}, "
        f"positieve rate = {100.0 * positive_rate_train:.2f} %"
    )

    print("Benchmark interior point (IPOPT)...")
    interior_stats = benchmark_interior(problems, tolerance=interior_comparison_tolerance)

    print("Train neuraal netwerk voor active-set voorspelling...")
    model, history, positive_weight = train_active_set_model(
        X_train,
        y_train,
        X_val,
        y_val,
        n,
        m,
        k,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
    )
    print(
        f"Training afgerond. Laatste train loss = {history.history['loss'][-1]:.5f}, "
        f"laatste val loss = {history.history['val_loss'][-1]:.5f}, "
        f"positive_weight = {positive_weight:.2f}"
    )

    print("Stem threshold en ridge af op validatieset...")
    val_probabilities, _ = predict_active_probabilities(model, X_val, batch_size=batch_size)
    val_diagnostics_candidates = {
        threshold: compute_active_set_diagnostics(y_val, val_probabilities, threshold)
        for threshold in threshold_candidates
    }
    best_result, tuning_rows = tune_active_set_hyperparameters(
        val_problems,
        y_val,
        val_probabilities,
        tolerance=oases_comparison_tolerance,
        threshold_candidates=threshold_candidates,
        ridge_candidates=ridge_candidates,
    )
    print_tuning_summary(best_result, tuning_rows)
    best_val_diagnostics = val_diagnostics_candidates[best_result["threshold"]]
    val_examples = collect_diagnostic_examples(y_val, val_probabilities, best_result["threshold"], limit=5)
    print_active_set_diagnostics("validatie", best_val_diagnostics, examples=val_examples)

    print("Benchmark qpOASES zonder NN warm start...")
    cold_stats = benchmark_oases_cold(problems, tolerance=oases_comparison_tolerance)

    print("Benchmark qpOASES met echte active set (oracle-bovengrens)...")
    oracle_stats = benchmark_oases_with_active_probabilities(
        problems,
        y,
        tolerance=oases_comparison_tolerance,
        threshold=0.5,
        kkt_ridge=best_result["ridge"],
        include_prediction_time=False,
    )

    print("Benchmark qpOASES met NN active-set warm start...")
    all_probabilities, prediction_times = predict_active_probabilities(model, X, batch_size=batch_size)
    all_diagnostics = compute_active_set_diagnostics(y, all_probabilities, best_result["threshold"])
    nn_stats = benchmark_oases_with_active_probabilities(
        problems,
        all_probabilities,
        tolerance=oases_comparison_tolerance,
        threshold=best_result["threshold"],
        kkt_ridge=best_result["ridge"],
        include_prediction_time=True,
        prediction_times=prediction_times,
    )
    print_active_set_diagnostics("volledige benchmarkset", all_diagnostics)

    summaries = [
        summarize_stats("Interior point (IPOPT)", interior_stats),
        summarize_stats("qpOASES cold start", cold_stats),
        summarize_stats("qpOASES oracle active set", oracle_stats),
        summarize_stats("qpOASES NN active-set warm start", nn_stats),
    ]

    print()
    print_summary_table(summaries)

    model_path = Path(__file__).resolve().parent / f"quadratic_active_set_model_n{n}_m{m}_k{k}.keras"
    metadata_path = Path(__file__).resolve().parent / f"quadratic_active_set_model_n{n}_m{m}_k{k}_meta.npz"
    model.save(model_path)
    np.savez(
        metadata_path,
        positive_weight=positive_weight,
        avg_active_train=avg_active_train,
        positive_rate_train=positive_rate_train,
        threshold=best_result["threshold"],
        ridge=best_result["ridge"],
        val_precision=best_val_diagnostics["precision"],
        val_recall=best_val_diagnostics["recall"],
        val_f1=best_val_diagnostics["f1"],
        val_mean_jaccard=best_val_diagnostics["mean_jaccard"],
        full_precision=all_diagnostics["precision"],
        full_recall=all_diagnostics["recall"],
        full_f1=all_diagnostics["f1"],
        full_mean_jaccard=all_diagnostics["mean_jaccard"],
    )

    print()
    print(f"Model opgeslagen naar {model_path}")
    print(f"Warm-start metadata opgeslagen naar {metadata_path}")


if __name__ == "__main__":
    main(k=1)
