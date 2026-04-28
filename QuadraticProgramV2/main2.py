from pathlib import Path
import sys
from statistics import mean
from time import perf_counter

import numpy as np
import tensorflow as tf

try:
    from absl import logging as absl_logging
except ImportError:
    absl_logging = None


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from main import active_set_from_solution, flatten_sample
from NeuraalNetwerk import build_multitask_model
from QPGeneration import Generate_QP_dataset
from SolveQPCasOases import SolveQPCasOases


def build_working_set_warm_start(problem, x_guess, active_mask, ridge):
    Q, c, A, _, Aeq, _ = problem
    x0 = np.array(x_guess, dtype=float, copy=True)
    active_mask = np.array(active_mask, dtype=bool)

    if active_mask.size != A.shape[0]:
        raise ValueError("active_mask heeft niet de juiste lengte.")
    if not active_mask.any() and Aeq.shape[0] == 0:
        return x0, np.zeros(A.shape[0], dtype=float)

    blocks = []
    if active_mask.any():
        blocks.append(A[active_mask])
    if Aeq.shape[0] > 0:
        blocks.append(Aeq)

    stacked = np.vstack(blocks)
    compact_duals = np.linalg.lstsq(
        stacked @ stacked.T + ridge * np.eye(stacked.shape[0]),
        -stacked @ (Q @ x0 + c),
        rcond=None,
    )[0]
    if not np.all(np.isfinite(compact_duals)):
        return x0, np.zeros(A.shape[0] + Aeq.shape[0], dtype=float)

    lam_ineq = np.zeros(A.shape[0], dtype=float)
    active_count = int(active_mask.sum())
    if active_count:
        lam_ineq[active_mask] = np.maximum(compact_duals[:active_count], 0.0)

    lam_eq = compact_duals[active_count:] if Aeq.shape[0] > 0 else np.zeros(0, dtype=float)
    lam_g0 = np.concatenate([lam_ineq, lam_eq])
    return x0, np.nan_to_num(lam_g0, nan=0.0, posinf=0.0, neginf=0.0)


def run_oases_benchmark(
    problems,
    tolerance,
    predict_problem=None,
    threshold=None,
    ridge=None,
    cached_predictions=None,
):
    stats_list = []
    cached_predictions = [None] * len(problems) if cached_predictions is None else cached_predictions
    use_working_set = threshold is not None and ridge is not None

    for problem, cached in zip(problems, cached_predictions):
        x0 = None
        lam_g0 = None
        predict_time = 0.0
        active_count = 0

        if predict_problem is not None:
            if cached is None:
                x_guess, active_probabilities, predict_time = predict_problem(problem)
            else:
                x_guess, active_probabilities = cached
            x0 = np.array(x_guess, dtype=float, copy=True)

            if use_working_set:
                warm_start_begin = perf_counter()
                active_mask = active_probabilities >= threshold
                x0, lam_g0 = build_working_set_warm_start(problem, x_guess, active_mask, ridge)
                if cached is None:
                    predict_time += perf_counter() - warm_start_begin
                active_count = int(active_mask.sum())

        Q, c, A, b, Aeq, beq = problem
        _, stats = SolveQPCasOases(
            Q,
            c,
            A,
            b,
            Aeq,
            beq,
            x0=x0,
            lam_g0=lam_g0,
            return_stats=True,
            tolerance=tolerance,
        )
        stats["predict_time_seconds"] = predict_time
        stats["wall_time_seconds"] = predict_time + stats["solve_time_seconds"]
        if use_working_set:
            stats["predicted_active_count"] = active_count
        stats_list.append(stats)

    return stats_list


def print_summary_table(summaries):
    header = (
        f"{'Methode':<40}"
        f"{'Totale tijd (s)':>17}"
        f"{'Gem. tijd (s)':>17}"
        f"{'Gem. NN tijd (s)':>18}"
        f"{'Gem. solver tijd (s)':>22}"
        f"{'Gem. iteraties':>16}"
        f"{'Succes (%)':>12}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary['name']:<40}"
            f"{summary['total_time']:>17.5f}"
            f"{summary['avg_time']:>17.5f}"
            f"{summary['avg_nn_time']:>18.5f}"
            f"{summary['avg_solver_time']:>22.5f}"
            f"{summary['avg_iter']:>16.2f}"
            f"{summary['success_rate']:>12.2f}"
        )


def main(
    k=1,
    samples=300,
    n=200,
    m=60,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    seed=7,
    hidden_layers=(96, 64),
    x_head_width=48,
    active_head_width=32,
    dropout_rate=0.05,
    l2_weight=1e-6,
    learning_rate=5e-4,
    x_loss_weight=1.0,
    active_loss_weight=0.75,
    early_stopping_patience=5,
    reduce_lr_patience=2,
    generation_tolerance=1e-8,
    active_tolerance=1e-5,
    oases_comparison_tolerance=1e-5,
    threshold_candidates=(0.35, 0.50),
    ridge_candidates=(1e-6, 1e-4),
    save_artifacts=True,
):
    if k not in (0, 1):
        raise ValueError("k moet 0 of 1 zijn.")
    if n <= 0 or n > 200:
        raise ValueError("n moet tussen 1 en 200 liggen.")
    if m <= 0:
        raise ValueError("m moet strikt positief zijn.")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split moet strikt tussen 0 en 1 liggen.")

    print(f"Genereer {samples} QP-problemen met n={n}, m={m}, k={k} met qpOASES als referentie-oplosser...")
    raw_data = Generate_QP_dataset(
        samples=samples,
        n=n,
        ineq=m,
        eq=k,
        solver="oases",
        rng_instance=np.random.default_rng(seed),
        solver_tolerance=generation_tolerance,
    )
    problems = [problem for problem, _ in raw_data]
    X = np.array([flatten_sample(*problem) for problem, _ in raw_data], dtype=np.float32)
    y_x = np.array([solution for _, solution in raw_data], dtype=np.float32)
    y_active = np.array(
        [
            active_set_from_solution(problem[2], problem[3], solution, tolerance=active_tolerance)
            for problem, solution in raw_data
        ],
        dtype=np.float32,
    )

    if len(problems) < 2:
        raise ValueError("Er zijn minstens 2 samples nodig om te splitsen.")

    indices = np.random.default_rng(seed).permutation(len(problems))
    val_count = min(max(1, int(round(len(problems) * validation_split))), len(problems) - 1)
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

    X_train, X_val = X[train_indices], X[val_indices]
    y_x_train, y_x_val = y_x[train_indices], y_x[val_indices]
    y_active_train, y_active_val = y_active[train_indices], y_active[val_indices]
    val_problems = [problems[i] for i in val_indices]

    avg_active_train = float(np.mean(np.sum(y_active_train, axis=1)))
    positive_rate_train = float(np.mean(y_active_train))
    positive_weight = 1.0 if positive_rate_train <= 1e-8 else float(
        np.clip((1.0 - positive_rate_train) / positive_rate_train, 1.0, 20.0)
    )
    print(
        f"Train/validatie split: {len(train_indices)} / {len(val_indices)} samples, "
        f"gem. actieve ongelijkheden in train = {avg_active_train:.2f}, "
        f"positieve rate = {100.0 * positive_rate_train:.2f} %"
    )

    print("Train multitask neuraal netwerk voor x en working set...")
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    model = build_multitask_model(
        n,
        m,
        k,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight,
        learning_rate=learning_rate,
        positive_weight=positive_weight,
        normalizer=normalizer,
        x_head_width=x_head_width,
        active_head_width=active_head_width,
        x_loss_weight=x_loss_weight,
        active_loss_weight=active_loss_weight,
    )
    history = model.fit(
        X_train,
        {"x_output": y_x_train, "active_output": y_active_train},
        validation_data=(X_val, {"x_output": y_x_val, "active_output": y_active_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
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
        ],
        verbose=0,
    )
    print(
        f"Training afgerond. val_x_mae={history.history.get('val_x_output_mae', [float('nan')])[-1]:.5f}, "
        f"val_active_precision={history.history.get('val_active_output_precision', [float('nan')])[-1]:.5f}, "
        f"val_active_recall={history.history.get('val_active_output_recall', [float('nan')])[-1]:.5f}, "
        f"positive_weight={positive_weight:.2f}"
    )

    @tf.function(reduce_retracing=True)
    def predict_single(sample):
        return model(sample, training=False)

    predict_single(tf.zeros((1, X.shape[1]), dtype=tf.float32))

    def predict_problem(problem):
        sample = tf.convert_to_tensor(flatten_sample(*problem).reshape(1, -1), dtype=tf.float32)
        start_time = perf_counter()
        prediction = predict_single(sample)
        return (
            prediction["x_output"].numpy().reshape(-1),
            prediction["active_output"].numpy().reshape(-1),
            perf_counter() - start_time,
        )

    print("Voorspel validatieset QP per QP voor threshold/ridge tuning...")
    cached_val_predictions = []
    for problem in val_problems:
        x_guess, active_probabilities, _ = predict_problem(problem)
        cached_val_predictions.append((x_guess, active_probabilities))

    val_active_probabilities = np.array([probabilities for _, probabilities in cached_val_predictions], dtype=np.float32)
    y_true_bool = y_active_val >= 0.5

    print("Stem threshold en ridge af op de validatieset...")
    best_result = None
    tuning_rows = []
    for threshold in threshold_candidates:
        y_pred_bool = val_active_probabilities >= threshold
        tp = int(np.logical_and(y_true_bool, y_pred_bool).sum())
        fp = int(np.logical_and(~y_true_bool, y_pred_bool).sum())
        fn = int(np.logical_and(y_true_bool, ~y_pred_bool).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        intersections = np.logical_and(y_true_bool, y_pred_bool).sum(axis=1)
        unions = np.logical_or(y_true_bool, y_pred_bool).sum(axis=1)
        jaccard = np.ones_like(unions, dtype=np.float64)
        np.divide(intersections, unions, out=jaccard, where=unions > 0)
        diagnostics = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "exact_match": float(np.mean(np.all(y_true_bool == y_pred_bool, axis=1))),
            "mean_jaccard": float(np.mean(jaccard)),
        }

        for ridge in ridge_candidates:
            stats_list = run_oases_benchmark(
                val_problems,
                tolerance=oases_comparison_tolerance,
                threshold=threshold,
                ridge=ridge,
                cached_predictions=cached_val_predictions,
            )
            row = {
                "threshold": threshold,
                "ridge": ridge,
                "avg_wall_time": mean(stat["wall_time_seconds"] for stat in stats_list),
                "avg_iter": mean(stat["iter_count"] for stat in stats_list),
                "success_rate": mean(1.0 if stat["success"] else 0.0 for stat in stats_list),
                **diagnostics,
            }
            tuning_rows.append(row)

            if best_result is None:
                best_result = row
            elif row["success_rate"] > best_result["success_rate"] + 1e-12:
                best_result = row
            elif abs(row["success_rate"] - best_result["success_rate"]) <= 1e-12:
                if row["avg_wall_time"] < best_result["avg_wall_time"] - 1e-12:
                    best_result = row
                elif (
                    abs(row["avg_wall_time"] - best_result["avg_wall_time"]) <= 1e-12
                    and row["avg_iter"] < best_result["avg_iter"]
                ):
                    best_result = row

    print("Validatie voor working-set warm start:")
    for row in tuning_rows:
        print(
            f"  threshold={row['threshold']:.2f}, ridge={row['ridge']:.0e}, "
            f"gem. solver tijd={row['avg_wall_time']:.5f} s, "
            f"gem. iter={row['avg_iter']:.2f}, "
            f"succes={100.0 * row['success_rate']:.2f} %, "
            f"precision={100.0 * row['precision']:.2f} %, "
            f"recall={100.0 * row['recall']:.2f} %, "
            f"F1={row['f1']:.4f}, exact-match={100.0 * row['exact_match']:.2f} %"
        )
    print(f"Gekozen instelling: threshold={best_result['threshold']:.2f}, ridge={best_result['ridge']:.0e}")
    print(
        f"Validatie active-set kwaliteit: precision={100.0 * best_result['precision']:.2f} %, "
        f"recall={100.0 * best_result['recall']:.2f} %, "
        f"F1={best_result['f1']:.4f}, "
        f"exact-match={100.0 * best_result['exact_match']:.2f} %, "
        f"gem. Jaccard={best_result['mean_jaccard']:.4f}"
    )

    print("Benchmark qpOASES cold start...")
    cold_stats = run_oases_benchmark(problems, tolerance=oases_comparison_tolerance)
    print("Benchmark qpOASES met gok op de waarden...")
    value_stats = run_oases_benchmark(
        problems,
        tolerance=oases_comparison_tolerance,
        predict_problem=predict_problem,
    )
    print("Benchmark qpOASES met gok op de waarden en de working set...")
    working_set_stats = run_oases_benchmark(
        problems,
        tolerance=oases_comparison_tolerance,
        predict_problem=predict_problem,
        threshold=best_result["threshold"],
        ridge=best_result["ridge"],
    )

    summaries = []
    for name, stats_list in [
        ("qpOASES cold start", cold_stats),
        ("qpOASES met waarde-gok", value_stats),
        ("qpOASES met waarde-gok + working set", working_set_stats),
    ]:
        summaries.append(
            {
                "name": name,
                "total_time": sum(stat["wall_time_seconds"] for stat in stats_list),
                "avg_time": mean(stat["wall_time_seconds"] for stat in stats_list),
                "avg_nn_time": mean(stat.get("predict_time_seconds", 0.0) for stat in stats_list),
                "avg_solver_time": mean(stat["solve_time_seconds"] for stat in stats_list),
                "avg_iter": mean(stat["iter_count"] for stat in stats_list),
                "success_rate": 100.0 * mean(1.0 if stat["success"] else 0.0 for stat in stats_list),
                "avg_predicted_active": mean(stat.get("predicted_active_count", 0) for stat in stats_list),
            }
        )

    print()
    print_summary_table(summaries)
    print(f"Gem. voorspelde active constraints in de working-set variant: {summaries[-1]['avg_predicted_active']:.2f}")

    if not save_artifacts:
        return

    model_path = MODULE_DIR / f"quadratic_qpoases_multitask_n{n}_m{m}_k{k}.h5"
    metadata_path = MODULE_DIR / f"quadratic_qpoases_multitask_n{n}_m{m}_k{k}_meta.npz"
    previous_verbosity = None
    if absl_logging is not None:
        previous_verbosity = absl_logging.get_verbosity()
        absl_logging.set_verbosity(absl_logging.ERROR)

    try:
        model.save(model_path, include_optimizer=False)
    finally:
        if absl_logging is not None and previous_verbosity is not None:
            absl_logging.set_verbosity(previous_verbosity)

    np.savez(
        metadata_path,
        n=n,
        m=m,
        k=k,
        samples=samples,
        positive_weight=positive_weight,
        avg_active_train=avg_active_train,
        positive_rate_train=positive_rate_train,
        threshold=best_result["threshold"],
        ridge=best_result["ridge"],
        val_precision=best_result["precision"],
        val_recall=best_result["recall"],
        val_f1=best_result["f1"],
        val_exact_match=best_result["exact_match"],
        val_mean_jaccard=best_result["mean_jaccard"],
        hidden_layers=np.array(hidden_layers, dtype=np.int32),
        x_head_width=x_head_width,
        active_head_width=active_head_width,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight,
        learning_rate=learning_rate,
        x_loss_weight=x_loss_weight,
        active_loss_weight=active_loss_weight,
    )
    print()
    print(f"Model opgeslagen naar {model_path}")
    print(f"Warm-start metadata opgeslagen naar {metadata_path}")


if __name__ == "__main__":
    main()
