import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def random_feasible_qp(n, m, k, rng):
    """Generate a convex QP with at least one feasible point."""
    B = rng.normal(size=(n, n))
    Q = B.T @ B
    c = rng.normal(size=n)

    x_hidden = rng.normal(size=n)

    A = rng.normal(size=(m, n))
    b = A @ x_hidden + rng.uniform(0.1, 1.0, size=m)

    if k > 0:
        Aeq = rng.normal(size=(k, n))
        beq = Aeq @ x_hidden
    else:
        Aeq = np.empty((0, n))
        beq = np.empty(0)

    return Q, c, A, b, Aeq, beq, x_hidden


def objective_value(Q, c, x):
    return 0.5 * x @ Q @ x + c @ x


def max_violation(A, b, Aeq, beq, x):
    ineq_violation = 0.0
    eq_violation = 0.0

    if A.size:
        ineq_violation = float(np.maximum(A @ x - b, 0.0).max(initial=0.0))
    if Aeq.size:
        eq_violation = float(np.abs(Aeq @ x - beq).max(initial=0.0))

    return max(ineq_violation, eq_violation)


def project_equalities(x, Aeq, beq):
    if Aeq.size == 0:
        return x.copy()

    pseudo_inverse = np.linalg.pinv(Aeq)
    residual = Aeq @ x - beq
    return x - pseudo_inverse @ residual


def project_inequalities(x, A, b, sweeps=4):
    if A.size == 0:
        return x.copy()

    x_proj = x.copy()
    for _ in range(sweeps):
        for a_row, b_row in zip(A, b):
            violation = float(a_row @ x_proj - b_row)
            norm_sq = float(a_row @ a_row)
            if violation > 0.0 and norm_sq > 1e-12:
                x_proj = x_proj - (violation / norm_sq) * a_row
    return x_proj


def make_feasible(x, A, b, Aeq, beq, rounds=8):
    x_proj = x.copy()
    for _ in range(rounds):
        x_proj = project_equalities(x_proj, Aeq, beq)
        x_proj = project_inequalities(x_proj, A, b)
    return x_proj


def solve_with_history(Q, c, A, b, Aeq, beq, x0, iterations=10, step_size=0.35):
    """Simple projected-gradient style iteration to visualize the search path."""
    history = [x0.copy()]
    x = make_feasible(x0, A, b, Aeq, beq)
    history.append(x.copy())

    step = step_size
    for _ in range(iterations):
        grad = Q @ x + c
        candidate = x - step * grad
        candidate = make_feasible(candidate, A, b, Aeq, beq)

        current_value = objective_value(Q, c, x)
        candidate_value = objective_value(Q, c, candidate)

        tries = 0
        while candidate_value > current_value and tries < 8:
            step *= 0.5
            candidate = x - step * grad
            candidate = make_feasible(candidate, A, b, Aeq, beq)
            candidate_value = objective_value(Q, c, candidate)
            tries += 1

        x = candidate
        history.append(x.copy())

    return np.array(history)


def compute_plot_bounds(history, x_hidden):
    points = np.vstack([history, x_hidden.reshape(1, -1)])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    padding = 0.35 * span

    x_min, y_min = mins - padding
    x_max, y_max = maxs + padding
    return x_min, x_max, y_min, y_max


def draw_inequalities(ax, A, b, x_grid, y_grid):
    X, Y = np.meshgrid(x_grid, y_grid)

    for idx, (a_row, b_row) in enumerate(zip(A, b), start=1):
        values = a_row[0] * X + a_row[1] * Y - b_row
        upper = float(values.max())

        if upper > 0.0:
            ax.contourf(
                X,
                Y,
                values,
                levels=[0.0, upper],
                colors=["#f9d8d8"],
                alpha=0.25,
                hatches=["////"],
            )

        ax.contour(
            X,
            Y,
            values,
            levels=[0.0],
            colors=[f"C{(idx - 1) % 10}"],
            linewidths=1.6,
        )


def draw_equalities(ax, Aeq, beq, x_grid, y_grid):
    if Aeq.size == 0:
        return

    X, Y = np.meshgrid(x_grid, y_grid)
    for idx, (a_row, b_row) in enumerate(zip(Aeq, beq), start=1):
        values = a_row[0] * X + a_row[1] * Y - b_row
        ax.contour(
            X,
            Y,
            values,
            levels=[0.0],
            colors=[f"C{(idx + 4) % 10}"],
            linewidths=1.4,
            linestyles="--",
        )


def draw_objective_contours(ax, Q, c, x_grid, y_grid):
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([X, Y], axis=-1)
    values = 0.5 * np.einsum("...i,ij,...j->...", grid_points, Q, grid_points)
    values += grid_points @ c
    ax.contour(X, Y, values, levels=12, colors="0.75", linewidths=0.8)


def plot_problem_iterations(
    problem_index,
    Q,
    c,
    A,
    b,
    Aeq,
    beq,
    x_hidden,
    history,
    save_dir=None,
):
    x_min, x_max, y_min, y_max = compute_plot_bounds(history, x_hidden)
    x_grid = np.linspace(x_min, x_max, 300)
    y_grid = np.linspace(y_min, y_max, 300)

    panel_count = len(history)
    cols = min(4, math.ceil(math.sqrt(panel_count)))
    rows = math.ceil(panel_count / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows), squeeze=False)
    fig.suptitle(f"QP probleem {problem_index}: alle iteraties op een scherm", fontsize=15)

    optimum_guess = history[-1]

    for iteration, ax in enumerate(axes.flat):
        if iteration >= panel_count:
            ax.axis("off")
            continue

        draw_objective_contours(ax, Q, c, x_grid, y_grid)
        draw_inequalities(ax, A, b, x_grid, y_grid)
        draw_equalities(ax, Aeq, beq, x_grid, y_grid)

        path = history[: iteration + 1]
        ax.plot(path[:, 0], path[:, 1], color="black", linewidth=1.2, alpha=0.8)
        ax.scatter(path[:-1, 0], path[:-1, 1], color="black", s=20, alpha=0.65)
        ax.scatter(path[-1, 0], path[-1, 1], color="red", s=70, zorder=5)
        ax.scatter(optimum_guess[0], optimum_guess[1], color="green", marker="*", s=150, zorder=6)
        ax.scatter(x_hidden[0], x_hidden[1], color="royalblue", marker="x", s=70, zorder=6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        value = objective_value(Q, c, path[-1])
        violation = max_violation(A, b, Aeq, beq, path[-1])
        ax.set_title(
            f"Iteratie {iteration}\n"
            f"f(x) = {value:.3f}, max violatie = {violation:.2e}"
        )

    handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.2, label="pad van de iteraties"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="huidige gok"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="green", markersize=12, label="laatste gok"),
        plt.Line2D([0], [0], marker="x", color="royalblue", linestyle="None", markersize=8, label="verborgen feasible punt"),
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"qp_probleem_{problem_index}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    return fig


def plot_generated_qps(
    samples=2,
    n=2,
    m=4,
    k=1,
    iterations=8,
    seed=7,
    save_dir=None,
):
    if n != 2:
        raise ValueError("Deze visualisatie werkt alleen voor 2D-QP's (n=2).")

    rng = np.random.default_rng(seed)
    output_dir = Path(save_dir) if save_dir is not None else None
    figures = []

    for problem_index in range(1, samples + 1):
        Q, c, A, b, Aeq, beq, x_hidden = random_feasible_qp(n, m, k, rng)
        x0 = rng.normal(size=n) + np.array([2.0, -2.0])
        history = solve_with_history(Q, c, A, b, Aeq, beq, x0, iterations=iterations)
        fig = plot_problem_iterations(
            problem_index,
            Q,
            c,
            A,
            b,
            Aeq,
            beq,
            x_hidden,
            history,
            save_dir=output_dir,
        )
        figures.append(fig)

    return figures


if __name__ == "__main__":
    plot_generated_qps(samples=2, n=2, m=4, k=1, iterations=8, seed=7)
    plt.show()
