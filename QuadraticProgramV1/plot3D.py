from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from QuadraticProgram.plot2D import (
        max_violation,
        objective_value,
        random_feasible_qp,
        solve_with_history,
    )
except ModuleNotFoundError:
    from plot2D import (
        max_violation,
        objective_value,
        random_feasible_qp,
        solve_with_history,
    )


def compute_bounds_3d(history, x_hidden):
    points = np.vstack([history, x_hidden.reshape(1, -1)])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    padding = 0.35 * span

    lower = mins - padding
    upper = maxs + padding
    return np.column_stack([lower, upper])


def draw_constraint_plane(ax, a_row, b_row, bounds, color, resolution=20):
    dominant_idx = int(np.argmax(np.abs(a_row)))
    if abs(a_row[dominant_idx]) < 1e-12:
        return

    free_indices = [idx for idx in range(3) if idx != dominant_idx]
    u_idx, v_idx = free_indices

    u_values = np.linspace(bounds[u_idx, 0], bounds[u_idx, 1], resolution)
    v_values = np.linspace(bounds[v_idx, 0], bounds[v_idx, 1], resolution)
    U, V = np.meshgrid(u_values, v_values)

    W = (b_row - a_row[u_idx] * U - a_row[v_idx] * V) / a_row[dominant_idx]

    coords = [None, None, None]
    coords[u_idx] = U
    coords[v_idx] = V
    coords[dominant_idx] = W
    X, Y, Z = coords

    valid = (W >= bounds[dominant_idx, 0]) & (W <= bounds[dominant_idx, 1])
    X = np.where(valid, X, np.nan)
    Y = np.where(valid, Y, np.nan)
    Z = np.where(valid, Z, np.nan)

    ax.plot_surface(X, Y, Z, color=color, alpha=0.14, linewidth=0, shade=False)


def draw_trajectory(ax, history):
    segments = zip(history[:-1], history[1:])
    point_colors = plt.cm.Reds(np.linspace(0.35, 0.95, len(history)))

    for idx, (start, end) in enumerate(segments, start=1):
        direction = end - start
        ax.quiver(
            start[0],
            start[1],
            start[2],
            direction[0],
            direction[1],
            direction[2],
            color=point_colors[min(idx, len(point_colors) - 1)],
            linewidth=1.8,
            arrow_length_ratio=0.18,
        )

    ax.plot(history[:, 0], history[:, 1], history[:, 2], color="black", linewidth=1.4, alpha=0.75)
    ax.scatter(
        history[:-1, 0],
        history[:-1, 1],
        history[:-1, 2],
        c=point_colors[:-1],
        s=34,
        alpha=0.95,
        depthshade=False,
    )
    ax.scatter(
        history[-1, 0],
        history[-1, 1],
        history[-1, 2],
        color="red",
        s=110,
        depthshade=False,
        label="laatste rode gok",
    )

    for idx, point in enumerate(history):
        ax.text(point[0], point[1], point[2], str(idx), fontsize=8, color="black")


def plot_problem_3d(problem_index, Q, c, A, b, x_hidden, history, save_dir=None):
    bounds = compute_bounds_3d(history, x_hidden)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = [f"C{idx % 10}" for idx in range(len(A))]
    for color, a_row, b_row in zip(colors, A, b):
        draw_constraint_plane(ax, a_row, b_row, bounds, color)

    draw_trajectory(ax, history)
    ax.scatter(
        x_hidden[0],
        x_hidden[1],
        x_hidden[2],
        color="royalblue",
        marker="x",
        s=90,
        depthshade=False,
        label="verborgen feasible punt",
    )

    start = history[0]
    ax.scatter(
        start[0],
        start[1],
        start[2],
        color="darkorange",
        marker="s",
        s=70,
        depthshade=False,
        label="startpunt",
    )

    final_guess = history[-1]
    value = objective_value(Q, c, final_guess)
    violation = max_violation(A, b, np.empty((0, 3)), np.empty(0), final_guess)

    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_zlim(bounds[2, 0], bounds[2, 1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.view_init(elev=24, azim=38)
    ax.set_title(
        f"3D QP probleem {problem_index}\n"
        f"een figuur met volledig traject, f(x_eind) = {value:.3f}, max violatie = {violation:.2e}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"qp_3d_probleem_{problem_index}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    return fig


def plot_generated_qps_3d(samples=1, n=3, m=5, k=0, iterations=8, seed=7, save_dir=None):
    if n != 3:
        raise ValueError("Deze 3D-visualisatie werkt alleen voor n=3.")
    if k != 0:
        raise ValueError("Deze 3D-visualisatie verwacht k=0.")

    rng = np.random.default_rng(seed)
    figures = []

    for problem_index in range(1, samples + 1):
        Q, c, A, b, Aeq, beq, x_hidden = random_feasible_qp(n, m, k, rng)
        x0 = rng.normal(size=n) + np.array([2.0, -2.0, 1.5])
        history = solve_with_history(Q, c, A, b, Aeq, beq, x0, iterations=iterations)
        fig = plot_problem_3d(problem_index, Q, c, A, b, x_hidden, history, save_dir=save_dir)
        figures.append(fig)

    return figures


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "_plots_3d"
    plot_generated_qps_3d(samples=1, n=3, m=5, k=0, iterations=8, seed=7, save_dir=output_dir)
    plt.show()
