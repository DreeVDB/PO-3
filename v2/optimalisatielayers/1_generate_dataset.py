"""
Stap 1: Genereer QP-dataset en sla op naar schijf.
"""

from pathlib import Path
import numpy as np

try:
    from QuadraticProgramV1.main import flatten_sample
    from QPGeneration import Generate_QP_dataset
except ModuleNotFoundError:
    from QuadraticProgramV1.main import flatten_sample
    from QPGeneration import Generate_QP_dataset


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


def main(k=1):
    samples = 500
    n = 500
    m = 500
    seed = 7
    generation_tolerance = 1e-10

    if k not in (0, 1):
        raise ValueError("k moet 0 of 1 zijn.")

    dataset_dir = (
        Path.home()
        / "OneDrive - KU Leuven"
        / "Bestanden van Dré Vandenbroeke - P&O3"
        / "dataset"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Genereer {samples} QP-problemen met n={n}, m={m}, k={k}...")
    problems, X, y = build_benchmark_dataset(
        samples, n, m, k, seed=seed, generation_tolerance=generation_tolerance
    )

    # Sla X en y op als .npy
    np.save(dataset_dir / f"X_n{n}_m{m}_k{k}.npy", X)
    np.save(dataset_dir / f"y_n{n}_m{m}_k{k}.npy", y)

    # Sla elk probleem (Q, c, A, b, Aeq, beq) op als afzonderlijke arrays
    problems_dir = dataset_dir / f"problems_n{n}_m{m}_k{k}"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for i, (Q, c, A, b, Aeq, beq) in enumerate(problems):
        np.savez(
            problems_dir / f"problem_{i:04d}.npz",
            Q=Q, c=c, A=A, b=b, Aeq=Aeq, beq=beq,
        )

    print(f"Dataset opgeslagen naar {dataset_dir}")
    print(f"  X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"  {len(problems)} problemen opgeslagen in {problems_dir}")


if __name__ == "__main__":
    main(k=1)
