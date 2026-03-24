import numpy as np

try:
    from SolveQPCasInt import SolveQPCasInt
    from SolveQPCasOases import SolveQPCasOases
except ModuleNotFoundError:
    from QuadraticProgram.SolveQPCasInt import SolveQPCasInt
    from QuadraticProgram.SolveQPCasOases import SolveQPCasOases


rng = np.random.default_rng()


def random_feasible_qp(n, ineq, eq, rng_instance=None):
    rng_local = rng if rng_instance is None else rng_instance

    B = rng_local.normal(size=(n, n))
    Q = B.T @ B

    c = rng_local.normal(size=n)
    x_hidden = rng_local.normal(size=n)

    A = rng_local.normal(size=(ineq, n))
    b = A @ x_hidden + rng_local.uniform(0.1, 1.0, size=ineq)

    Aeq = rng_local.normal(size=(eq, n))
    beq = Aeq @ x_hidden

    return Q, c, A, b, Aeq, beq


def Generate_QP_dataset(samples, n, ineq, eq, solver="interior", return_solver_stats=False, rng_instance=None):
    if solver == "interior":
        solver_fn = SolveQPCasInt
    elif solver == "oases":
        solver_fn = SolveQPCasOases
    else:
        raise ValueError("solver moet 'interior' of 'oases' zijn.")

    dataset = []
    rng_local = rng if rng_instance is None else rng_instance

    for _ in range(samples):
        Q, c, A, b, Aeq, beq = random_feasible_qp(n, ineq, eq, rng_instance=rng_local)
        if return_solver_stats:
            x, stats = solver_fn(Q, c, A, b, Aeq, beq, return_stats=True)
            dataset.append([(Q, c, A, b, Aeq, beq), x, stats])
        else:
            x = solver_fn(Q, c, A, b, Aeq, beq)
            dataset.append([(Q, c, A, b, Aeq, beq), x])

    return dataset


if __name__ == "__main__":
    data = Generate_QP_dataset(1, 2, 3, 1, solver="interior")
    print(data)
