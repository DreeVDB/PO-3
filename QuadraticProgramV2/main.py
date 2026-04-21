import numpy as np


def flatten_sample(Q, c, A, b, Aeq, beq):
    return np.concatenate([Q.flatten(), c, A.flatten(), b, Aeq.flatten(), beq])


def active_set_from_solution(A, b, x, tolerance=1e-5):
    if A.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    residual = A @ x - b
    active_mask = np.abs(residual) <= tolerance
    return active_mask.astype(np.float32)


def build_dataset(raw_data, active_tolerance=1e-5):
    X = []
    y = []
    for problem, solution in raw_data:
        Q, c, A, b, Aeq, beq = problem
        X.append(flatten_sample(Q, c, A, b, Aeq, beq))
        y.append(active_set_from_solution(A, b, solution, tolerance=active_tolerance))

    return np.array(X), np.array(y)
