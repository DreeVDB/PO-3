from time import perf_counter

import numpy as np
import osqp
import scipy.sparse as sp


def _build_constraint_matrix(A, Aeq, n):
    blocks = []
    if A.shape[0] > 0:
        blocks.append(sp.csc_matrix(A))
    if Aeq.shape[0] > 0:
        blocks.append(sp.csc_matrix(Aeq))

    if not blocks:
        return sp.csc_matrix((0, n))
    return sp.vstack(blocks).tocsc()


def _build_bounds(b, beq):
    lower = []
    upper = []

    if b.size > 0:
        lower.append(-np.inf * np.ones_like(b, dtype=float))
        upper.append(np.array(b, dtype=float))
    if beq.size > 0:
        beq = np.array(beq, dtype=float)
        lower.append(beq)
        upper.append(beq)

    if not lower:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return np.concatenate(lower), np.concatenate(upper)


def SolveQP_OSQP(Q, c, A, b, Aeq, beq, x0=None, return_stats=False, tolerance=1e-5):
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    Aeq = np.array(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)

    n = Q.shape[0]
    P = sp.csc_matrix(0.5 * (Q + Q.T))
    A_total = _build_constraint_matrix(A, Aeq, n)
    l, u = _build_bounds(b, beq)

    prob = osqp.OSQP()
    prob.setup(
        P=P,
        q=c,
        A=A_total,
        l=l,
        u=u,
        verbose=False,
        eps_abs=float(tolerance),
        eps_rel=float(tolerance),
        polish=False,
    )
    if x0 is not None:
        prob.warm_start(x=np.array(x0, dtype=float).reshape(-1))

    start_time = perf_counter()
    res = prob.solve()
    solve_time = perf_counter() - start_time

    solution = np.array(res.x, dtype=float).reshape(-1) if res.x is not None else np.full(n, np.nan)
    if not return_stats:
        return solution

    info = res.info
    stats = {
        "success": info.status_val in (1, 2),
        "iter_count": int(info.iter),
        "return_status": str(info.status),
        "solve_time_seconds": solve_time,
    }
    return solution, stats
