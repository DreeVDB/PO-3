import casadi as ca
import numpy as np
from time import perf_counter


_SOLVER_CACHE = {}


def _pack_qp_parameters(Q, c, A, b, Aeq, beq):
    return np.concatenate([Q.reshape(-1), c, A.reshape(-1), b, Aeq.reshape(-1), beq])


def _reshape_row_major(vector, rows, cols):
    if rows == 0 or cols == 0:
        return ca.SX.zeros(rows, cols)
    return ca.transpose(ca.reshape(vector, cols, rows))


def _unpack_symbolic_parameters(parameters, n, m, k):
    offset = 0

    q_size = n * n
    Q = _reshape_row_major(parameters[offset : offset + q_size], n, n)
    offset += q_size

    c = parameters[offset : offset + n]
    offset += n

    a_size = m * n
    A = _reshape_row_major(parameters[offset : offset + a_size], m, n)
    offset += a_size

    b = parameters[offset : offset + m]
    offset += m

    aeq_size = k * n
    Aeq = _reshape_row_major(parameters[offset : offset + aeq_size], k, n)
    offset += aeq_size

    beq = parameters[offset : offset + k]
    return Q, c, A, b, Aeq, beq


def _build_solver(n, m, k):
    cache_key = (n, m, k)
    if cache_key in _SOLVER_CACHE:
        return _SOLVER_CACHE[cache_key]

    x = ca.MX.sym("x", n)
    parameter_size = n * n + n + m * n + m + k * n + k
    p = ca.MX.sym("p", parameter_size)
    Q, c, A, b, Aeq, beq = _unpack_symbolic_parameters(p, n, m, k)

    objective = 0.5 * ca.dot(x, ca.mtimes(Q, x)) + ca.dot(c, x)

    constraint_blocks = []
    lbg = []
    ubg = []
    if m > 0:
        constraint_blocks.append(ca.mtimes(A, x) - b)
        lbg.extend([-ca.inf] * m)
        ubg.extend([0.0] * m)
    if k > 0:
        constraint_blocks.append(ca.mtimes(Aeq, x) - beq)
        lbg.extend([0.0] * k)
        ubg.extend([0.0] * k)

    constraints = ca.vertcat(*constraint_blocks) if constraint_blocks else ca.MX.zeros(0, 1)
    nlp = {"x": x, "p": p, "f": objective, "g": constraints}

    options = {
        "ipopt.print_level": 0,
        "print_time": False,
        "ipopt.tol": 1e-10,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-10,
        "ipopt.compl_inf_tol": 1e-10,
        "ipopt.dual_inf_tol": 1e-10,
        "ipopt.max_iter": 2000,
    }

    solver = ca.nlpsol(f"solver_ipopt_{n}_{m}_{k}", "ipopt", nlp, options)
    cached = (solver, lbg, ubg)
    _SOLVER_CACHE[cache_key] = cached
    return cached


def SolveQPCasInt(Q, c, A, b, Aeq, beq, x0=None, return_stats=False):
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    Aeq = np.array(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)

    n = Q.shape[0]
    m = A.shape[0]
    k = Aeq.shape[0]
    solver, lbg, ubg = _build_solver(n, m, k)

    solver_inputs = {
        "p": _pack_qp_parameters(Q, c, A, b, Aeq, beq),
        "lbg": lbg,
        "ubg": ubg,
    }
    if x0 is not None:
        solver_inputs["x0"] = np.array(x0, dtype=float)

    start_time = perf_counter()
    sol = solver(**solver_inputs)
    solve_time = perf_counter() - start_time

    solution = np.array(sol["x"]).reshape(-1)
    if not return_stats:
        return solution

    solver_stats = solver.stats()
    stats = {
        "success": bool(solver_stats.get("success", False)),
        "iter_count": int(solver_stats.get("iter_count", 0)),
        "return_status": solver_stats.get("return_status", ""),
        "solve_time_seconds": solve_time,
    }
    return solution, stats
