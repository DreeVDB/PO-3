import casadi as ca
import numpy as np

def SolveQPCasInt(Q, c, A, b, Aeq, beq):
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    Aeq = np.array(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)

    n = Q.shape[0]
    x = ca.MX.sym("x", n)

    # Objective
    obj = 0.5 * ca.dot(x, Q @ x) + ca.dot(c, x)

    # Constraints
    g_list = []
    if A.shape[0] > 0:
        g_list.append(A @ x - b)
    if Aeq.shape[0] > 0:
        g_list.append(Aeq @ x - beq)
    g = ca.vertcat(*g_list) if g_list else ca.MX([])

    lbg = []
    ubg = []
    if A.shape[0] > 0:
        lbg += [-ca.inf] * A.shape[0]
        ubg += [0.0] * A.shape[0]
    if Aeq.shape[0] > 0:
        lbg += [0.0] * Aeq.shape[0]
        ubg += [0.0] * Aeq.shape[0]

    # Solver options (hier pas je methode, tolerantie, max iter aan)
    options = {
        "ipopt.print_level": 0,
        "print_time": False,

        # Toleranties voor nauwkeurigheid van de oplossing
        "ipopt.tol": 1e-10,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-10,
        "ipopt.compl_inf_tol": 1e-10,
        "ipopt.dual_inf_tol": 1e-10,

        # Max iterations
        "ipopt.max_iter": 2000,
    }

    nlp = {"x": x, "f": obj, "g": g}
    solver = ca.nlpsol("solver", "ipopt", nlp, options)
    sol = solver(lbg=lbg, ubg=ubg)

    return np.array(sol["x"]).reshape(-1)