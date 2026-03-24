
import casadi as ca
import numpy as np

def SolveQPCas(Q, c, A, b, Aeq, beq):
    """
    Lost een convex QP op:
        minimize 0.5 xᵀ Q x + cᵀ x
        subject to A x <= b
                   Aeq x = beq
    met CasADi's NLP solver (ipopt).
    """

    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    Aeq = np.array(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)

    n = Q.shape[0]

    # CasADi variabele
    x = ca.MX.sym("x", n)

    # QP-doelstelling
    obj = 0.5 * ca.dot(x, Q @ x) + ca.dot(c, x)

    # Constraints samenvoegen
    g_list = []

    # Ongelijkheden A x <= b → A x - b <= 0
    if A.shape[0] > 0:
        g_list.append(A @ x - b)

    # Gelijkheden Aeq x = beq
    if Aeq.shape[0] > 0:
        g_list.append(Aeq @ x - beq)

    g = ca.vertcat(*g_list) if len(g_list) > 0 else ca.MX([])

    # Bounds voor constraints
    ng = g.size1()

    # Ongelijkheden → [-∞, 0]
    lbg = []
    ubg = []

    if A.shape[0] > 0:
        lbg += [-ca.inf] * A.shape[0]
        ubg += [0.0] * A.shape[0]

    # Gelijkheden → [0, 0]
    if Aeq.shape[0] > 0:
        lbg += [0.0] * Aeq.shape[0]
        ubg += [0.0] * Aeq.shape[0]

    # Maak NLP
    nlp = {"x": x, "f": obj, "g": g}
    solver = ca.nlpsol("solver", "ipopt", nlp,
                       {"ipopt.print_level": 0,
                        "print_time": False})

    sol = solver(lbg=lbg, ubg=ubg)

    return np.array(sol["x"]).reshape(-1)