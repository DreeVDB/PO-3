import casadi as ca
import numpy as np

import casadi as ca
import numpy as np

def SolveQPCas(Q, c, A, b, Aeq, beq):
    """
    Lost een QP op:
        minimize 0.5 xᵀ Q x + cᵀ x
        subject to A x ≤ b
                   Aeq x = beq
    met CasADi's QP solver.
    """

    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    Aeq = np.array(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)

    n = Q.shape[0]
    m = A.shape[0]
    k = Aeq.shape[0]

    # QP datastructuur
    qp = {}

    qp["H"] = ca.DM(Q)
    qp["g"] = ca.DM(c)

    # Combineer ongelijkheid + gelijkheid constraints
    A_total = np.vstack([A, Aeq])
    qp["A"] = ca.DM(A_total)

    # Bounds maken
    lba = np.concatenate([-np.inf * np.ones(m), beq])
    uba = np.concatenate([b, beq])

    qp["lba"] = ca.DM(lba)
    qp["uba"] = ca.DM(uba)

    # Variable bounds (onbegrensd)
    qp["lbx"] = -ca.inf * np.ones(n)
    qp["ubx"] =  ca.inf * np.ones(n)

    # Los het QP op
    sol = ca.qp(qp)

    return np.array(sol["x"]).reshape(-1)
