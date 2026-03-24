import casadi as ca
import numpy as np

def SolveQPCasOases(Q, c, A, b, Aeq, beq):
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    Aeq = np.array(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)

    n = Q.shape[0]
    m = A.shape[0]
    k = Aeq.shape[0]

    # Inequalities (Ax ≤ b) + equalities (Aeq x = beq)
    A_total = np.vstack([A, Aeq])

    lba = np.concatenate([
        -np.inf * np.ones(m),    # A x ≥ -∞
        beq                      # Aeq x ≥ beq
    ])
    uba = np.concatenate([
        b,                       # A x ≤ b
        beq                      # Aeq x ≤ beq
    ])

    # QPOASES expects:
    # H, g, A, lbx, ubx, lba, uba
    qp = {
        "H": ca.DM(Q),
        "g": ca.DM(c),
        "A": ca.DM(A_total),
        "lba": ca.DM(lba),
        "uba": ca.DM(uba),
        "lbx": -ca.inf * np.ones(n),
        "ubx":  ca.inf * np.ones(n)
    }

    opts = {
        "printLevel": "none",
        "maxIter": 1000,
        "terminationTolerance": 1e-10
    }

    solver = ca.qpsol("solver", "qpoases", qp, opts)

    sol = solver()

    return np.array(sol["x"]).reshape(-1)