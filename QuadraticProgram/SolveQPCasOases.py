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
    m = A.shape[0]
    k = Aeq.shape[0]

    # Combineer inequality en equality constraints
    # A x <= b        → lower bound = -∞, upper bound = b
    # Aeq x = beq     → lower = upper = beq
    A_total = np.vstack([A, Aeq])

    lba = np.concatenate([
        -np.inf * np.ones(m),   # ongelijkheden
        beq                     # gelijkheden
    ])

    uba = np.concatenate([
        b,                      # ongelijkheden
        beq                     # gelijkheden
    ])

    # QP structure
    qp = {
        "h": ca.DM(Q),              # Hessian
        "g": ca.DM(c),              # linear term
        "a": ca.DM(A_total),        # constraint matrix
        "lba": ca.DM(lba),          # constraint lower bounds
        "uba": ca.DM(uba),          # constraint upper bounds
        "lbx": -ca.inf * np.ones(n),
        "ubx":  ca.inf * np.ones(n),
    }

    # QPOASES options
    opts = {
        "printLevel": "none",    # geen output
        "enableRegularisation": True,
        "maxIter": 1000,         # max aantal iteraties
        "terminationTolerance": 1e-10,  # tolerantie
        "boundRelaxation": 1e-12,
    }

    # Maak QPOASES solver
    solver = ca.qpsol("solver", "qpoases", qp, opts)

    sol = solver()

    return np.array(sol["x"]).reshape(-1)