import casadi as ca
import numpy as np

def SolveQPCas(Q, c, A, b, Aeq, beq):
    n = Q.shape[0]

    # CasADi QP datastructuur
    qp = {}

    # 0.5 xᵀ Q x + cᵀ x
    qp['H'] = ca.DM(Q)
    qp['g'] = ca.DM(c)

    # Combineer A en Aeq
    A_total = A if Aeq is None else np.vstack([A, Aeq])
    qp['A'] = ca.DM(A_total)

    # Lower en upper bounds voor constraints
    m = A.shape[0]

    # Ongelijkheden: A x ≤ b  ->  laten we lba = -inf, uba = b
    lba = -ca.inf*m
    uba = b

    # Gelijkheden: Aeq x = beq -> lba = uba = beq
    if Aeq is not None:
        k = Aeq.shape[0]
        lba = np.concatenate([lba, beq])
        uba = np.concatenate([uba, beq])

    qp['lba'] = ca.DM(lba)
    qp['uba'] = ca.DM(uba)

    # Variabelen onbegrensd
    qp['lbx'] = -ca.inf*n
    qp['ubx'] = ca.inf*n

    # QP oplossen
    sol = ca.qp(qp)

    return np.array(sol['x']).reshape(-1)
