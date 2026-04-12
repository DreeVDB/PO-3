import osqp
import numpy as np
import scipy.sparse as sp

def SolveQP(Q, c, A, b, Aeq, beq):
    Q = sp.csc_matrix(Q)
    A_total = sp.vstack([A, Aeq]).tocsc()

    # inequality: A x <= b
    # equality: Aeq x = beq -> encoded as lower = upper = beq
    l = np.hstack([-np.inf*np.ones(len(b)), beq])
    u = np.hstack([b, beq])

    prob = osqp.OSQP()
    prob.setup(P=Q, q=c, A=A_total, l=l, u=u, verbose=False)
    res = prob.solve()

    return res.x