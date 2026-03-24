import numpy as np
import highspy as hs
import osqp
import numpy as np
import scipy.sparse as sp
from SolveQP import SolveQP
import casadi as ca
import numpy as np
from SolveQPCas import SolveQPCas

rng = np.random.default_rng()

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

def random_feasible_qp(n, m, k):
    # 1. Q is positief semidefinitief
    B = rng.normal(size=(n, n))
    Q = B.T @ B

    # 2. lineaire term
    c = rng.normal(size=n)

    # 3. kies een verborgen punt dat alles feasible maakt
    x_hidden = rng.normal(size=n)

    # 4. ongelijkheid Ax ≤ b
    A = rng.normal(size=(m, n))
    # b wordt gekozen zodat A x_hidden ≤ b altijd waar is
    b = A @ x_hidden + rng.uniform(0.1, 1.0, size=m)

    # 5. gelijkheid Aeq x = beq
    Aeq = rng.normal(size=(k, n))
    beq = Aeq @ x_hidden  # identiek dus altijd consistent

    return Q, c, A, b, Aeq, beq

def Generate_QP_dataset(samples, n, m, k):
    dataset = []
    for i in range(samples):
        Q, c, A, b, Aeq, beq = random_feasible_qp(n, m, k)
        x = SolveQP(Q, c, A, b, Aeq, beq)
        dataset.append([(Q, c, A, b, Aeq, beq), x])
    return dataset

data=Generate_QP_dataset(1, 2, 6, 3)
print(data)


