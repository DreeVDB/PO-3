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


def random_feasible_qp(n, ineq, eq):
    # 1. Q is positief semidefinitief
    B = rng.normal(size=(n, n))
    Q = B.T @ B

    # 2. lineaire term
    c = rng.normal(size=n)

    # 3. kies een verborgen punt dat alles feasible maakt
    x_hidden = rng.normal(size=n)

    # 4. ongelijkheid Ax ≤ b
    A = rng.normal(size=(ineq, n))
    # b wordt gekozen zodat A x_hidden ≤ b altijd waar is
    b = A @ x_hidden + rng.uniform(0.1, 1.0, size=ineq)

    # 5. gelijkheid Aeq x = beq
    Aeq = rng.normal(size=(eq, n))
    beq = Aeq @ x_hidden  # identiek dus altijd consistent

    return Q, c, A, b, Aeq, beq

def Generate_QP_dataset(samples, n, ineq, eq):
    dataset = []
    for i in range(samples):
        Q, c, A, b, Aeq, beq = random_feasible_qp(n, ineq, eq)
        x = SolveQPCas(Q, c, A, b, Aeq, beq)
        dataset.append([(Q, c, A, b, Aeq, beq), x])
    return dataset

data=Generate_QP_dataset(1, 2, 6, 3)
print(data)


