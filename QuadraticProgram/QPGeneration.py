import numpy as np
import highspy as hs
from SolveQPHS import SolveQPHS

rng = np.random.default_rng()

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
        x = SolveQPHS(Q, c, A, b, Aeq, beq)
        dataset.append((Q, c, A, b, Aeq, beq, x))
    return dataset

data=Generate_QP_dataset(2, 2, 6, 3)
print(data)


