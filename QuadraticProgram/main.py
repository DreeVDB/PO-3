import numpy as np
from qpsolvers import solve_qp 







def flatten_sample(Q, c, A, b, Aeq, beq):
    # Het netwerk verwacht één 1D inputvector per sample.
    # Matrices worden rij-voor-rij platgemaakt (row-major, standaard in numpy),
    # daarna worden alle delen achter elkaar geplakt in dezelfde volgorde
    # als de input_size berekening in build_model.
    return np.concatenate([Q.flatten(), c, A.flatten(), b, Aeq.flatten(), beq])

