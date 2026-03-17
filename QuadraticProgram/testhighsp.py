import warnings
from qpsolvers import solve_qp
import numpy as np

# Eenvoudig QP-probleem:
# minimaliseer  0.5 * x^T P x + q^T x
# onder         Ax = b,  lb <= x <= ub
#
# Oplossing: x = [0.5, 0.5]

P = np.array([[2.0, 0.0],
              [0.0, 2.0]])  # positief definiet

q = np.array([-1.0, -1.0])

# Gelijkheidsbeperking: x1 + x2 = 1
A = np.array([[1.0, 1.0]])
b = np.array([1.0])

# Grenzen
lb = np.array([0.0, 0.0])
ub = np.array([1.0, 1.0])

print("HiGHS via qpsolvers testen...")
x = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver="highs")

print(f"Oplossing: x = {x}")
print(f"Verwacht:  x = [0.5, 0.5]")

tol = 1e-6
assert x is not None, "Solver gaf None terug!"
assert np.allclose(x, [0.5, 0.5], atol=tol), f"Oplossing klopt niet: {x}"

print("Test geslaagd!")