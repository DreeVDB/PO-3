import numpy as np
from qpsolvers import available_solvers, solve_qp


def main():
    print("Beschikbare qpsolvers backends:", available_solvers)

    if "highs" not in available_solvers:
        raise RuntimeError(
            "HiGHS is niet beschikbaar. Installeer in deze omgeving: "
            "python -m pip install qpsolvers highspy"
        )

    # Testprobleem:
    # min 0.5 * x^T P x + q^T x
    # s.t. x >= 0
    #
    # Met P = I en q = [-1, -1] is de verwachte oplossing x = [1, 1].
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([-1.0, -1.0])
    G = np.array([[-1.0, 0.0], [0.0, -1.0]])
    h = np.array([0.0, 0.0])
    x0 = np.array([0.2, 0.2])

    x = solve_qp(P, q, G, h, solver="highs", initvals=x0)

    print("Gevonden oplossing:", x)

    expected = np.array([1.0, 1.0])
    if x is None:
        raise RuntimeError("De solver gaf geen oplossing terug.")

    if np.allclose(x, expected, atol=1e-7):
        print("TEST GESLAAGD: qpsolvers + highspy werkt correct.")
    else:
        raise RuntimeError(
            f"TEST GEFAALD: verwacht {expected}, maar kreeg {x}."
        )


if __name__ == "__main__":
    main()