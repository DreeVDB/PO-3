import numpy as np
import highspy as hs

def SolveQPHS(Q, c, A, b, Aeq, beq):  
    n = Q.shape[0]
    m = A.shape[0]
    k = Aeq.shape[0]

    # HiGHS vereist dat ongelijkheden in de vorm Ax ≤ b zijn, dus we moeten A en b aanpassen
    # We zetten de ongelijkheden om naar -A x ≤ -b
    A_ineq = -A
    b_ineq = -b

    # HiGHS vereist ook dat gelijkheden in de vorm Aeq x = beq zijn, dus we kunnen deze direct gebruiken
    A_eq = Aeq
    b_eq = beq

    # Initialiseer de HiGHS solver
    model = hs.Highs()

    # Voeg variabelen toe (n variabelen)
    model.add_variables(n)

    # Stel de doelstelling in: min 0.5 x^T Q x + c^T x
    model.set_objective(Q, c)

    # Voeg ongelijkheden toe: A_ineq x ≤ b_ineq
    for i in range(m):
        model.add_constraint(A_ineq[i], sense='L', rhs=b_ineq[i])

    # Voeg gelijkheden toe: A_eq x = b_eq
    for i in range(k):
        model.add_constraint(A_eq[i], sense='E', rhs=b_eq[i])

    # Los het probleem op
    model.optimize()

    # Haal de oplossing op
    if model.status() == 'Optimal':
        solution = model.get_solution()
        return solution
    else:
        raise RuntimeError(f"HiGHS kon geen optimale oplossing vinden. Status: {model.status()}")