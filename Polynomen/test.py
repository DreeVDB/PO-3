import numpy as np
from scipy.optimize import fsolve

# Definieer de functie
def f(x):
    return np.exp(2*x) - 2*x - 2 * np.exp(4*x) + 2*x**2 + 5

# Maak een schatting (beginwaarde)
# Niet-lineaire vergelijkingen kunnen meerdere nulpunten hebben!
beginwaarde = 0

# Zoek het nulpunt en meet de tijd
import time
start = time.perf_counter()
nulpunt = fsolve(f, beginwaarde)
end = time.perf_counter()

duur = end - start
print(f"Het gevonden nulpunt is: {nulpunt[0]}")
print(f"Tijd voor root finding: {duur:.6f} seconden")