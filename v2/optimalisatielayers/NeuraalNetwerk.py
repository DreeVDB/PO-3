import keras
from keras import layers
import qpsolvers
import numpy as np

def build_model(n, m, k,layers_sizes=[128, 128, 64]):
    # Het netwerk krijgt alle QP-parameters als één platte vector:
    #   Q   : n×n matrix  → n²  waarden
    #   c   : n vector    → n   waarden
    #   A   : m×n matrix  → m·n waarden
    #   b   : m vector    → m   waarden
    #   Aeq : k×n matrix  → k·n waarden
    #   beq : k vector    → k   waarden
    # Totaal: n² + n + m·n + m + k·n + k
    input_size = n**2 + n + m*n + m + k*n + k
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        *[layers.Dense(layers_sizes[i], activation="relu") for i in range(len(layers_sizes))],
        layers.Dense(n),  # predicted x*
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model