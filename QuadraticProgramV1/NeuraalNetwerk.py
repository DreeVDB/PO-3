import keras
from keras import layers
import qpsolvers
import numpy as np

def build_model(n, m, k, hidden_layers=None, aantal_layers=4, aantal_nodes=128):
    # Het netwerk krijgt alle QP-parameters als één platte vector:
    #   Q   : n×n matrix  → n²  waarden
    #   c   : n vector    → n   waarden
    #   A   : m×n matrix  → m·n waarden
    #   b   : m vector    → m   waarden
    #   Aeq : k×n matrix  → k·n waarden
    #   beq : k vector    → k   waarden
    # Totaal: n² + n + m·n + m + k·n + k
    input_size = n**2 + n + m*n + m + k*n + k
    if hidden_layers is None:
        if aantal_layers < 1:
            raise ValueError("aantal_layers moet minstens 1 zijn.")
        if aantal_nodes < 1:
            raise ValueError("aantal_nodes moet minstens 1 zijn.")
        hidden_layers = [aantal_nodes] * aantal_layers
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        *[layers.Dense(width, activation="relu") for width in hidden_layers],
        layers.Dense(n),  # predicted x*
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model
