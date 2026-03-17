import keras
from keras import layers, regularizers
import qpsolvers
import numpy as np




def build_model():
    model = keras.Sequential([
        layers.Input(shape=(6,)),   # [Q00, Q01, Q10, Q11, c0, c1]
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(2),            # predicted x* = [x0, x1]
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

