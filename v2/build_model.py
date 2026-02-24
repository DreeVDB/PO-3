import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(6,)),   # 6 inputs (coëfficiënten)
            layers.Dense(64, activation="relu"),     # Relu: Max(0,x)
            layers.Dense(64, activation="relu"),
            layers.Dense(1),    # 1 output (reële wortel)
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        # model.compile bepaalt hoe het NN leert
        # Adam is een geavanceerde gradiënt descent
        # mse = mean squared error lossfunctie
    return model