def build_model_test(n, m, k, n_layers=3, layer_size=128):
    input_dim = ...
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for _ in range(n_layers):
        x = tf.keras.layers.Dense(layer_size, activation="relu")(x)

    outputs = tf.keras.layers.Dense(n)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model