import keras
from keras import layers, regularizers
import tensorflow as tf


@keras.utils.register_keras_serializable(package="QuadraticProgramV2")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    def __init__(self, positive_weight=1.0, name="weighted_binary_crossentropy"):
        super().__init__(name=name)
        self.positive_weight = float(positive_weight)

    def call(self, y_true, y_pred):
        positive_weight = tf.constant(self.positive_weight, dtype=tf.float32)
        epsilon = tf.constant(keras.backend.epsilon(), dtype=tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        weighted_loss = -(
            positive_weight * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(weighted_loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"positive_weight": self.positive_weight})
        return config


def build_model(
    n,
    m,
    k,
    hidden_layers=None,
    dropout_rate=0.0,
    l2_weight=0.0,
    learning_rate=1e-3,
    positive_weight=1.0,
    normalizer=None,
):
    input_size = n**2 + n + m * n + m + k * n + k
    if hidden_layers is None:
        hidden_layers = [384, 256, 128]

    inputs = layers.Input(shape=(input_size,))
    x = inputs

    if normalizer is not None:
        x = normalizer(x)

    for width in hidden_layers:
        x = layers.Dense(
            width,
            kernel_regularizer=regularizers.l2(l2_weight),
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("gelu")(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(m, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=WeightedBinaryCrossentropy(positive_weight=positive_weight),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
