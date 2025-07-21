import keras.regularizers
from keras.saving import register_keras_serializable
import tensorflow as tf


def nuclear_norm(matrix: tf.Tensor) -> tf.Tensor:
    """
    Computes the nuclear norm of a given matrix, aka. the trace norm (sum of singular values)
    """
    singular_matrix = tf.linalg.svd(matrix, compute_uv=False)
    return tf.reduce_sum(singular_matrix)

@register_keras_serializable(package="Custom", name="NuclearRegularizer")
class NuclearRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, coefficient: float):
        self.coefficient = coefficient

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self.coefficient * nuclear_norm(x)

    def get_config(self):
        return {'coefficient': self.coefficient}
