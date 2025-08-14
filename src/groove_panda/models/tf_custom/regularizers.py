from keras.src.saving import register_keras_serializable
import tensorflow as tf

from groove_panda.models.tf_custom.math_utils import nuclear_norm

"""
Regularizers are used to influence the way the parameters of a model evolve during the training.
"""


# Registering for keras, otherwise it has problems with models that use this regularizer, e.g. when loading a config
@register_keras_serializable(package="Custom", name="NuclearRegularizer")
class NuclearRegularizer(tf.keras.regularizers.Regularizer):  # type: ignore
    """
    This is an unusual regularizer for experimenting purposes.
    It is based on the nuclear norm (aka. shadow-1-norm or trace norm).
    It urges the parameter matrix to have small singular values.
    This norm has a stronger bias on smaller singular values
    (compared to e.g. Spectral Norm or Frobenius Norm (sqrt of the l2 norm in keras)),
    so it leads to a smaller rank of the matrix.
    This is especially helpful to analyze the actually used dimensions of layers, e.g. the embedding layer.
    """

    def __init__(self, lambda_: float):
        """
        :param lambda_: Coefficient of the norm in the loss function, determines the strength of the regularization.
        """
        self.lambda_ = lambda_

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self.lambda_ * nuclear_norm(x)

    def get_config(self):
        return {"lambda": self.lambda_}
