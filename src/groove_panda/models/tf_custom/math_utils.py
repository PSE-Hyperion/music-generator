import tensorflow as tf


def nuclear_norm(matrix: tf.Tensor) -> tf.Tensor:
    """
    Computes the nuclear norm of a given matrix, aka. the trace norm (sum of singular values)
    """
    singular_matrix = tf.linalg.svd(matrix, compute_uv=False)
    return tf.reduce_sum(singular_matrix)
