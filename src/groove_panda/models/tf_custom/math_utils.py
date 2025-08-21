import numpy as np
import tensorflow as tf


def nuclear_norm(matrix: tf.Tensor) -> tf.Tensor:
    """
    Computes the nuclear norm of a given matrix, aka. the trace norm (sum of singular values)
    """
    singular_matrix = tf.linalg.svd(matrix, compute_uv=False)
    return tf.reduce_sum(singular_matrix)

def expected_mse(distribution: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    index_range = tf.cast(tf.shape(distribution)[0], tf.float32)
    max_possible_distance = index_range - 1

    indexes = tf.cast(tf.range(start = 0, limit = index_range), tf.float32)
    target = tf.cast(target, tf.float32)
    difference = indexes - target
    normalized_difference = difference / max_possible_distance
    mse_distances = normalized_difference ** 2
    expected_mse_value = tf.sqrt(tf.reduce_sum(mse_distances * distribution))

    return expected_mse_value  # noqa: RET504

def sum_under_range(distribution: tf.Tensor, limit: tf.Tensor) -> tf.Tensor:
    limit = tf.maximum(tf.cast(limit, tf.int32), 0)
    under_limit = distribution[:limit]
    return tf.reduce_sum(under_limit)

def generate_normal_distribution_array(sigma, epsilon):
    # Create a 1D tensor with the indices as float entries.
    # Those will be the arguments for the gauss probability function
    if sigma <= 0:
        return np.asarray([1])
    max_index = int(np.floor(sigma * np.sqrt(-2 * np.log(epsilon))))
    distribution_arguments = np.arange(-max_index, max_index + 1)
    distribution = np.exp(-0.5 * (distribution_arguments / sigma) ** 2)
    distribution = distribution / np.sum(distribution)

    return distribution  # noqa: RET504
    # Calculate he gauss probability function

