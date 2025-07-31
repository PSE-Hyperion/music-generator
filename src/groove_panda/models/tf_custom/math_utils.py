import numpy as np
import tensorflow as tf


def nuclear_norm(matrix: tf.Tensor) -> tf.Tensor:
    """
    Computes the nuclear norm of a given matrix, aka. the trace norm (sum of singular values)
    """
    singular_matrix = tf.linalg.svd(matrix, compute_uv=False)
    return tf.reduce_sum(singular_matrix)

def get_normal_distribution_array(sigma, epsilon):
    # Create a 1D tensor with the indices as float entries.
    # Those will be the arguments for the gauss probability function
    max_index = int(np.floor(sigma * np.sqrt(-2 * np.log(epsilon))))
    distribution_arguments = np.arange(-max_index, max_index + 1)
    distribution = np.exp(-0.5 * (distribution_arguments / sigma) ** 2)
    distribution = distribution / np.sum(distribution)

    return distribution
    # Calculate he gauss probability function

