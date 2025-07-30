import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, Loss


class NormalDistributedCrossEntropy(Loss):
    def __init__(self, sigma=1, name=None, reduction='sum_over_batch_size'):
        super().__init__(name, reduction)
        self.sigma = sigma
        self.reduction = reduction

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        #  shape of y_true is a 2D matrix with batch on axis 0 and probability on axis 1.
        # Create an empty 2D matrix for the ground truth distribution, but softened.
        batch_size = tf.shape(y_pred)[0]
        num_classes = tf.shape(y_pred)[1]

        softened = tf.zeros(shape=(batch_size, num_classes))

        indices_center = tf.stack([tf.range(batch_size), y_true], axis=1)
        indices_left = tf.stack([tf.range(batch_size), tf.clip_by_value(y_true - 1, 0, num_classes - 1)], axis=1)
        indices_right = tf.stack([tf.range(batch_size), tf.clip_by_value(y_true + 1, 0, num_classes - 1)], axis=1)

        softened += tf.tensor_scatter_nd_add(softened, indices_center, tf.fill([batch_size], 1.0 - 2 * self.sigma))
        softened += tf.tensor_scatter_nd_add(softened, indices_left, tf.fill([batch_size], self.sigma))
        softened += tf.tensor_scatter_nd_add(softened, indices_right, tf.fill([batch_size], self.sigma))

        row_sums = tf.reduce_sum(softened, axis=1, keepdims=True)
        softened = softened / row_sums  # Normierung pro Sample

        cce_loss = CategoricalCrossentropy(reduction=self.reduction)
        return cce_loss(softened, y_pred)

