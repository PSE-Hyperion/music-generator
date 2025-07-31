import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, Loss


class SoftCategorialCrossEntropy(Loss):
    def __init__(self, distribution: list[float], true_idx: int, name=None, reduction='sum_over_batch_size'):
        super().__init__(name, reduction)
        self.distribution = tf.convert_to_tensor(distribution)
        self.true_idx = true_idx
        self.len_left = true_idx
        self.len_right = len(distribution) - true_idx
        self.reduction = reduction

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        #  shape of y_true is a 2D matrix with batch on axis 0 and probability on axis 1.
        # Create an empty 2D matrix for the ground truth distribution, but softened.
        batch_size = tf.shape(y_pred)[0]
        num_classes = tf.shape(y_pred)[1]


        def create_soft_label(class_idx):
            start_idx = tf.
            pass

        cce_loss = CategoricalCrossentropy(reduction=self.reduction)
        return cce_loss(softened, y_pred)

