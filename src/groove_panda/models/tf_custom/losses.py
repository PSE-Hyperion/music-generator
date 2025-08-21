from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence, Loss
from tensorflow.keras.saving import register_keras_serializable

from groove_panda.models.tf_custom import math_utils


@register_keras_serializable(package="Custom", name="SoftCategoricalKLDivergence")
class SoftCategoricalKLDivergence(Loss):
    def __init__(
        self,
        distribution: list[float]|np.ndarray,
        true_position: Literal['center', 'start'] = 'center',
        name=None,
        reduction='sum_over_batch_size',
        from_logits: bool = False,
    ):
        super().__init__(name, reduction)
        self.distribution = tf.convert_to_tensor(distribution, dtype=tf.float32)
        if true_position == 'center':
            self.true_idx = tf.cast(len(distribution) // 2, dtype=tf.int32)
        else:
            self.true_idx = tf.cast(0, dtype=tf.int32)
            
        self.len_left = self.true_idx
        self.len_right = tf.cast(len(distribution), dtype=tf.int32) - self.true_idx
        self.reduction = reduction
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        #  shape of y_true is a 2D matrix with batch on axis 0 and probability on axis 1.
        # Create an empty 2D matrix for the ground truth distribution, but softened.
        num_classes = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)


        def create_soft_label(class_idx):
            class_idx = tf.cast(class_idx, tf.int32)

            # Leerer Tensor
            soft = tf.zeros([num_classes], dtype=tf.float32)

            # Bereich, den wir einsetzen wollen
            start = class_idx - self.len_left
            dist_len = tf.shape(self.distribution)[0]

            # Ziel-Start und -Ende (geclipped)
            insert_start = tf.maximum(start, 0)
            insert_end = tf.minimum(start + dist_len, num_classes)

            # Welche Teile der Distribution passen rein?
            dist_start = tf.maximum(0, -start)
            dist_end = dist_start + (insert_end - insert_start)

            values_to_insert = self.distribution[dist_start:dist_end]
            values_to_insert = values_to_insert / tf.reduce_sum(values_to_insert)  # Normalisieren

            # Setzen mit tf.tensor_scatter_nd_update
            indices = tf.range(insert_start, insert_end)
            indices = tf.expand_dims(indices, 1)
            soft = tf.tensor_scatter_nd_update(soft, indices, values_to_insert)

            return soft  # noqa: RET504

        y_true_soft = tf.map_fn(create_soft_label, y_true, fn_output_signature=tf.float32)
        kl_loss = KLDivergence(reduction=self.reduction)
        return kl_loss(y_true_soft, y_pred)

@register_keras_serializable(package="Custom", name="NormalDistributedCategorialKLDivergence")
class NormalDistributedCategorialKLDivergence(SoftCategoricalKLDivergence):
    def __init__(
        self,
        sigma,
        epsilon,
        name=None,
        reduction='sum_over_batch_size',
        from_logits: bool = False,
    ):
        distribution = math_utils.generate_normal_distribution_array(sigma, epsilon)
        super().__init__(distribution=distribution, name=name, reduction=reduction, from_logits=from_logits)

@register_keras_serializable(package="Custom", name="CategoricalExpectedMSE")
class CategoricalExpectedMSE(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return tf.map_fn(
            lambda x: math_utils.expected_mse(x[0], x[1]),
            (y_pred, y_true),
            fn_output_signature=tf.float32
        )

@register_keras_serializable(package="Custom", name="CategoricalExpectedMSE")
class SumUnderRange(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return tf.map_fn(
            lambda x: math_utils.sum_under_range(x[0], x[1]),
            (y_pred, y_true),
            fn_output_signature=tf.float32
        )
