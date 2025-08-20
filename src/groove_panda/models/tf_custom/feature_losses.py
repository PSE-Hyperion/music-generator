from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.saving import register_keras_serializable

from groove_panda.models.tf_custom.losses import (
    CategoricalExpectedMSE,
    NormalDistributedCategorialKLDivergence,
    SumUnderRange,
)


@register_keras_serializable(package="Custom", name="Basic")
class Basic(Loss):
    def __init__(self, gausian_sigma, gausian_epsilon, distance_lambda, **kwargs):
        self.gausian_sigma = gausian_sigma
        self.gausian_epsilon = gausian_epsilon
        self.distance_lambda = distance_lambda
        super().__init__(self, **kwargs)

    def call(self, y_true, y_pred):
        loss_1 = NormalDistributedCategorialKLDivergence(self.gausian_sigma, self.gausian_epsilon)
        loss_2 = CategoricalExpectedMSE()
        return loss_1(y_true, y_pred) + loss_2(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Bar")
class Bar(Loss):
    """
    Uses cross entropy + expected distance + punishment for bar numbers lower than y_true - 1 (impossible bars)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        under_range = SumUnderRange()
        cross_entropy = SparseCategoricalCrossentropy()
        distance = CategoricalExpectedMSE()
        return under_range(y_true, y_pred) * 100 +  cross_entropy(y_true, y_pred) + distance(y_true, y_pred) * 50

@register_keras_serializable(package="Custom", name="Velocity")
class Velocity(Loss):
    """
    Uses soft KL divergence (normal distributed)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = 2
        self.epsilon = 1e-6
    def call(self, y_true, y_pred):
        soft_kl = NormalDistributedCategorialKLDivergence(sigma = self.sigma, epsilon = self.epsilon)
        return soft_kl(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Duration")
class Duration(Loss):
    """
    Uses soft KL divergence (normal distributed)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = 0.5
        self.epsilon = 1e-6
    def call(self, y_true, y_pred):
        soft_kl = NormalDistributedCategorialKLDivergence(sigma = self.sigma, epsilon = self.epsilon)
        return soft_kl(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Pitch")
class Pitch(Loss):
    """
    Uses cross entropy + expected distance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance_lambda = 10
    def call(self, y_true, y_pred):
        cross_entropy = SparseCategoricalCrossentropy()
        distance = CategoricalExpectedMSE()
        return self.distance_lambda * distance(y_true, y_pred) + cross_entropy(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Position")
class Position(Loss):
    """
    Uses cross entropy
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        cross_entropy = SparseCategoricalCrossentropy()
        return cross_entropy(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Tempo")
class Tempo(Loss):
    """
    Uses cross entropy + expected distance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        cross_entropy = SparseCategoricalCrossentropy()
        distance = CategoricalExpectedMSE()
        return cross_entropy(y_true, y_pred) + 10 * distance(y_true, y_pred)
