from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable

from groove_panda.models.tf_custom.losses import (
    CategoricalExpectedMSE,
    NormalDistributedCategorialKLDivergence,
    SumUnderRange,
)


@register_keras_serializable(package="Custom", name="Bar")
class Bar(Loss):
    """
    Uses cross entropy + expected distance + punishment for bar numbers lower than y_true - 1 (impossible bars)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        under_range = SumUnderRange()
        hard_kl = NormalDistributedCategorialKLDivergence(sigma=0, epsilon=0)
        distance = CategoricalExpectedMSE()
        return under_range(y_true, y_pred) * 100 +  hard_kl(y_true, y_pred) + distance(y_true, y_pred) * 20

@register_keras_serializable(package="Custom", name="Velocity")
class Velocity(Loss):
    """
    Uses soft KL divergence (normal distributed)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        soft_kl = NormalDistributedCategorialKLDivergence(sigma=2, epsilon=1e-6)
        return soft_kl(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Duration")
class Duration(Loss):
    """
    Uses soft KL divergence (normal distributed)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        soft_kl = NormalDistributedCategorialKLDivergence(sigma=0.5, epsilon=1e-6)
        return soft_kl(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Pitch")
class Pitch(Loss):
    """
    Uses cross entropy + expected distance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        hard_kl = NormalDistributedCategorialKLDivergence(sigma=0, epsilon=0)
        distance = CategoricalExpectedMSE()
        return hard_kl(y_true, y_pred) + distance(y_true, y_pred) * 5

@register_keras_serializable(package="Custom", name="Position")
class Position(Loss):
    """
    Uses cross entropy
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        hard_kl = NormalDistributedCategorialKLDivergence(sigma=0, epsilon=0)
        return hard_kl(y_true, y_pred)

@register_keras_serializable(package="Custom", name="Tempo")
class Tempo(Loss):
    """
    Uses cross entropy + expected distance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        hard_kl = NormalDistributedCategorialKLDivergence(sigma=0, epsilon=0)
        distance = CategoricalExpectedMSE()
        return hard_kl(y_true, y_pred) + distance(y_true, y_pred) + distance(y_true, y_pred) * 10
