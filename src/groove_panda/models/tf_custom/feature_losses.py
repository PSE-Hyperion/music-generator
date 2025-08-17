from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy

from groove_panda.models.tf_custom.losses import (
    CategoricalExpectedMSE,
    NormalDistributedCategorialKLDivergence,
    SumUnderRange,
)


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

class Bar(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        under_range = SumUnderRange()
        cross_entropy = SparseCategoricalCrossentropy()
        distance = CategoricalExpectedMSE()
        return under_range(y_true, y_pred) * 100 +  cross_entropy(y_true, y_pred) + distance(y_true, y_pred) * 50

class Velocity(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = 2
        self.epsilon = 1e-6
    def call(self, y_true, y_pred):
        soft_kl = NormalDistributedCategorialKLDivergence(sigma = self.sigma, epsilon = self.epsilon)
        return soft_kl(y_true, y_pred)

class Duration(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = 0.5
        self.epsilon = 1e-6
    def call(self, y_true, y_pred):
        soft_kl = NormalDistributedCategorialKLDivergence(sigma = self.sigma, epsilon = self.epsilon)
        return soft_kl(y_true, y_pred)

class Pitch(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance_lambda = 10
    def call(self, y_true, y_pred):
        cross_entropy = SparseCategoricalCrossentropy()
        distance = CategoricalExpectedMSE()
        return self.distance_lambda * distance(y_true, y_pred) + cross_entropy(y_true, y_pred)

class Position(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        cross_entropy = SparseCategoricalCrossentropy()
        return cross_entropy(y_true, y_pred)

class Tempo(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        cross_entropy = SparseCategoricalCrossentropy()
        distance = CategoricalExpectedMSE()
        return cross_entropy(y_true, y_pred) + 10 * distance(y_true, y_pred)
