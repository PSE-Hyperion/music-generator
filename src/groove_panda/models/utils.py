from groove_panda.config import Config

config = Config()


def get_loss_weights() -> dict[str, float]:
    """
    Return normalized loss weights, so all weight together of each feature sums up to 1
    If config was not loaded or its not defined them, we fall back to the defaults.
    """

    loss_weights_relative = config.loss_weights.copy()  # Get weights from config

    total = sum(loss_weights_relative.values())

    if total == 0:
        raise ValueError("Sum of loss_weights must not be 0")  # prevents divide by zero

    return {key: value / total for key, value in loss_weights_relative.items()}
