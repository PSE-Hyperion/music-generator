from groove_panda.config import Config


def get_loss_weights() -> dict[str, float]:
    """
    Return normalized loss weights, so all weight together of each feature sums up to 1
    If config was not loaded or its not defined them, we fall back to the defaults.
    """
    config = Config()

    loss_weights_relative = getattr(config, "loss_weights", None)  # Get weights from config or use defaults
    if not loss_weights_relative:
        loss_weights_relative = config.LOSS_WEIGHTS_DEFAULT.copy()

    loss_weights_relative = {str(key): float(value) for key, value in loss_weights_relative.items()}

    total = sum(loss_weights_relative.values())
    if total == 0:
        raise ValueError("Sum of loss_weights must not be 0")  # prevents divide by zero

    return {key: value / total for key, value in loss_weights_relative.items()}
