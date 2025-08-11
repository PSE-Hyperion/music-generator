from groove_panda.config import Config


def get_loss_weights() -> dict[str, float]:
    """
    Return normalized loss weights, so all weight together of each feature sums up to 1
    If config was not loaded or its not defined them, we fall back to the defaults.
    """
    config = Config()

    # get relative weights from the loaded config or if there are non the default ones
    loss_weights_relative = getattr(config, "loss_weights", None)
    if not loss_weights_relative:
        loss_weights_relative = dict(config.LOSS_WEIGHTS_DEFAULT_TPL)  # using default loss weights

    loss_weights_relative = {key: float(value) for key, value in loss_weights_relative.items()}

    total = sum(loss_weights_relative.values())
    if total <= 0:
        raise ValueError("Sum of loss_weights must be > 0")  # must be greater 0 or we could divide by 0 later,ohoh

    return {key: value / total for key, value in loss_weights_relative.items()}
