import datetime
import random
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


def generate_unique_name(name: str) -> str:
    """
    Takes a user given name and appends the current date on the right (for easy hierarchical date-based searching)
    and a random ID in the middle to reduce the chance of file name collisions.
    The user-given name comes first to not stand in the way of autocomplete.
    """

    date_str = datetime.datetime.now().strftime("%d_%m_%Y")

    random_id = random.randint(10000, 99999)

    return f"{name}_{random_id}_{date_str}"
