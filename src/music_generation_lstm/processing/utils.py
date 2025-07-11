from collections.abc import Iterable
from typing import Any

import numpy as np


def items_to_array_rowwise(input: Iterable[Any]):
    """
    Creates a (possibly multi-dimensional) numpy array from an iterable of items.
    Each row represents an item. It's representation is determined by the __array__ method of the item.
    Attention: The shape of all item arrays has to be identical.
    """
    return np.array(input)

def concat_arrays(input: list[np.Array])
