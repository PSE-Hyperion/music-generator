from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def concat_arrays(input_data: Iterable[NDArray]) -> tuple[NDArray, NDArray, NDArray]:
    arrays: Sequence = list(input_data)

    columns = arrays[0].shape[1]
    assert all(a.shape[1] == columns for a in arrays), "Invalid array formats"

    lengths = [a.shape[0] for a in arrays]
    starts = np.cumsum([0] + lengths[:-1])  # noqa: RUF005
    ends = starts + lengths
    concatenated_arrays = np.concatenate(arrays, axis=0)

    return concatenated_arrays, starts, ends

def items_to_array_rowwise(input_data: Iterable[Any]):
    """
    Creates a (possibly multi-dimensional) numpy array from an iterable of items.
    Each row represents an item. It's representation is determined by the __array__ method of the item.
    Attention: The shape of all item arrays has to be identical.
    """
    array =  np.array([np.asarray(item) for item in input_data])
    print(array.dtype, array.shape)
    return array
