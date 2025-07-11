from collections.abc import Iterable

import numpy as np
import tensorflow as tf

from music_generation_lstm.processing.tokenization import tokens


def tokens_to_arrays(token_sequences: Iterable[Iterable[tokens.BaseToken]]):
    for sequence in token_sequences:
        pass
    array = np.array(next(iter(token_sequences)))
    print(array)
    tf.data.Dataset.c

def to_tf_dataset(array):
    pass
