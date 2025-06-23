# functionality to preprocess score objects and postprocess int lists
# functionality for turning integer lists into sequences and reshaping them to ndarray

import numpy as np

from config import SEQUENCE_LENGTH


def numerize(tokens : list[str], map : dict[str, int]) -> list[int]:
    return [map[token] for token in tokens]

def sequenize(nums : list[int]):
    print("Start sequenizing...", end="\r")
    X, y = [], []
    total = len(nums) - SEQUENCE_LENGTH
    for i in range(total):
        input_seq = nums[i:i + SEQUENCE_LENGTH]
        output_note = nums[i + SEQUENCE_LENGTH]
        X.append(input_seq)
        y.append(output_note)

    print("Finished sequenizing")
    return (X, y)

def reshape_X(X, num_featuers : int):
    X = np.array(X)
    X = np.reshape(X, (len(X), SEQUENCE_LENGTH, 1))
    X = X / float(num_featuers)
    return X


def denumerize():
    pass
