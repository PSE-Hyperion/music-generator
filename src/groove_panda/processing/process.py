import logging

import numpy as np

from groove_panda.config import Config
from groove_panda.processing.tokenization.tokenizer import Sixtuple, SixtupleTokenMaps

config = Config()
logger = logging.getLogger(__name__)


class NumericTuple:
    """
    The numerical representation of token tuples (instead of string tokens, integers are stored).
    """

    def __init__(self, *values: int):
        # The *values parameter makes sequential NumericTuple(1, 2, 3) and grouped NumericTuple(*[1, 2, 3]) possible.
        self._values = list(values)

    def __repr__(self):
        return f"NumericTuple({', '.join(str(v) for v in self.values)})"

    @property
    def values(self):
        return self._values


def numerize(sixtuples: list[Sixtuple], sixtuple_token_maps: SixtupleTokenMaps) -> list[NumericTuple]:
    """
    Turns the incoming list of sixtuples into a list of numeric tuples using the given sixtuple token maps, to translate
    tokens into integers.

    Sixtuple is a class of six features, encoded as strings, while NumericTuple is a class of features, as defined in
    configs (dynamic size), encoded as integers.
    """
    logger.info("Start numerize...")

    numeric_tuples = []

    feature_maps = sixtuple_token_maps.maps  # [(name, dict), ...]

    for sixtuple in sixtuples:
        numeric_values = []
        for name, mapping in feature_maps:
            # Use getattr to get the attribute dynamically from sixtuple. Features that aren't defined in configs, are
            # ignored.
            value = getattr(sixtuple, name)
            try:
                numeric_value = mapping[value]
            except KeyError as e:
                raise KeyError(f"Value {value} for feature '{name}' not found in {mapping}") from e
            numeric_values.append(numeric_value)

        numeric_tuple = NumericTuple(*numeric_values)
        numeric_tuples.append(numeric_tuple)

    logger.info("Finished numerize.")
    return numeric_tuples


def create_continuous_sequence(numeric_tuples: list[NumericTuple]) -> np.ndarray:
    """
    Return the entire sequence of numeric tuples as an array of integers of dim (n, m) for

    n = length of numeric tuples and

    m = amount of features in numeric sixtuple.
    """
    logger.info("Creating continuous sequence...")

    sequence = np.array(
        [event.values for event in numeric_tuples],
        dtype=np.int32,
    )

    logger.info("Finished creating continuous sequence")
    return sequence


def extract_subsequence(
    full_sequence: np.ndarray, sequence_length: int, stride: int = 1, start_idx: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a subsequence of given length from the full sequence
    """
    if len(full_sequence) < sequence_length + 1:
        raise ValueError(f"Sequence too short: {len(full_sequence)} < {sequence_length + 1}")

    if start_idx is None:
        # Calculate max start index considering stride
        max_start_steps = (len(full_sequence) - sequence_length) // stride
        start_step = np.random.randint(0, max_start_steps + 1)
        start_idx = start_step * stride

    x = full_sequence[start_idx : start_idx + sequence_length]
    y = full_sequence[start_idx + sequence_length]

    return x, y


def sequence_to_model_input(sequence: list[tuple[int, int, int, int, int, int]]) -> dict[str, np.ndarray]:
    """
    Convert a sequence of numeric sixtuples to model input format
    """

    # Convert to numpy array
    seq_array = np.array(sequence)

    # Create input dictionary for the model
    feature_names = [feature.name for feature in config.features]

    model_input = {}
    for i, feature_name in enumerate(feature_names):
        # Add batch dimension (1, sequence_length)
        model_input[f"input_{feature_name}"] = seq_array[:, i].reshape(1, -1)

    return model_input


def reshape_x(x):
    #   reshapes X training data to numpy array (matrix) of shape (num_sequences, SEQUENCE_LENGTH, 6)
    #   embedding layers expect integers, so we dont need to normalize
    #

    logger.info("Started reshaping...")

    x = np.array(x, dtype=np.int32)

    logger.info("Finished reshaping")
    return x


def denumerize(_numeric_sixtuples: list[NumericTuple], _sixtuple_token_maps: SixtupleTokenMaps) -> list[Sixtuple]:
    #   Turns list of embedded numeric events into list of embedded token events,
    #   by using the maps provided by the given tokenizer instance
    #
    #

    return []


def build_input_dict():  # Unsure about what this is, but afraid to delete - joao
    pass


# TODO
# def denumerize(numeric_sixtuples: list[NumericSixtuple], sixtuple_token_maps: SixtupleTokenMaps) -> list[Sixtuple]:
#    """ Turns list of embedded numeric events into list of embedded token events, by using the maps provided
#        by the given tokenizer instance """
#
#    return [] - this is done, no? - Joao
