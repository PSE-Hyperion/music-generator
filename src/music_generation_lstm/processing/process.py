import logging

import numpy as np

from music_generation_lstm.config import SEQUENCE_LENGTH
from music_generation_lstm.processing.tokenization.tokenizer import Sixtuple, SixtupleTokenMaps

logger = logging.getLogger(__name__)


class NumericSixtuple:
    def __init__(self, bar: int, position: int, pitch: int, duration: int, velocity: int, tempo: int):
        self._bar = bar
        self._position = position
        self._pitch = pitch
        self._duration = duration
        self._velocity = velocity
        self._tempo = tempo

    @property
    def bar(self):
        return self._bar

    @property
    def position(self):
        return self._position

    @property
    def pitch(self):
        return self._pitch

    @property
    def duration(self):
        return self._duration

    @property
    def velocity(self):
        return self._velocity

    @property
    def tempo(self):
        return self._tempo


def numerize(sixtuples: list[Sixtuple], sixtuple_token_maps: SixtupleTokenMaps) -> list[NumericSixtuple]:
    #   Turns a list of embedded token events into its numeric equivalent
    #   Uses maps of the given tokenizer
    #

    logger.info("Start numerize...")

    bar_map = sixtuple_token_maps.bar_map
    position_map = sixtuple_token_maps.position_map
    pitch_map = sixtuple_token_maps.pitch_map
    duration_map = sixtuple_token_maps.duration_map
    velocity_map = sixtuple_token_maps.velocity_map
    tempo_map = sixtuple_token_maps.tempo_map

    numeric_sixtuples = []
    for sixtuple in sixtuples:
        numeric_sixtuple = NumericSixtuple(
            bar_map[sixtuple.bar],
            position_map[sixtuple.position],
            pitch_map[sixtuple.pitch],
            duration_map[sixtuple.duration],
            velocity_map[sixtuple.velocity],
            tempo_map[sixtuple.tempo],
        )
        numeric_sixtuples.append(numeric_sixtuple)

    logger.info("Finished numerize.")

    return numeric_sixtuples


def sequenize(numeric_sixtuples: list[NumericSixtuple]):
    """creates sequences of feature tuples (extracts feature num val from embeddednumericevent class)
    and corresponding next event feature tuples
    uses sliding window of size of SEQUENCE_LENGTH
    X contains sequences of features of an event, y contains the next features of an event
    X = [[1, 2], [2, 3], [3, 4]], y = [3, 4, 5]
    sequence X[i] is followed by y[i]
    """

    logger.info("Start sequenizing...")

    x, y = [], []

    if len(numeric_sixtuples) < SEQUENCE_LENGTH + 1:
        raise Exception("Skipped a score, since the song was shorter than the sequence length")

    for i in range(len(numeric_sixtuples) - SEQUENCE_LENGTH):
        input_seq = [
            (event.bar, event.position, event.pitch, event.duration, event.velocity, event.tempo)
            for event in numeric_sixtuples[i : i + SEQUENCE_LENGTH]
        ]
        output_event = numeric_sixtuples[i + SEQUENCE_LENGTH]
        output_tuple = (
            output_event.bar,
            output_event.position,
            output_event.pitch,
            output_event.duration,
            output_event.velocity,
            output_event.tempo,
        )

        x.append(input_seq)
        y.append(output_tuple)

    logger.info("Finished sequenizing")
    return x, y


def sequence_to_model_input(sequence: list[tuple[int, int, int, int, int, int]]) -> dict[str, np.ndarray]:
    """
    Convert a sequence of numeric sixtuples to model input format
    """

    # Convert to numpy array
    seq_array = np.array(sequence)

    # Create input dictionary for the model
    feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]

    model_input = {}
    for i, feature_name in enumerate(feature_names):
        # Add batch dimension (1, sequence_length)
        model_input[feature_name] = seq_array[:, i].reshape(1, -1)

    return model_input


def reshape_X(x):
    #   reshapes X training data to numpy array (matrix) of shape (num_sequences, SEQUENCE_LENGTH, 6)
    #   embedding layers expect integers, so we dont need to normalize
    #

    logger.info("Started reshaping...")

    x = np.array(x, dtype=np.int32)

    logger.info("Finished reshaping")
    return x


def denumerize(numeric_sixtuples: list[NumericSixtuple], sixtuple_token_maps: SixtupleTokenMaps) -> list[Sixtuple]:
    #   Turns list of embedded numeric events into list of embedded token events, by using the maps provided by the given tokenizer instance
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
#    return []
