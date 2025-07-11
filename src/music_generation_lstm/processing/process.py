from collections.abc import Iterable

import numpy as np

from music_generation_lstm.config import SEQUENCE_LENGTH
from music_generation_lstm.processing.tokenization.tokenizer import HexTupleTokenMap
from music_generation_lstm.processing.tokenization.tokens import HexTuple


class NumericHexTuple:
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


def numerize(hex_tuples: Iterable[HexTuple], hex_tuple_token_maps: HexTupleTokenMap) -> list[NumericHexTuple]:
    #   Turns a list of embedded token events into it's numeric equivalent
    #   Uses maps of the given tokenizer
    #

    print("Start numerize...")

    bar_map = hex_tuple_token_maps.bar_map
    position_map = hex_tuple_token_maps.position_map
    pitch_map = hex_tuple_token_maps.pitch_map
    duration_map = hex_tuple_token_maps.duration_map
    velocity_map = hex_tuple_token_maps.velocity_map
    tempo_map = hex_tuple_token_maps.tempo_map

    numeric_hex_tuples = []
    for hex_tuple in hex_tuples:
        numeric_hex_tuple = NumericHexTuple(
            bar_map[hex_tuple.bar],
            position_map[hex_tuple.position],
            pitch_map[hex_tuple.pitch],
            duration_map[hex_tuple.duration],
            velocity_map[hex_tuple.velocity],
            tempo_map[hex_tuple.tempo],
        )
        numeric_hex_tuples.append(numeric_hex_tuple)

    print("Finished numerize.")

    return numeric_hex_tuples


def sequenize(numeric_hex_tuples: list[NumericHexTuple]):
    """ creates sequences of feature tuples (extracts feature num val from embeddednumericevent class)
        and corresponding next event feature tuples
        uses sliding window of size of SEQUENCE_LENGTH
        X contains sequences of features of an event, y contains the next features of an event
        X = [[1, 2], [2, 3], [3, 4]], y = [3, 4, 5]
        sequence X[i] is followed by y[i]
    """

    print("Start sequenizing...")

    X, y = [], []

    if len(numeric_hex_tuples) < SEQUENCE_LENGTH + 1:
        raise Exception("Skipped a score, since the song was shorter than the sequence length")

    for i in range(len(numeric_hex_tuples) - SEQUENCE_LENGTH):
        input_seq = [
            (event.bar, event.position, event.pitch, event.duration, event.velocity, event.tempo)
            for event in numeric_hex_tuples[i : i + SEQUENCE_LENGTH]
        ]
        output_event = numeric_hex_tuples[i + SEQUENCE_LENGTH]
        output_tuple = (
            output_event.bar,
            output_event.position,
            output_event.pitch,
            output_event.duration,
            output_event.velocity,
            output_event.tempo,
        )

        X.append(input_seq)
        y.append(output_tuple)

    print("Finished sequenizing")
    return X, y


def reshape_X(X):
    #   reshapes X training data to numpy array (matrix) of shape (num_sequences, SEQUENCE_LENGTH, 6)
    #   embedding layers expect integers, so we dont need to normalize
    #

    print("Started reshaping...")

    X = np.array(X, dtype=np.int32)

    print("Finished reshaping")
    return X

#TODO
#def denumerize(numeric_sixtuples: list[NumericSixtuple], sixtuple_token_maps: SixtupleTokenMaps) -> list[Sixtuple]:
#    """ Turns list of embedded numeric events into list of embedded token events, by using the maps provided
#        by the given tokenizer instance """
#
#    return []
