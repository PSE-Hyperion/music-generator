# functionality to preprocess score objects and postprocess int lists
# functionality for turning integer lists into sequences and reshaping them to ndarray

import numpy as np

from config import SEQUENCE_LENGTH
from tokenization.tokenizer import Tokenizer, EmbeddedTokenEvent

class EmbeddedNumericEvent():
    def __init__(self, type : int, pitch : int, duration : int, delta_offset : int, velocity : int, instrument : int):
        self.type = type
        self.pitch = pitch
        self.duration = duration
        self.delta_offset = delta_offset
        self.velocity = velocity
        self.instrument = instrument

def numerize(embedded_token_events : list[EmbeddedTokenEvent], tokenizer : Tokenizer) -> list[EmbeddedNumericEvent]:
    #
    #
    #

    print("Start numerize...")

    embedded_numeric_events = []
    for embedded_token_event in embedded_token_events:
        embedded_numeric_event = EmbeddedNumericEvent(
            tokenizer.type_map[embedded_token_event.type],
            tokenizer.pitch_map[embedded_token_event.pitch],
            tokenizer.duration_map[embedded_token_event.duration],
            tokenizer.delta_offset_map[embedded_token_event.delta_offset],
            tokenizer.velocity_map[embedded_token_event.velocity],
            tokenizer.instrument_map[embedded_token_event.instrument],
        )
        embedded_numeric_events.append(embedded_numeric_event)

    print("Finished numerize.")

    return embedded_numeric_events

def sequenize(embedded_numeric_events: list[EmbeddedNumericEvent]):
    #   creates sequences of feature tuples (extracts feature num val from embeddednumericevent class) and corresponding next event feature tuples
    #   uses sliding window of size of SEQUENCE_LENGTH
    #   X contains sequences of features of an event, y contains the next features of an event
    #   X = [[1, 2], [2, 3], [3, 4]], y = [3, 4, 5]
    #   sequence X[i] is followed by y[i]

    print("Start sequenizing...")

    X, y = [], []


    if len(embedded_numeric_events) < SEQUENCE_LENGTH + 1:
        raise Exception("Skipped a score, since the song was shorter than the sequence length")

    for i in range(len(embedded_numeric_events) - SEQUENCE_LENGTH):
        input_seq = [
            (
                event.type,
                event.pitch,
                event.duration,
                event.delta_offset,
                event.velocity,
                event.instrument
            )
            for event in embedded_numeric_events[i:i + SEQUENCE_LENGTH]
        ]
        output_event = embedded_numeric_events[i + SEQUENCE_LENGTH]
        output_tuple = (
            output_event.type,
            output_event.pitch,
            output_event.duration,
            output_event.delta_offset,
            output_event.velocity,
            output_event.instrument
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


def denumerize(embedded_numeric_events : list[int], tokenizer : Tokenizer) -> list[EmbeddedTokenEvent]:
    return []
