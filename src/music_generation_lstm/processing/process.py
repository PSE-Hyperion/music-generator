# functionality to preprocess score objects and postprocess int lists
# functionality for turning integer lists into sequences and reshaping them to ndarray

import numpy as np

from config import SEQUENCE_LENGTH
from tokenization.tokenizer import Tokenizer, TokenEvent

class NumerizedTokenEvent():
    def __init__(self, type : int, pitch : int, duration : int, delta_offset : int, velocity : int, instrument : int):
        self.type = type
        self.pitch = pitch
        self.duration = duration
        self.delta_offset = delta_offset
        self.velocity = velocity
        self.instrument = instrument

def numerize(token_events : list[TokenEvent], tokenizer : Tokenizer) -> list[NumerizedTokenEvent]:
    #   Turns a list of token events into its numeric representation
    #   Uses maps of the given tokenizer
    #

    print("Start numerize...")

    embedded_numeric_events = []
    for token_event in token_events:
        numerized_token_event = NumerizedTokenEvent(
            tokenizer.type_map[token_event.type],
            tokenizer.pitch_map[token_event.pitch],
            tokenizer.duration_map[token_event.duration],
            tokenizer.delta_offset_map[token_event.delta_offset],
            tokenizer.velocity_map[token_event.velocity],
            tokenizer.instrument_map[token_event.instrument],
        )
        embedded_numeric_events.append(numerized_token_event)

    print("Finished numerize.")

    return embedded_numeric_events

def sequenize(embedded_numeric_events: list[NumerizedTokenEvent]):
    #   creates sequences of feature tuples (extracts feature num val from NumerizedNumericEvent class) and corresponding targets event feature tuples
    #   uses sliding window of size of SEQUENCE_LENGTH
    #   inputs contains sequences of features of an event, targets contains the next features of an event
    #   Ex: inputs = [[1, 2], [2, 3], [3, 4]], targets = [3, 4, 5]
    #   sequence inputs[i] is followed by targets[i]

    print("Start preparing the training sequences...")

    inputs, targets = [], []


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
        target_event = embedded_numeric_events[i + SEQUENCE_LENGTH]
        target_tuple = (
            target_event.type,
            target_event.pitch,
            target_event.duration,
            target_event.delta_offset,
            target_event.velocity,
            target_event.instrument
        )

        inputs.append(input_seq)
        targets.append(target_tuple)

    print("Finished preparing the training sequences")
    return inputs, targets

def reshape_X(X):
    #   reshapes X training data to numpy array (matrix) of shape (num_sequences, SEQUENCE_LENGTH, 6)
    #   embedding layers expect integers, so we dont need to normalize
    #

    print("Started reshaping...")

    X = np.array(X, dtype=np.int32)

    print("Finished reshaping")
    return X


def denumerize(embedded_numeric_events : list[NumerizedTokenEvent], tokenizer : Tokenizer) -> list[TokenEvent]:
    #   Turns list of embedded numeric events into list of embedded token events, by using the maps provided by the given tokenizer instance
    #
    #

    return []
