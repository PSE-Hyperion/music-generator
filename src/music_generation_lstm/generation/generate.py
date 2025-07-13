# generate integer sequence using the given model and generation length

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.models import Model

from music_generation_lstm.config import FEATURE_NAMES, SEQUENCE_LENGTH
from music_generation_lstm.processing.process import NumericSixtuple
from music_generation_lstm.processing.tokenization.tokenizer import Sixtuple


def generate_int_sequence():
    pass


def prepare_input_sequence(tokenized_input: list[Sixtuple], sequence_length=SEQUENCE_LENGTH) -> list[Sixtuple]:
    """
    if len(tokenized_input) < sequence_length:
        padding_length = sequence_length - len(tokenized_input)
        # padding = pad_vector * padding_length -> TODO: Create a padding vector in case the input is small
    """
    input_sequence = tokenized_input[-sequence_length:]

    return input_sequence


def reshape_input_sequence(numeric_sequence: list[NumericSixtuple]) -> NDArray[np.int32]:
    """
    Receives a numeric sequence and turns it into a NumPy array with
    shape (1, SEQUENCE_LENGTH, NUMBER_OF_FEATURES) that is compatible
    as an input for our LSTM model.
    """
    # Initialize an empty list, this list will contain each event as a row
    sequence_feature_rows = []

    # Turn each event into a row with 6 numbers, one for each feature, and append it to the list
    for event in numeric_sequence:
        row = [event.bar, event.position, event.pitch, event.duration, event.velocity, event.tempo]
        sequence_feature_rows.append(row)

    # Turn the list of rows into a NumPy array, which our model needs
    sequence_feature_matrix = np.array(sequence_feature_rows, dtype=np.int32)
    print("2D matrix shape: ", sequence_feature_matrix.shape)

    # Our model expects a batch of sequences, instead of a lone sequence
    # For this, we will wrap our sequence matrix into a batch of size 1 (1 sequence)
    # Ex: [ [event1], [event2] ] -> [ [[event1], [event2]] ]
    # This new vector has one item for every sequence. Since we have 1 sequence, it has one item.
    seed_sequence_batch = np.expand_dims(sequence_feature_matrix, axis=0)
    print("Seed batch shape: ", seed_sequence_batch.shape)

    return seed_sequence_batch


def split_input_into_features(input_sequence_matrix: NDArray[np.int32]) -> dict[str, NDArray[np.int32]]:
    """
    Receives a matrix that contains a sequence of events. Shape: (1, SEQUENCE_LENGTH, NUMBER_OF_FEATURES)
    Creates and returns a dictionary that maps each feature name to a sequence of only that feature.
    This is because our model embeds each feature individually, before joining them and generating.

    Ex: [ [(pitch:) 1, (duration:) 2], [(pitch:) 0, (duration:) 5] ]
    ->
    {
    "pitch":        [1, 0]
    "duration":     [2, 5]
    }
    """
    input_feature_dict = {}

    for i, feature in enumerate(FEATURE_NAMES):
        feature_array = input_sequence_matrix[:, :, i]
        input_feature_dict[feature] = feature_array

    return input_feature_dict


def generate_token(
    model: Model, input_feature_dict: dict[str, NDArray[np.int32]], temperature=float
) -> NumericSixtuple:
    """
    CLEANUP REQUIRED AFTER THIS. I JUST WANT IT TO WORK!!!
    """
    raw_outputs = model.predict(input_feature_dict)

    output_dict = dict(zip(model.output_names, raw_outputs))

    next_ids = []

    for feature in FEATURE_NAMES:
        logits_2d = output_dict[f"{feature}_output"]  # Change to _out or _output depending on model architecture
        last_logits = logits_2d[0]  # shape (vocab_size,)

        # 1) Scale by temperature
        scaled_logits = last_logits / temperature

        # 2) Softmax as before
        exps = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exps / np.sum(exps)

        # 3) Sample
        choice = np.random.choice(len(probs), p=probs)
        next_ids.append(choice)

        """
        CLEANUP REQUIRED BEFORE THIS. I JUST WANT IT TO WORK!!!
        """

    next_numeric_event = NumericSixtuple(*next_ids)
    return next_numeric_event
