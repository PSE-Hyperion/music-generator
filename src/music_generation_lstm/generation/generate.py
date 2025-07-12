# generate integer sequence using the given model and generation length

from music_generation_lstm.config import SEQUENCE_LENGTH
from music_generation_lstm.processing.tokenization.tokenizer import Sixtuple


def generate_int_sequence():
    pass


def generate_input_sequence(tokenized_input: list[Sixtuple], sequence_length=SEQUENCE_LENGTH) -> list[Sixtuple]:
    if len(tokenized_input) < sequence_length:
        padding_length = sequence_length - len(tokenized_input)
        # padding = pad_vector * padding_length -> TODO: Create a padding vector in case the input is small

    input_sequence = tokenized_input[-sequence_length:]

    return input_sequence
