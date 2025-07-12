import json
import os

import numpy as np

from music_generation_lstm.config import DEFAULT_GENERATION_LENGTH, INPUT_MIDI_DIR
from music_generation_lstm.generation.generate import generate_input_sequence
from music_generation_lstm.midi.parser import parse_midi
from music_generation_lstm.models import models, train as tr
from music_generation_lstm.models.model_io import load_model, save_model
from music_generation_lstm.processing import parallel_processing, processed_io
from music_generation_lstm.processing.tokenization import token_map_io
from music_generation_lstm.processing.tokenization.tokenizer import Tokenizer


def process(dataset_id: str, processed_dataset_id: str):
    #   parses midi file(s) to music21.stream.Score
    #   tokenize score(s)
    #   numerize tokens
    #   save processed data (ready for training data)

    parallel_processing.parallel_process(dataset_id, processed_dataset_id)


def train(model_id: str, processed_dataset_id: str):
    """
    Step 1:   Get processed datasets .npz file paths via provided processed_dataset_id

    Step 2:   Build LSTM model architecture

    Step 3:   Train model using lazy loading for memory-efficient training on large datasets (also plots training data)

    Step 4:   Save model weights as model_id
    """

    # Get file paths for all processed data files
    file_paths = processed_io.get_processed_file_paths(processed_dataset_id)

    # Load metadata for vocab sizes
    # bad shouldn't be in here

    token_maps_dir = os.path.join("data/token_maps", processed_dataset_id)
    with open(os.path.join(token_maps_dir, "metadata.json")) as f:
        metadata = json.load(f)

    vocab_sizes = {
        "bar": metadata[token_map_io.TOTAL_UNIQUE_BAR_TOKENS],
        "position": metadata[token_map_io.TOTAL_UNIQUE_POSITION_TOKENS],
        "pitch": metadata[token_map_io.TOTAL_UNIQUE_PITCH_TOKENS],
        "duration": metadata[token_map_io.TOTAL_UNIQUE_DURATION_TOKENS],
        "velocity": metadata[token_map_io.TOTAL_UNIQUE_VELOCITY_TOKENS],
        "tempo": metadata[token_map_io.TOTAL_UNIQUE_TEMPO_TOKENS],
    }

    # Get input shape from first file
    with np.load(file_paths[0]) as data:
        input_shape = data["X"].shape[1:]  # Remove batch dimension

    model = models.LSTMModel(model_id, input_shape)
    model.build(vocab_sizes=vocab_sizes)

    tr.train_model(model, file_paths)

    save_model(model)


def generate(model_name: str, input_name: str, output_name: str, generation_length=DEFAULT_GENERATION_LENGTH):
    # Get model
    model, config = load_model(model_name)

    print(f"Loaded model {model_name}.")

    # Get input MIDI from the given name
    input_midi_path = os.path.join(INPUT_MIDI_DIR, input_name) + ".mid"
    input_midi = parse_midi(input_midi_path)

    print(f"Preparing sequence for {model_name}")

    # Tokenize the input
    input_tokenizer = Tokenizer()
    tokenized_input = input_tokenizer.tokenize(input_midi)

    # Create a valid input sequence (making sure that input is large enough, and padding if not)
    input_sequence = generate_input_sequence(tokenized_input)

    """
    I will check if the file exists here with the os.join blabla
    and then I will create the correct directory, and then use the parse_midi
    method from parser to give me the score, which I will then process and so
    on and so forth. Will be an adventure.
    """
    #   Retrieve start sequence from given MIDI
    #   Generate a new sequence from the start sequence
    #   Write the generation in a folder

    print("Generating music with AI...")


def show():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("show")


def exit():
    print("You've exited the program.")
