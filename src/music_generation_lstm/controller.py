import inspect
import json
import os
import sys

import numpy as np

from music_generation_lstm.config import (
    DEFAULT_GENERATION_LENGTH,
    FEATURE_NAMES,
    INPUT_MIDI_DIR,
    RESULTS_MIDI_DIR,
    TEMPERATURE,
)
from music_generation_lstm.generation.generate import (
    generate_token,
    prepare_input_sequence,
    reshape_input_sequence,
    split_input_into_features,
)
from music_generation_lstm.midi.parser import parse_midi
from music_generation_lstm.models import models, train as tr
from music_generation_lstm.models.model_io import load_model, save_model
from music_generation_lstm.processing import parallel_processing, processed_io
from music_generation_lstm.processing.process import numerize
from music_generation_lstm.processing.tokenization import token_map_io
from music_generation_lstm.processing.tokenization.tokenizer import SixtupleTokenMaps, Tokenizer, detokenize


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

    save_model(model, processed_dataset_id)


def generate(
    model_name: str,
    input_name: str,
    output_name: str,
    generation_length=DEFAULT_GENERATION_LENGTH,
    temperature=TEMPERATURE,
):
    # Get model and its configuration
    base_model, config = load_model(model_name)
    model = base_model.model
    print(f"Loaded model {model_name}.")

    # Get input MIDI from the given name
    input_midi_path = os.path.join(INPUT_MIDI_DIR, input_name) + ".mid"
    input_stream = parse_midi(input_midi_path)

    print(f"Preparing sequence for {model_name}")

    # Prepare token maps for the model to pick from, obtained from the dataset it was trained on
    token_maps = token_map_io.load_token_maps(config["processed dataset name"])
    sixtuple_token_maps = SixtupleTokenMaps()

    sixtuple_token_maps.create_from_dicts(
        token_maps["bar_map"],
        token_maps["position_map"],
        token_maps["pitch_map"],
        token_maps["duration_map"],
        token_maps["velocity_map"],
        token_maps["tempo_map"],
    )

    # Tokenize the input
    input_tokenizer = Tokenizer()
    tokenized_input = input_tokenizer.tokenize(input_stream)

    # Create a valid input sequence (making sure that input is large enough, and padding if not)
    input_sequence = prepare_input_sequence(tokenized_input)

    # Convert the sequence into a numeric sequence for the LSTM model
    numeric_sequence = numerize(input_sequence, sixtuple_token_maps)

    # Prepare the sliding window (we will add an output token and remove the last input)
    numeric_window = numeric_sequence.copy()

    # Initialize an empty event list and start the generation loop
    event_list = []
    print("Generating music with AI...")
    for generation in range(generation_length):
        # Reshape the sequence into the correct format for the model -> (batch size: 1, SEQUENCE_LENGTH, features: 6)
        input_sequence_matrix = reshape_input_sequence(numeric_window)

        # Reshape the sequence into a dictionary with one item for each feature
        input_feature_dict = split_input_into_features(input_sequence_matrix)

        next_numeric_event = generate_token(model, input_feature_dict, temperature)

        # Slide the numeric window forward by one, to include the generated token
        numeric_window.pop(0)
        numeric_window.append(next_numeric_event)

        next_raw_event = sixtuple_token_maps.decode_numeric(next_numeric_event)
        event_list.append(next_raw_event)

    generated_stream = detokenize(event_list)  # There are two almost identical "detokenize" methods.

    # Save the generation to the specified path by the user
    output_path = os.path.join(RESULTS_MIDI_DIR, output_name + ".mid")
    generated_stream.write("midi", fp=output_path)
    print(f"Saved generated MIDI to {output_path}")


def show():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("show")


def exit():
    print("You've exited the program.")
