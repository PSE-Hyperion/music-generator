import json
import os

import numpy as np

from music_generation_lstm.config import ALLOWED_MUSIC_FILE_EXTENSIONS, GENERATION_TEMPERATURE
from music_generation_lstm.generation.generate import MusicGenerator, load_seed_sequence_from_midi
from music_generation_lstm.midi import writer
from music_generation_lstm.models import models, train as tr
from music_generation_lstm.models.model_io import load_model, save_model
from music_generation_lstm.processing import parallel_processing, processed_io
from music_generation_lstm.processing.tokenization import token_map_io


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

    # Save model with dataset ID for generation
    save_model(model, processed_dataset_id)


def generate(model_name: str, input_name: str, output_name: str):
    """
    Generate music using a trained model
    """

    print(f"Starting music generation with model: {model_name}")

    # Load the trained model
    model, config = load_model(model_name)

    # Get the processed dataset ID from the model config
    processed_dataset_id = config.get("processed_dataset_id")
    if not processed_dataset_id:
        raise ValueError(
            "The loaded model config does not contain a 'processed_dataset_id'. "
            "Please ensure the model was saved with the correct dataset reference."
        )

    # Load seed sequence from input MIDI file
    input_midi_path = None
    for ext in ALLOWED_MUSIC_FILE_EXTENSIONS:
        candidate = os.path.join("data/midi/input", f"{input_name}{ext}")
        if os.path.exists(candidate):
            input_midi_path = candidate
            break
    if input_midi_path is None:
        raise FileNotFoundError(f"Input MIDI file not found: {input_name}")

    print(f"Loading seed sequence from: {input_midi_path}")
    seed_sequence = load_seed_sequence_from_midi(input_midi_path, processed_dataset_id)

    generator = MusicGenerator(model.model, processed_dataset_id, GENERATION_TEMPERATURE)

    # Generate music stream
    generated_stream = generator.generate_music_stream(seed_sequence)

    # Save the generated music
    writer.write_midi(output_name, generated_stream)

    print(f"Music generation completed! Output saved as: {output_name}")


def show():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("show")


def exit():
    print("You've exited the program.")
