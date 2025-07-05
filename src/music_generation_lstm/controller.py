import json
import os

from music_generation_lstm.midi import parser
from music_generation_lstm.models import model_io, models, train as tr
from music_generation_lstm.processing import process as proc, processed_io
from music_generation_lstm.processing.tokenization import token_map_io
from music_generation_lstm.processing.tokenization.tokenizer import Tokenizer


def process(dataset_id: str, processed_dataset_id: str):
    #   parses midi file(s) to music21.stream.Score
    #   tokenize score(s)
    #   numerize tokens
    #   save processed data (ready for training data)

    midi_paths = parser.get_midi_paths_from_dataset(dataset_id)
    tokenizer = Tokenizer(processed_dataset_id)

    total = len(midi_paths)

    for index, midi_path in enumerate(
        midi_paths, start=1
    ):  # seperate load and processing, share vocab data in shared tokenizer object
        print(f"[PROGRESS] Processing ({index}/{total})")
        score = parser.parse_midi(midi_path)

        sixtuples = tokenizer.tokenize(score)

        numeric_sixtuples = proc.numerize(sixtuples, tokenizer.sixtuple_token_maps)
        X, y = proc.sequenize(numeric_sixtuples)
        X = proc.reshape_X(X)

        processed_io.save_processed_data(processed_dataset_id, midi_path, X, y, tokenizer)
    token_map_io.save_token_maps(processed_dataset_id, tokenizer.sixtuple_token_maps)


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
    with open(os.path.join(token_maps_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    vocab_sizes = {
        "bar": metadata[token_map_io.TOTAL_UNIQUE_BAR_TOKENS],
        "position": metadata[token_map_io.TOTAL_UNIQUE_POSITION_TOKENS],
        "pitch": metadata[token_map_io.TOTAL_UNIQUE_PITCH_TOKENS],
        "duration": metadata[token_map_io.TOTAL_UNIQUE_DURATION_TOKENS],
        "velocity": metadata[token_map_io.TOTAL_UNIQUE_VELOCITY_TOKENS],
        "tempo": metadata[token_map_io.TOTAL_UNIQUE_TEMPO_TOKENS],
    }

    import numpy as np  # bad, this shouldn't be in here

    # Get input shape from first file
    with np.load(file_paths[0]) as data:
        input_shape = data["X"].shape[1:]  # Remove batch dimension

    model = models.LSTMModel(model_id, input_shape)
    model.build(vocab_sizes=vocab_sizes)

    tr.train_model(model, file_paths)

    model_io.save_model(model)


def generate():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("generate")


def show():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("show")


def exit():
    print("You've exited the program.")
