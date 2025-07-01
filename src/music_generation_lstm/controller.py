import models.models as models
import numpy as np

from processing import process as p
from midi import parser, writer
from tokenization.tokenizer import Tokenizer

from models import model_io
from models import train as t
from processing import processed_io
from tokenization import token_map_io

def process(dataset_id: str, processed_dataset_id: str):
    #   parses midi file(s) to music21.stream.Score
    #   tokenize score(s)
    #   numerize tokens
    #   save processed data (ready for training data)


    midi_paths = parser.get_midi_paths_from_dataset(dataset_id)
    tokenizer = Tokenizer(processed_dataset_id)

    total = len(midi_paths)

    for index, midi_path in enumerate(midi_paths, start=1):    # seperate load and processing, share vocab data in shared tokenizer object
        print(f"[PROGRESS] Processing ({index}/{total})")
        score = parser.parse_midi(midi_path)

        sixtuples = tokenizer.tokenize(score)   # might be handled now

        embedded_numeric_events = p.numerize(sixtuples, tokenizer.sixtuple_token_maps)   # might be handled now
        X, y = p.sequenize(embedded_numeric_events)   # might be handled now
        X = p.reshape_X(X)

        processed_io.save_processed_data(processed_dataset_id, midi_path, X, y, tokenizer) # might be handled now
    token_map_io.save_token_maps(processed_dataset_id, tokenizer.sixtuple_token_maps)


def train_b(model_id: str, processed_dataset_id: str):                 # TRAIN DOESNT WORK NOW, SINCE THE PROCESSED DATA IS SAVED DIFFERENTLY
    #   get processed via id
    #   build model
    #   train model
    #   save model


    X, y, input_shape, map_id = processed_io.load_tokenized_data(processed_dataset_id)

    # uh uh, stinky: update asap
    import os
    import json
    token_maps_dir = os.path.join("data/token_maps", map_id)
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

    model = models.LSTMModel(model_id, input_shape)
    model.build(vocab_sizes=vocab_sizes)

    # Also calls plot method
    t.train_model(model, X, y)

    # save model


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
    import os
    import json
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

    # Get input shape from first file
    with np.load(file_paths[0]) as data:
        input_shape = data['X'].shape[1:]  # Remove batch dimension


    model = models.LSTMModel(model_id, input_shape)
    model.build(vocab_sizes=vocab_sizes)

    t.train_model(model, file_paths, vocab_sizes)

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
