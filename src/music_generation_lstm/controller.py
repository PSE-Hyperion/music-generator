import json
import os
import logging

import numpy as np

from music_generation_lstm.models import models, train as tr
from music_generation_lstm.models.model_io import load_model, save_model
from music_generation_lstm.processing import parallel_processing, processed_io
from music_generation_lstm.processing.tokenization import token_map_io
from music_generation_lstm.data_managment import delete_dataset_data, delete_result_data

logger = logging.getLogger(__name__)

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
        input_shape = data["X"].shape[1:]  # Remove batch dimension

    model = models.LSTMModel(model_id, input_shape)
    model.build(vocab_sizes=vocab_sizes)

    tr.train_model(model, file_paths)

    save_model(model)


def delete_dataset(dataset_id: str):
    """
    Deletes a dataset given trough its dataset_id, will delete in data-> midi-> datasets 
    deletes the empty dataset folder.
    """
    delete_dataset_data(dataset_id)


def delete_result(result_id: str):
    """
    Deletes a file given trough the result_id, will delete in data -> midi -> results
    """
    delete_result_data(result_id)



def generate(model_name: str, input_name: str, output_name: str):
    #   Get model
    model, config = load_model(model_name)

    #   Get input MIDI

    #   Retrieve start sequence from given MIDI
    #   Generate a new sequence from the start sequence
    #   Write the generation in a folder

    logger.info("Generating music with AI...")


def show():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    logger.info("show") 


def exit():
    logger.info("You've exited the program.")
