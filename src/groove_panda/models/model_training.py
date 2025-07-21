import json
import os
import numpy as np
from groove_panda.processing import processed_io
from groove_panda.processing.tokenization import token_map_io
from groove_panda.models import models, train as tr
from groove_panda.models.model_io import save_model

def train_model(model_id: str, processed_dataset_id: str, preset_name: str):
    """
    Step 1: Get processed datasets .npz file paths via provided processed_dataset_id
    Step 2: Build LSTM model architecture
    Step 3: Train model using lazy loading for memory-efficient training
    Step 4: Save model weights as model_id
    """

    # Get file paths for all processed data files
    file_paths = processed_io.get_processed_file_paths(processed_dataset_id)

    # Load metadata for vocab sizes
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
        input_shape = data["x"].shape[1:]

    model = models.LSTMModel(model_id, input_shape)
    model.build(vocab_sizes=vocab_sizes, preset_name=preset_name)

    tr.train_model_eager(model, file_paths)
    save_model(model, processed_dataset_id)
