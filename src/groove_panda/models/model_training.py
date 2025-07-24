import json
import os

from groove_panda.config import MODEL_PRESETS
from groove_panda.models import models, train as tr
from groove_panda.models.flexible_sequence_generator import FlexibleSequenceGenerator
from groove_panda.models.model_io import save_model
from groove_panda.processing import processed_io
from groove_panda.processing.tokenization import token_map_io


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

    # Get preset config
    preset = MODEL_PRESETS[preset_name]
    sequence_length = preset["sequence_length"]

    model = models.LSTMModel(model_id, (sequence_length, 6))
    model.build(vocab_sizes=vocab_sizes, preset_name=preset_name)

    # Use flexible sequence generator instead of loading all data
    train_generator = FlexibleSequenceGenerator(
        file_paths=file_paths,
        sequence_length=sequence_length,
        stride=preset["stride"],
        batch_size=preset.get("batch_size", 32),
        shuffle=True,
    )

    tr.train_model_eager(model, train_generator)
    save_model(model, processed_dataset_id)
