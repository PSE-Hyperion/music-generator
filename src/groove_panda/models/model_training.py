import json
import os

from groove_panda import directories
from groove_panda.config import Config
from groove_panda.models import models, train as tr
from groove_panda.models.flexible_sequence_generator import FlexibleSequenceGenerator
from groove_panda.models.model_io import get_model_path, load_model, save_model
from groove_panda.models.utils import generate_unique_name
from groove_panda.processing import processed_io
from groove_panda.processing.tokenization import token_map_io

config = Config()


def train_model(model_id: str, processed_dataset_id: str, preset_name: str) -> None:
    """
    Step 1: Get processed datasets .npz file paths via provided processed_dataset_id
    Step 2: Build LSTM model architecture
    Step 3: Train model using lazy loading for memory-efficient training
    Step 4: Save model weights as model_id
    """

    # Get file paths for all processed data files
    file_paths = processed_io.get_processed_file_paths(processed_dataset_id)

    # Load metadata for vocab sizes
    token_maps_dir = os.path.join(directories.TOKEN_MAPS_DIR, processed_dataset_id)
    with open(os.path.join(token_maps_dir, "metadata.json")) as f:
        metadata = json.load(f)

    vocab_sizes = {
        feature.name: metadata[token_map_io.TOTAL_UNIQUE_BLANK_TOKENS % feature.name] for feature in config.features
    }

    # Get preset config
    preset = config.model_presets[preset_name]
    sequence_length = preset["sequence_length"]

    input_shape = (sequence_length, len(config.features))

    model_path = get_model_path(model_id)

    # Keep training if model exists, otherwise create new model
    if os.path.exists(model_path):
        model = load_model(model_id)[0]  # Get the model, discard the config
    else:
        model = models.LSTMModel(generate_unique_name(model_id), input_shape)
        model.build(vocab_sizes=vocab_sizes, preset_name=preset_name)

    # Use flexible sequence generator instead of loading all data
    train_generator = FlexibleSequenceGenerator(
        file_paths=file_paths,
        sequence_length=sequence_length,
        stride=preset["stride"],
        batch_size=preset["batch_size"],
        shuffle=True,
    )

    tr.train_model_eager(model, train_generator)
    save_model(model, processed_dataset_id)
