import json
import logging
import os
from typing import Final

from groove_panda import directories
from groove_panda.config import Config
from groove_panda.processing.tokenization.tokenizer import SixtupleTokenMaps

config = Config()
logger = logging.getLogger(__name__)

TOTAL_UNIQUE_BLANK_TOKENS: Final = "total_unique_%s_tokens"
TOTAL_UNIQUE_TOKENS: Final = "total_unique_tokens"
MAP: Final = "_map"


def save_token_maps(processed_dataset_id: str, token_maps: SixtupleTokenMaps):
    """
    Saves all the token maps into a file (with metadata).

    You can find the file in config.token_maps_dir plus id
    """

    logger.info("Start saving maps...")

    folder_path = os.path.join(directories.token_maps_dir, processed_dataset_id)
    os.makedirs(folder_path, exist_ok=False)

    _save_token_maps_metadata(folder_path, token_maps)

    # Saves individual maps as readable json (could be made to pkl later)
    for name, d in token_maps.maps:
        file_path = os.path.join(folder_path, f"{name}{MAP}.json")
        with open(file_path, "w") as f:
            json.dump(d, f, indent=4)

    logger.info("Finished saving maps")


def _save_token_maps_metadata(folder_path: str, token_maps: SixtupleTokenMaps):
    total_unique_tokens = token_maps.total_size
    logger.info("Total unique tokens: %s", total_unique_tokens)
    metadata = {TOTAL_UNIQUE_BLANK_TOKENS % name: size for name, size in token_maps.map_sizes}
    metadata[TOTAL_UNIQUE_TOKENS] = total_unique_tokens
    with open(os.path.join(folder_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Saves metadata of all maps
    with open(os.path.join(folder_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def load_token_maps(processed_dataset_id: str) -> tuple[dict, dict, dict]:
    """
    Load token maps, metadata and reverse mapping for a processed dataset
    """
    token_maps_dir = os.path.join(directories.token_maps_dir, processed_dataset_id)

    # Load metadata
    metadata_path = os.path.join(token_maps_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Token maps metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load token maps
    token_maps = {}
    map_files = [
        ("bar", "bar_map.json"),
        ("position", "position_map.json"),
        ("pitch", "pitch_map.json"),
        ("duration", "duration_map.json"),
        ("velocity", "velocity_map.json"),
        ("tempo", "tempo_map.json"),
    ]

    for feature_name, filename in map_files:
        map_path = os.path.join(token_maps_dir, filename)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Token map not found: {map_path}")
        with open(map_path) as f:
            token_maps[feature_name] = json.load(f)

    # Create reverse mappings
    reverse_mappings = {}
    for feature_name, token_map in token_maps.items():
        reverse_mappings[feature_name] = {v: k for k, v in token_map.items()}

    return token_maps, metadata, reverse_mappings
