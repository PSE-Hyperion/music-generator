import json
import logging
import os
from typing import Final

from music_generation_lstm.config import TOKEN_MAPS_DIR
from music_generation_lstm.processing.tokenization.tokenizer import SixtupleTokenMaps

logger = logging.getLogger(__name__)

TOTAL_UNIQUE_BAR_TOKENS: Final = "total_unique_bar_tokens"
TOTAL_UNIQUE_POSITION_TOKENS: Final = "total_unique_position_tokens"
TOTAL_UNIQUE_PITCH_TOKENS: Final = "total_unique_pitch_tokens"
TOTAL_UNIQUE_DURATION_TOKENS: Final = "total_unique_duration_tokens"
TOTAL_UNIQUE_VELOCITY_TOKENS: Final = "total_unique_velocity_tokens"
TOTAL_UNIQUE_TEMPO_TOKENS: Final = "total_unique_tempo_tokens"
TOTAL_UNIQUE_TOKENS: Final = "total_unique_tokens"


def save_token_maps(processed_dataset_id: str, token_maps: SixtupleTokenMaps):
    """
    Saves all the token maps into a file (with metadata).

    You can find the file in TOKEN_MAPS_DIR plus id
    """

    logger.info("Start saving maps...")

    total_unique_tokens = token_maps.total_size

    logger.info("Total unique tokens: %s", total_unique_tokens)

    folder_path = os.path.join(TOKEN_MAPS_DIR, processed_dataset_id)
    os.makedirs(folder_path, exist_ok=False)

    # save important information with maps just in case
    metadata = {
        TOTAL_UNIQUE_BAR_TOKENS: token_maps.bar_map_size,
        TOTAL_UNIQUE_POSITION_TOKENS: token_maps.position_map_size,
        TOTAL_UNIQUE_PITCH_TOKENS: token_maps.pitch_map_size,
        TOTAL_UNIQUE_DURATION_TOKENS: token_maps.duration_map_size,
        TOTAL_UNIQUE_VELOCITY_TOKENS: token_maps.velocity_map_size,
        TOTAL_UNIQUE_TEMPO_TOKENS: token_maps.tempo_map_size,
        TOTAL_UNIQUE_TOKENS: total_unique_tokens,
    }

    # Saves metadata of all maps
    with open(os.path.join(folder_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Saves individual maps as readable json (could be made to pkl later)
    with open(os.path.join(folder_path, "bar_map.json"), "w") as f:
        json.dump(token_maps.bar_map, f, indent=4)
    with open(os.path.join(folder_path, "position_map.json"), "w") as f:
        json.dump(token_maps.position_map, f, indent=4)
    with open(os.path.join(folder_path, "pitch_map.json"), "w") as f:
        json.dump(token_maps.pitch_map, f, indent=4)
    with open(os.path.join(folder_path, "duration_map.json"), "w") as f:
        json.dump(token_maps.duration_map, f, indent=4)
    with open(os.path.join(folder_path, "velocity_map.json"), "w") as f:
        json.dump(token_maps.velocity_map, f, indent=4)
    with open(os.path.join(folder_path, "tempo_map.json"), "w") as f:
        json.dump(token_maps.tempo_map, f, indent=4)

    logger.info("Finished saving maps")


def load_token_maps(processed_dataset_id: str) -> tuple[dict, dict, dict]:
    """
    Load token maps, metadata and reverse mapping for a processed dataset
    """
    token_maps_dir = os.path.join(TOKEN_MAPS_DIR, processed_dataset_id)

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
