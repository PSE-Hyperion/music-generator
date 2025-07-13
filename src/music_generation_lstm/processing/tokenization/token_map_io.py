import json
import os
from typing import Final

from music_generation_lstm.config import TOKEN_MAPS_DIR
from music_generation_lstm.processing.tokenization.tokenizer import SixtupleTokenMaps

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

    print("Start saving maps...")

    total_unique_tokens = token_maps.total_size

    print(f"Total unique tokens: {total_unique_tokens}")

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

    print("Finished saving maps")


def load_token_maps(processed_dataset_name: str) -> dict[str, dict[str, int]]:
    """
    Receives the name of a dataset that has been processed.
    Retreives the token maps corresponding to that dataset.
    Creates and returns a dictionary with one item for each map (i.e. pitch, velocity etc.)
    """
    # Set base path
    processed_dataset_maps_path = os.path.join(TOKEN_MAPS_DIR, processed_dataset_name)

    # Retreive all maps
    with open(os.path.join(processed_dataset_maps_path, "bar_map.json"), "r") as map:
        bar_map = json.load(map)
    with open(os.path.join(processed_dataset_maps_path, "position_map.json"), "r") as map:
        position_map = json.load(map)
    with open(os.path.join(processed_dataset_maps_path, "pitch_map.json"), "r") as map:
        pitch_map = json.load(map)
    with open(os.path.join(processed_dataset_maps_path, "duration_map.json"), "r") as map:
        duration_map = json.load(map)
    with open(os.path.join(processed_dataset_maps_path, "velocity_map.json"), "r") as map:
        velocity_map = json.load(map)
    with open(os.path.join(processed_dataset_maps_path, "tempo_map.json"), "r") as map:
        tempo_map = json.load(map)

    # Return dictionary of maps
    return {
        "bar_map": bar_map,
        "position_map": position_map,
        "pitch_map": pitch_map,
        "duration_map": duration_map,
        "velocity_map": velocity_map,
        "tempo_map": tempo_map,
    }
