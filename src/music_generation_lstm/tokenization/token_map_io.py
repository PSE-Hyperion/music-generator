
import os
import json

from config import TOKEN_MAPS_DIR
from tokenization.tokenizer import SixtupleTokenMaps
from typing import Final

TOTAL_UNIQUE_BAR_TOKENS : Final = "total_unique_bar_tokens"
TOTAL_UNIQUE_POSITION_TOKENS : Final = "total_unique_position_tokens"
TOTAL_UNIQUE_PITCH_TOKENS : Final = "total_unique_pitch_tokens"
TOTAL_UNIQUE_DURATION_TOKENS : Final = "total_unique_duration_tokens"
TOTAL_UNIQUE_VELOCITY_TOKENS : Final = "total_unique_velocity_tokens"
TOTAL_UNIQUE_TEMPO_TOKENS : Final = "total_unique_tempo_tokens"
TOTAL_UNIQUE_TOKENS : Final = "total_unique_tokens"



def save_token_maps(processed_dataset_id : str, token_maps : SixtupleTokenMaps):
    #   saves all the tokenmaps into a file
    #   you can find the file in TOKEN_MAPS_DIR plus id
    #

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
        TOTAL_UNIQUE_TOKENS: total_unique_tokens
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

def load_token_maps():
    pass
