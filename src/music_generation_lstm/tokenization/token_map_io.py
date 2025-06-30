
import os
import json
from config import TOKEN_MAPS_DIR, SEQUENCE_LENGTH

from tokenization import tokenizer

# saves all the tokenmaps into a file
# you can find the file in TOKEN_MAPS_DIR plus id
def save_token_maps(tokenizer: tokenizer):
    print("Start saving maps...")

    total_unique_tokens = len(tokenizer.type_map) + len(tokenizer.pitch_map)+len(tokenizer.duration_map)+len(tokenizer.delta_offset_map)+len(tokenizer.velocity_map)+len(tokenizer.instrument_map)

    print(f"Total unique tokens: {total_unique_tokens}")

    # save important information in tokenizer just in case, could also be saved in data
    tokenizer.sequence_length = SEQUENCE_LENGTH
    tokenizer.num_features_type = len(tokenizer.type_map)
    tokenizer.num_features_pitch = len(tokenizer.pitch_map)
    tokenizer.num_features_duration = len(tokenizer.duration_map)
    tokenizer.num_features_delta_offset = len(tokenizer.delta_offset_map)
    tokenizer.num_features_velocity = len(tokenizer.velocity_map)
    tokenizer.num_features_instrument = len(tokenizer.instrument_map)


    folder_path = os.path.join(TOKEN_MAPS_DIR, tokenizer.processed_dataset_id)
    os.makedirs(folder_path, exist_ok=False)
    with open(os.path.join(folder_path, "type_map.json"), "w") as f:
        json.dump(tokenizer.type_map, f, indent=4)
    with open(os.path.join(folder_path, "pitch_map.json"), "w") as f:
        json.dump(tokenizer.pitch_map, f, indent=4)
    with open(os.path.join(folder_path, "duration_map.json"), "w") as f:
        json.dump(tokenizer.duration_map, f, indent=4)
    with open(os.path.join(folder_path, "delta_offset_map.json"), "w") as f:
        json.dump(tokenizer.delta_offset_map, f, indent=4)
    with open(os.path.join(folder_path, "velocity_map.json"), "w") as f:
        json.dump(tokenizer.velocity_map, f, indent=4)
    with open(os.path.join(folder_path, "instrument_map.json"), "w") as f:
        json.dump(tokenizer.instrument_map, f, indent=4)

    print("Finished saving maps")

def load_token_maps():
    pass
