import json
import os
import shutil
from typing import Final

import numpy as np

from music_generation_lstm.config import PROCESSED_DIR
from music_generation_lstm.processing.tokenization.tokenizer import Tokenizer

JSON_METADATA_SHAPE: Final = "input_shape"
JSON_METADATA_MAP_ID: Final = "map_id"


# Saves the tokenized dataset and metadata, X and y are numpy arrays, X is a sequence of integer inputs for the model
def save_processed_data(processed_dataset_id: str, music_path: str, X, y, tokenizer: Tokenizer):
    music_file_name = os.path.splitext(os.path.basename(music_path))[0]
    target_folder_path = os.path.join(PROCESSED_DIR, processed_dataset_id, music_file_name)
    os.makedirs(target_folder_path, exist_ok=False)
    try:
        print(f"Start saving processed dataset as {processed_dataset_id}...", end="\r")

        # Save .npz file inside subfolder
        np.savez_compressed(os.path.join(target_folder_path, music_file_name), X=X, y=y)

        metadata = {JSON_METADATA_SHAPE: f"{X.shape}", JSON_METADATA_MAP_ID: f"{tokenizer.processed_dataset_id}"}

        metadata_path = os.path.join(target_folder_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        shutil.rmtree(target_folder_path)
        raise Exception(f"Failed to save tokenized data: {e}")

    print(f"Finished saving processed dataset as {processed_dataset_id}.")
    print(f"Input Shape: {X.shape}")


# Loads tokenized dataset and associated metadata for a given processed_dataset_id.
def load_tokenized_data(processed_dataset_id: str):
    print("Enter tokenized data getter")

    target_folder_path = os.path.join(PROCESSED_DIR, processed_dataset_id)
    target_data_path = os.path.join(target_folder_path, processed_dataset_id + ".npz")
    target_metadata_path = os.path.join(target_folder_path, "metadata.json")

    if not os.path.exists(target_data_path):
        raise Exception(f"File not found. Searched for {target_data_path}")

    npz_data = np.load(target_data_path)
    X = npz_data["X"]
    y = npz_data["y"]

    if not os.path.exists(target_metadata_path):
        raise Exception("Metadata couldn't be found")

    with open(target_metadata_path) as f:
        config = json.load(f)
    data_input_shape = config[JSON_METADATA_SHAPE]
    data_map_id = config[JSON_METADATA_MAP_ID]

    print("Tokenized data loaded")

    return X, y, data_input_shape, data_map_id


# Deletes the entire folder of a tokenized dataset.
def delete_data(name: str):
    data_dir = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(data_dir):
        print(f"Deleting data {name} failed")
        return
    shutil.rmtree(data_dir)


# Returns dataset IDs (folder names) inside PROCESSED_DIR excluding non-data-files: metadata, system files,...
def get_all_data_str_list() -> list[str]:
    data_str_list = []
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for entry in os.listdir(PROCESSED_DIR):
        if not (entry == "metadata.json" or entry == ".gitkeep"):
            print("This will never happen")
            data_str_list.append(entry)

    return data_str_list


def does_data_exist(name: str) -> bool:
    data_folder_dir = os.path.join(PROCESSED_DIR, name)
    return os.path.exists(data_folder_dir)


def get_processed_file_paths(processed_dataset_id: str) -> list[str]:
    """
    Get all .npz file paths for a processed dataset.

    Args:
        processed_dataset_id: ID of the processed dataset

    Returns:
        List of absolute paths to .npz files
    """
    processed_dir = os.path.join(PROCESSED_DIR, processed_dataset_id)

    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed dataset directory not found: {processed_dir}")

    file_paths = []

    # Walk through all subdirectories to find .npz files
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".npz"):
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        raise FileNotFoundError(f"No .npz files found in processed dataset: {processed_dataset_id}")

    return sorted(file_paths)  # Sort for consistent ordering
