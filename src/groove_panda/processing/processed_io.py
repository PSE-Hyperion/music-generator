import json
import logging
import os
import shutil
from typing import Final

import numpy as np

from groove_panda import directories
from groove_panda.config import Config

config = Config()
logger = logging.getLogger(__name__)

JSON_METADATA_SHAPE: Final = "input_shape"
JSON_METADATA_MAP_ID: Final = "map_id"


def save_processed_data(processed_dataset_id: str, music_path: str, x, y) -> None:
    """Saves the tokenized dataset and metadata, X and y are numpy arrays, X is a sequence of integer inputs for the
    model"""
    music_file_name = os.path.splitext(os.path.basename(music_path))[0]
    target_folder_path = os.path.join(directories.PROCESSED_DATASET_DIR, processed_dataset_id, music_file_name)
    os.makedirs(target_folder_path, exist_ok=False)
    try:
        logger.info("Start saving processed dataset as %s...", processed_dataset_id)

        # Save .npz file inside subfolder
        np.savez_compressed(os.path.join(target_folder_path, music_file_name), x=x, y=y)

        metadata = {JSON_METADATA_SHAPE: f"{x.shape}", JSON_METADATA_MAP_ID: f"{processed_dataset_id}"}

        metadata_path = os.path.join(target_folder_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        shutil.rmtree(target_folder_path)
        raise Exception(f"Failed to save tokenized data: {e}") from e

    logger.info("Finished saving processed dataset as %s", processed_dataset_id)
    logger.info("Input Shape: %s", x.shape)


def save_continuous_data(processed_dataset_id: str, music_path: str, continuous_sequence) -> None:
    """
    Save continuous sequence data instead of pre-chunked sequences
    """
    music_file_name = os.path.splitext(os.path.basename(music_path))[0]
    target_folder_path = os.path.join(directories.PROCESSED_DATASET_DIR, processed_dataset_id, music_file_name)
    os.makedirs(target_folder_path, exist_ok=False)

    try:
        logger.info("Start saving continuous sequence as %s...", processed_dataset_id)

        # Save .npz file with continuous sequence
        np.savez_compressed(os.path.join(target_folder_path, music_file_name), continuous_sequence=continuous_sequence)

        metadata = {
            JSON_METADATA_SHAPE: f"{continuous_sequence.shape}",
            JSON_METADATA_MAP_ID: f"{processed_dataset_id}",
            "data_type": "continuous_sequence",
        }

        metadata_path = os.path.join(target_folder_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        shutil.rmtree(target_folder_path)
        raise Exception(f"Failed to save continuous sequence data: {e}") from e

    logger.info("Finished saving continuous sequence as %s", processed_dataset_id)
    logger.info("Sequence Shape: %s", continuous_sequence.shape)


# Loads tokenized dataset and associated metadata for a given processed_dataset_id.
def load_tokenized_data(processed_dataset_id: str) -> tuple[np.ndarray, np.ndarray, tuple, str]:
    logger.info("Enter tokenized data getter")

    target_folder_path = os.path.join(directories.PROCESSED_DATASET_DIR, processed_dataset_id)
    target_data_path = os.path.join(target_folder_path, processed_dataset_id + ".npz")
    target_metadata_path = os.path.join(target_folder_path, "metadata.json")

    if not os.path.exists(target_data_path):
        raise Exception(f"File not found. Searched for {target_data_path}")

    npz_data = np.load(target_data_path)
    x: np.ndarray = npz_data["x"]
    y: np.ndarray = npz_data["y"]

    if not os.path.exists(target_metadata_path):
        raise Exception("Metadata couldn't be found")

    with open(target_metadata_path) as f:
        metadata = json.load(f)
    data_input_shape: tuple = metadata[JSON_METADATA_SHAPE]
    data_map_id: str = metadata[JSON_METADATA_MAP_ID]

    logger.info("Tokenized data loaded")

    return x, y, data_input_shape, data_map_id


def delete_data(name: str) -> None:
    """Deletes the entire folder of a tokenized dataset."""
    data_dir = os.path.join(directories.PROCESSED_DATASET_DIR, name)
    if not os.path.exists(data_dir):
        logger.error("Deleting data %s failed", name)
        return
    shutil.rmtree(data_dir)


def get_all_data_str_list() -> list[str]:
    """Returns dataset IDs (folder names) inside PROCESSED_DIR excluding non-data-files: metadata, system files,..."""
    data_str_list = []
    os.makedirs(directories.PROCESSED_DATASET_DIR, exist_ok=True)
    for entry in os.listdir(directories.PROCESSED_DATASET_DIR):
        if entry not in {"metadata.json", ".gitkeep"}:
            logger.error("This will never happen")
            data_str_list.append(entry)

    return data_str_list


def does_data_exist(name: str) -> bool:
    data_folder_dir = os.path.join(directories.PROCESSED_DATASET_DIR, name)
    return os.path.exists(data_folder_dir)


def get_processed_file_paths(processed_dataset_id: str) -> list[str]:
    """
    Get all .npz file paths for a processed dataset.

    Args:
        processed_dataset_id: ID of the processed dataset

    Returns:
        List of absolute paths to .npz files
    """
    processed_dir = os.path.join(directories.PROCESSED_DATASET_DIR, processed_dataset_id)

    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed dataset directory not found: {processed_dir}")

    file_paths = []

    # Walk through all subdirectories to find .npz files
    for root, _dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".npz"):
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        raise FileNotFoundError(f"No .npz files found in processed dataset: {processed_dataset_id}")

    return sorted(file_paths)  # Sort for consistent ordering
