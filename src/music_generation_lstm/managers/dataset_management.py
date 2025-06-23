
import json
import os
import shutil

import numpy as np

from config import PROCESSED_DIR
from typing import Final

class DatasetManager():

    JSON_METADATA_SHAPE: Final = "shape"
    JSON_METADATA_MAP: Final = "map"

    @staticmethod
    def save_tokenized_data(id : str, X, y, input_shape, token_to_int): # tuple of tokenized data list and input shape and dict
        target_folder_path = os.path.join(PROCESSED_DIR, id)
        os.makedirs(target_folder_path, exist_ok=False)
        try:
            print(f"Start saving processed dataset as {id}...", end="\r")

            # Save .npz file inside subfolder
            np.savez_compressed(os.path.join(target_folder_path, id), X=X, y=y)

            # Write shared metadata once
            metadata = {
                DatasetManager.JSON_METADATA_SHAPE: input_shape,
                DatasetManager.JSON_METADATA_MAP: token_to_int
            }

            metadata_path = os.path.join(target_folder_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            shutil.rmtree(target_folder_path)
            raise Exception(f"Failed to save tokenized data: {e}")

        print(f"Finished saving processed dataset as {id}.")
        print(f"Input Shape: {input_shape}\nDictionary size: {len(token_to_int)}")

    @staticmethod
    def load_tokenized_data(id: str):
        print("Enter tokenized data getter")
        target_folder_path = os.path.join(PROCESSED_DIR, id)
        target_data_path = os.path.join(target_folder_path, id + ".npz")
        target_metadata_path = os.path.join(target_folder_path, "metadata.json")
        print(1)
        if not os.path.exists(target_data_path):
            raise Exception(f"File not found. Searched for {target_data_path}")
        print(2)
        # Load X, y, num_classes
        npz_data = np.load(target_data_path)
        X = npz_data["X"]
        y = npz_data["y"]
        print(3)
        if not os.path.exists(target_metadata_path):
            raise Exception("Metadata couldn't be found")
        print(4)

        with open(target_metadata_path) as f:
            config = json.load(f)
        data_input_shape = config[DatasetManager.JSON_METADATA_SHAPE]
        data_note_to_int = config[DatasetManager.JSON_METADATA_MAP]


        print("Tokenized data loaded")

        return X, y, data_input_shape, data_note_to_int

    @staticmethod
    def delete_data(name : str):
        data_dir = os.path.join(PROCESSED_DIR, name)
        if not os.path.exists(data_dir):
            print(f"Deleting data {name} failed")
            return
        shutil.rmtree(data_dir)

    @staticmethod
    def get_all_data_str_list() -> list[str]:
        data_str_list = []
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        for entry in os.listdir(PROCESSED_DIR):
            if not (entry == "metadata.json" or entry == ".gitkeep"):
                print("This will never happen")
                data_str_list.append(entry)

        return data_str_list

    @staticmethod
    def does_data_exist(name : str) -> bool:
        data_folder_dir = os.path.join(PROCESSED_DIR, name)
        return os.path.exists(data_folder_dir)

