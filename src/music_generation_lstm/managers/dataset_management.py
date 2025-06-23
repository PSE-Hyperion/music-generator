
import json
import os
import shutil

import numpy as np

from config import PROCESSED_DIR

class DatasetManager():

    @staticmethod
    def save_tokenized_data(data_name : str, tokenized_data) -> bool: # tuple of tokenized data list and input shape and dict
        target_folder_path = os.path.join(PROCESSED_DIR, data_name)
        os.makedirs(target_folder_path, exist_ok=True)
        try:
            print(f"Start to process {data_name}")

            # Save .npz file inside subfolder
            np.savez_compressed(os.path.join(target_folder_path, data_name), X=tokenized_data.X, y=tokenized_data.y)

            print(f"Token saved.\nName: {data_name}\nInput Shape: {tokenized_data.input_shape}\nDictionary size: {len(tokenized_data.note_to_int)}")

            # Write shared metadata once
            metadata = {
                "midi_file_names": tokenized_data.midi_file_names,
                "input_shape": tokenized_data.input_shape,
                "note_to_int": tokenized_data.note_to_int
            }

            metadata_path = os.path.join(target_folder_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"Processing of {data_name} successful. Metadata saved at {metadata_path}")
            return True

        except Exception as e:
            print(f"Failed to save tokenized data: {e}")
            shutil.rmtree(target_folder_path)
            return False

    @staticmethod
    def load_tokenized_data(data_name: str):
        target_folder_path = os.path.join(PROCESSED_DIR, data_name)
        target_data_path = os.path.join(target_folder_path, data_name + ".npz")
        target_metadata_path = os.path.join(target_folder_path, "metadata.json")

        if not os.path.exists(target_data_path):
            print(f"File not found. Searched for {target_data_path}")
            return

        # Load X, y, num_classes
        npz_data = np.load(target_data_path)
        X = npz_data["X"]
        y = npz_data["y"]

        if not os.path.exists(target_metadata_path):
            print("Metadata couldn't be found")
            return

        with open(target_metadata_path) as f:
            config = json.load(f)
        data_midi_file_names = config["midi_file_names"]
        data_input_shape = config["input_shape"]
        data_note_to_int = config["note_to_int"]


        print("Tokenized data loaded")

        return (data_midi_file_names, X, y, data_input_shape, data_note_to_int)

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
                data_str_list.append(entry)

        return data_str_list

    @staticmethod
    def does_data_exist(name : str) -> bool:
        data_folder_dir = os.path.join(PROCESSED_DIR, name)
        return os.path.exists(data_folder_dir)

