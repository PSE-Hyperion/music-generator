import os
import shutil #maybe not neccesary, but usefull for removing directories
from music_generation_lstm.config import DATASETS_MIDI_DIR, RESULTS_MIDI_DIR

def delete_dataset_data(dataset_id: str):
    """
    Deletes a dataset given trough its dataset_id, will delete in data-> midi-> datasets 
    deletes the empty dataset folder.
    """
    dataset_path = os.path.join(DATASETS_MIDI_DIR, dataset_id)
    if not os.path.isdir(dataset_path): #if folder does not exist
        print("folder does not exist")
        return

    try:
        shutil.rmtree(dataset_path)
    except Exception as e:
        raise RuntimeError(f"Failed to delete dataset folder {dataset_path}: {e}")


def delete_result_data(result_id: str):
    """
    Deletes a file given trough the file ID, will delete in data -> midi -> results
    """

    file_path = os.path.join(RESULTS_MIDI_DIR, result_id)
     
    if not os.path.exists(file_path):
        print("file does not exist")
        return
    try:
        os.remove(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to delete file '{file_path}': {e}")