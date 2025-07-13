import os
from music_generation_lstm.config import DATASETS_MIDI_DIR, RESULTS_MIDI_DIR, INPUT_MIDI_DIR, TOKEN_MAPS_DIR


existing_result_ids = set()
existing_dataset_ids = set()
existing_processed_ids = set()

def delete_dataset_data(dataset_id: str):
    """
    Deletes a dataset given trough its dataset_id, will delete in data-> midi-> datasets 
    deletes the empty dataset folder.
    """
    dataset_path = os.path.join(DATASETS_MIDI_DIR, dataset_id)
    if not os.path.exists(dataset_path):
        print("Path does not exist.")
        return
    
    if not os.path.isdir(dataset_path):
        print(f"{dataset_path} is not a directory.")

    
    for file in os.listdir(dataset_path): # Delete all files
        file_path = os.path.join(dataset_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print(f"Could not remove: {file_path}")
    
    os.rmdir(dataset_path)# Delete empty folder
    

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
    

def delete_existing_processed(processed_id: str):
    """
    Deletes the folder containing token maps and metadata for the given processed_dataset_id.
    The folder path is TOKEN_MAPS_DIR/processed_dataset_id. And deletes the processed dataset given trough the id
    """
    processed_path = os.path.join(INPUT_MIDI_DIR, processed_id)
    map_path = os.path.join(TOKEN_MAPS_DIR, processed_id)

    if not os.path.exists(map_path):
        print(f"{map_path} does not exist.")
        return
    if not os.path.exists(processed_path):
        print(f"{processed_path} does not exist.")
        return
    if not os.path.isdir(map_path):
        print(f"{map_path} is not a directory.")
    
    os.remove(processed_path) # delete processed

    for file in os.listdir(map_path): # Delete all files from map
        file_path = os.path.join(map_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print(f"Could not remove: {file_path}")
    os.rmdir(map_path)# Delete empty folder



def add_result_id(result_id: str):
    if result_id not in existing_result_ids:
        existing_result_ids.add(result_id)

def get_existing_result_ids():
    for result in os.listdir(RESULTS_MIDI_DIR): # look at all the files in results, needed in case the programm got closed
        if( result != ".gitkeep"):
            existing_result_ids.add(result)   
    return sorted(existing_result_ids)

def get_existing_processed_ids():
    for processed in os.listdir(INPUT_MIDI_DIR): # neede in case the programm got closed
        if( processed != ".gitkeep"):
            existing_dataset_ids.add(processed)  
    return sorted(existing_dataset_ids)

def add_dataset_id(dataset_id: str):
    if dataset_id not in existing_dataset_ids:
        existing_dataset_ids.add(dataset_id)

def get_existing_dataset_ids():
    for dataset in os.listdir(DATASETS_MIDI_DIR): # neede in case the programm got closed
        if( dataset != ".gitkeep"):
            existing_dataset_ids.add(dataset)   
    return sorted(existing_dataset_ids)