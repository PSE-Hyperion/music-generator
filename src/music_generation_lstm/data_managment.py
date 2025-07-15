import os
import logging
from music_generation_lstm.config import DATASETS_MIDI_DIR, RESULTS_MIDI_DIR, TOKEN_MAPS_DIR, MODELS_DIR, PROCESSED_DIR


logger = logging.getLogger(__name__)

existing_result_ids = set()
existing_dataset_ids = set()
existing_processed_ids = set()
existing_model_ids = set()

def delete_dataset_data(dataset_id: str):
    """
    Deletes a dataset given trough its dataset_id, will delete in data-> midi-> datasets 
    deletes the empty dataset folder.
    """
    dataset_path = os.path.join(DATASETS_MIDI_DIR, dataset_id)

    delete_folder_contents(dataset_path)
    delete_empty_folder(dataset_path)

def delete_model_data(model_id: str):
    """
    Deletes model given trough its model_id, will delete in data-> models 
    deletes the empty dataset folder.
    """
    model_path = os.path.join(MODELS_DIR, model_id)

    delete_folder_contents(model_path)
    delete_empty_folder(model_path)
    

def delete_result_data(result_id: str):
    """
    Deletes a result given trough the result ID, will delete in data -> midi -> results
    """
    result_path = os.path.join(RESULTS_MIDI_DIR, result_id)   
    delete_File(result_path)

def delete_processed_data(processed_id: str):
    """
    Deletes the folder containing token maps and metadata for the given processed_dataset_id. 
    And deletes the processed dataset given trough the id
    """
    processed_path = os.path.join(PROCESSED_DIR, processed_id) #processed and map are deleted together
    map_path = os.path.join(TOKEN_MAPS_DIR, processed_id) #maybe should be possible to delete only one

    delete_File(processed_path)
    delete_folder_contents(map_path)
    delete_empty_folder(map_path)

def delete_all_results():
    delete_folder_contents(RESULTS_MIDI_DIR)

def delete_all_models():
    delete_folder_contents(MODELS_DIR)

def delete_all_datasets():
    delete_folder_contents(DATASETS_MIDI_DIR)

def delete_all_processed():
    delete_folder_contents(PROCESSED_DIR)

def delete_File(file_path):
    """
    Deletes a file given trough the file ID
    """
     
    if not os.path.exists(file_path):
        logger.info("file does not exist")
        return
    try:
        os.remove(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to delete file '{file_path}': {e}")

    
def delete_folder_contents(folder_path):
    """
    Deletes folder with contents
    deletes the empty dataset folder.
    """
    if not os.path.exists(folder_path):
        logger.info("Path does not exist.")
        return
    
    if not os.path.isdir(folder_path):
        logger.info(f"{folder_path} is not a directory.")

    
    for file in os.listdir(folder_path): # Delete all files
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            logger.info(f"Could not remove: {file_path}")
    
    

def delete_empty_folder(folder_path):
    os.rmdir(folder_path)# Delete empty folder

 


def add_result_id(result_id: str):
    if result_id not in existing_result_ids:
        existing_result_ids.add(result_id)

def get_existing_result_ids():
    for result in os.listdir(RESULTS_MIDI_DIR): # look at all the files in results, needed in case the programm got closed
        if( result != ".gitkeep"):
            existing_result_ids.add(result)   
    return sorted(existing_result_ids)

def get_existing_processed_ids():
    for processed in os.listdir(PROCESSED_DIR): # neede in case the programm got closed
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

def get_existing_model_ids():
    for model in os.listdir(MODELS_DIR): # neede in case the programm got closed
        if( model != ".gitkeep"):
            existing_model_ids.add(model)   
    return sorted(existing_model_ids)
