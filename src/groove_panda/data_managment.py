import logging
import os
import shutil

from groove_panda.config import Config

logger = logging.getLogger(__name__)
config = Config()


def delete_dataset_data(dataset_id: str):
    """
    Deletes a dataset given trough its dataset_id, will delete in data-> midi-> datasets
    deletes the empty dataset folder.
    """
    if dataset_id == "all":
        _delete_all_datasets()
    else:
        dataset_path = os.path.join(config.datasets_midi_dir, dataset_id)
        shutil.rmtree(dataset_path)


def delete_model_data(model_id: str):
    """
    Deletes model given trough its model_id, will delete in data-> models
    deletes the empty dataset folder.
    """
    if model_id == "all":
        _delete_all_models()
    else:
        model_path = os.path.join(config.models_dir, model_id)
        shutil.rmtree(model_path)


def delete_result_data(result_id: str):
    """
    Deletes a result given trough the result ID, will delete in data -> midi -> results
    """
    if result_id == "all":
        _delete_all_results()
    else:
        result_path = os.path.join(config.results_midi_dir, result_id)
        _delete_file(result_path)


def delete_processed_data(processed_id: str):
    """
    Deletes the folder containing token maps and metadata for the given processed_dataset_id.
    And deletes the processed dataset given trough the id
    """
    if processed_id == "all":
        _delete_all_processed()
    else:
        processed_path = os.path.join(config.processed_dir, processed_id)  # processed and map are deleted together
        map_path = os.path.join(config.token_maps_dir, processed_id)  # maybe should be possible to delete only one

        _delete_file(processed_path)
        shutil.rmtree(map_path)


def _delete_all_results():
    _delete_folder_contents(config.results_midi_dir)


def _delete_all_models():
    _delete_folder_contents(config.models_dir)


def _delete_all_datasets():
    _delete_folder_contents(config.datasets_midi_dir)


def _delete_all_processed():
    _delete_folder_contents(config.processed_dir)


def _delete_file(file_path):
    """
    Deletes a file given trough the file ID
    """

    if not os.path.exists(file_path):
        logger.info("file does not exist")
        return
    try:
        os.remove(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to delete file '{file_path}'") from e


def _delete_folder_contents(folder_path):
    """
    Deletes folder with contents(files or more folders)
    deletes not the empty folder
    """
    if not os.path.exists(folder_path):
        logger.info("Path does not exist.")
        return

    if not os.path.isdir(folder_path):
        logger.info(f"{folder_path} is not a directory.")

    for content in os.listdir(folder_path):  # Delete all files
        content_path = os.path.join(folder_path, content)
        if os.path.isfile(content_path):
            os.remove(content_path)

        elif os.path.isdir(content_path):  # folder with folders inside
            shutil.rmtree(content_path)
        else:
            logger.info(f"Could not remove: {content_path}")


def get_existing_result_ids() -> list[str]:
    """
    Iterates through all entries in the respected data folder and returns all names (ids) sorted.
    """
    existing_result_ids = set()
    for result in os.listdir(config.results_midi_dir):
        if result != ".gitkeep":
            existing_result_ids.add(result)
    return sorted(existing_result_ids)


def get_existing_processed_ids() -> list[str]:
    """
    Iterates through all entries in the respected data folder and returns all names (ids) sorted.
    """
    existing_processed_ids = set()
    for processed in os.listdir(config.processed_dir):
        if processed != ".gitkeep":
            existing_processed_ids.add(processed)
    return sorted(existing_processed_ids)


def get_existing_input_ids() -> list[str]:
    """
    Iterates through all entries in the respected data folder and returns all names (ids) sorted.

    Removes extensions from input file names.
    """
    existing_input_ids = set()
    for input in os.listdir(config.input_midi_dir):
        if input != ".gitkeep":
            input_without_extension = os.path.splitext(input)[0]
            existing_input_ids.add(input_without_extension)
    return sorted(existing_input_ids)


def get_existing_dataset_ids() -> list[str]:
    """
    Iterates through all entries in the respected data folder and returns all names (ids) sorted.
    """
    existing_dataset_ids = set()
    for dataset in os.listdir(config.datasets_midi_dir):
        if dataset != ".gitkeep":
            existing_dataset_ids.add(dataset)
    return sorted(existing_dataset_ids)


def get_existing_model_ids() -> list[str]:
    """
    Iterates through all entries in the respected data folder and returns all names (ids) sorted.
    """
    existing_model_ids = set()
    for model in os.listdir(config.models_dir):
        if model != ".gitkeep":
            existing_model_ids.add(model)
    return sorted(existing_model_ids)
