import logging

from groove_panda.data_managment import (
    delete_all_datasets,
    delete_all_models,
    delete_all_processed,
    delete_all_results,
    delete_dataset_data,
    delete_model_data,
    delete_processed_data,
    delete_result_data,
)
from groove_panda.generation.generate_music import generate_music
from groove_panda.models.model_training import train_model
from groove_panda.processing import parallel_processing

logger = logging.getLogger(__name__)


def process(dataset_id: str, processed_dataset_id: str):
    """
    parses midi file(s) to music21.stream.Score
    tokenize score(s)
    numerize tokens
    save processed data (ready for training data)
    """

    parallel_processing.parallel_process(dataset_id, processed_dataset_id)


def train(model_id: str, processed_dataset_id: str, preset_name: str):
    """
    train a model using the processed dataset
    """
    train_model(model_id, processed_dataset_id, preset_name)

def delete_dataset(dataset_id: str):
    """Deletes a dataset given trough its dataset_id, will delete in data-> midi-> datasets"""
    if dataset_id == "all":
        delete_all_datasets()
    else:
        delete_dataset_data(dataset_id)


def delete_result(result_id: str):
    """Deletes a file given trough the result_id, will delete in data -> midi -> results"""
    if result_id == "all":
        delete_all_results()
    else:
        delete_result_data(result_id)


def delete_processed(processed_id: str):
    """Deletes a processed given trough the processed_id, will delete in data -> processed"""
    if processed_id == "all":
        delete_all_processed()
    else:
        delete_processed_data(processed_id)


def delete_model(model_id: str):
    """Deletes a model given trough the model_id, will delete in data -> models"""
    if model_id == "all":
        delete_all_models()
    else:
        delete_model_data(model_id)


def generate(model_name: str, input_name: str, output_name: str):
    """
    Generate music using a trained model
    """
    generate_music(model_name, input_name, output_name)


def show():
    """
    get model via label
    get midi
    get start sequence from midi
    generate with model using start sequence
    write result in folder
    """

    logger.info("show")


def exit():
    logger.info("You've exited the program.")
