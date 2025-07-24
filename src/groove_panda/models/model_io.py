import json
import logging
import os
import shutil
from typing import Final

from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.models import load_model as load_keras_model  # type: ignore

from groove_panda.config import MODEL_TYPE, MODELS_DIR
from groove_panda.models.models import BaseModel, LSTMModel
from groove_panda.models.tf_custom.regularizers import (
    NuclearRegularizer,  # noqa: F401 # May be necessary for Keras when loading a model with this regularizer.
)

HISTORY_FILE_NAME: Final = "history.json"
CONFIG_FILE_NAME: Final = "config.json"
MODEL_FILE_NAME: Final = "model.keras"
METADATA_FILE_NAME: Final = "metadata.json"

MODEL_TYPES = {"LSTM": LSTMModel}

logger = logging.getLogger(__name__)


def save_model(model: BaseModel, processed_dataset_id: str):
    model_directory = os.path.join(MODELS_DIR, model.model_id)

    # Make sure directory exists and if not, create it
    os.makedirs(model_directory, exist_ok=True)

    model_path = os.path.join(model_directory, MODEL_FILE_NAME)

    # Save new model, overwriting old one if present.
    _overwrite_saved_model(model, model_path)

    # Create configuration .json
    config = {
        "name": str(model.model_id),
        "input shape": str(model.input_shape),
        "processed_dataset_id": str(processed_dataset_id),
        "epochs trained": str(model.epochs_trained),
        "version": str(model.version),
        # Further data will be saved here in future updates, such as model history,
        # input shape, time steps, features etc.
    }

    # Save configuration .json (overwritting old versions if present)
    config_filepath = os.path.join(model_directory, CONFIG_FILE_NAME)
    _overwrite_json(config_filepath, config)

    # Save history .json (overwritting old versions if present)
    if model.history is not None:
        model_history_dict = model.history.history
        model_history_filepath = os.path.join(model_directory, HISTORY_FILE_NAME)
        _overwrite_json(model_history_filepath, model_history_dict)
    else:
        logger.info("Model has no history to save.")

    logger.info("Model saved successfully.")


def load_model(name: str) -> tuple[BaseModel, dict[str, str]]:
    model_dir = os.path.join(MODELS_DIR, name)
    config_path = os.path.join(model_dir, CONFIG_FILE_NAME)
    history_path = os.path.join(model_dir, HISTORY_FILE_NAME)
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found for model {name}")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"No history file found for model {name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found for model {name}")

    with open(config_path) as fp:
        config = json.load(fp)
    with open(history_path) as fp:
        history_dict = json.load(fp)

    input_shape = config["input shape"]
    model = MODEL_TYPES[MODEL_TYPE](name, input_shape)

    keras_model = load_keras_model(model_path)

    model.set_model(keras_model)

    # Set the model's history to be the one of the previous session, including epochs.
    history = History()
    history.history = history_dict
    model.setup(
        history=history,
        version=int(config.get("version", 1)),  # To support older models, default values exist
        epochs_trained=int(config.get("epochs trained", 0)),
    )

    return model, config


def delete_model(name: str):
    model_dir = os.path.join(MODELS_DIR, name)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Failed deleting folder {name} at {model_dir}")

    shutil.rmtree(model_dir)


def get_all_models_str_list() -> list[str]:
    models_str_list = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    for entry in os.listdir(MODELS_DIR):
        if entry not in {METADATA_FILE_NAME, ".gitkeep"}:
            models_str_list.append(entry)

    return models_str_list


def _overwrite_saved_model(model: BaseModel, model_path: str):
    """
    Checks if an older version of the model already exists at that location.
    If so, deletes it and saves the new one. If not, it simply saves the model.
    """
    if os.path.exists(model_path):
        # Delete the .keras file
        os.remove(model_path)
    model.model.save(model_path)  # Using model.model since the "Model" type provides a save function


def _overwrite_json(file_path: str, data: dict):
    """
    Checks if outdated versions of the data to be saved exists and overwrites them if it's the case.
    If no data is present, the provided data is simply saved.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as fp:
        json.dump(data, fp, indent=2)  # Indent = 2 for readability


def get_model_path(model_id: str) -> str:
    return os.path.join(MODELS_DIR, model_id)
