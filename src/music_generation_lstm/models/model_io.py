import json
import logging
import os
import shutil

from tensorflow.keras.models import load_model as load_keras_model

from music_generation_lstm.config import MODELS_DIR
from music_generation_lstm.models.models import BaseModel

logger = logging.getLogger(__name__)


def save_model(model: BaseModel, processed_dataset_id: str):
    model_directory = os.path.join(MODELS_DIR, model.model_id)

    # Make sure directory exists and if not, create it
    os.makedirs(model_directory, exist_ok=True)

    model_path = os.path.join(model_directory, "model.keras")

    logger.info("Saving model %s to %s", model.model_id, model_directory)

    model.model.save(model_path)  # Using model.model since the "Model" type provides a save function

    # Create configuration .json
    config = {
        "name": model.model_id,
        "input shape": model.get_input_shape(),
        "processed_dataset_id": processed_dataset_id,
        # Further data will be saved here in future updates, such as model history,
        # input shape, time steps, features etc.
    }

    # Save configuration .json
    config_filepath = os.path.join(model_directory, "config.json")
    with open(config_filepath, "w") as fp:
        json.dump(config, fp)

    logger.info("Model saved successfully, let the AI takeover BEGIN!!! >:D")


def load_model(name: str) -> BaseModel | None:
    model_dir = os.path.join(MODELS_DIR, name)
    metadata_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.keras")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No config found for model {name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found for model {name}")

    with open(metadata_path) as fp:
        config = json.load(fp)

    input_shape = config["input shape"]
    model = BaseModel(name, input_shape)

    keras_model = load_keras_model(model_path)

    model.set_model(keras_model)

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
        if not (entry == "metadata.json" or entry == ".gitkeep"):
            models_str_list.append(entry)

    return models_str_list
