import json
import os
import shutil

from music_generation_lstm.config import MODELS_DIR
from music_generation_lstm.models.models import BaseModel


def save_model(model: BaseModel):
    model_directory = model.model_id + ".keras"
    model_path = os.path.join(MODELS_DIR, model_directory)

    # Make sure directory exists and if not, create it
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Saving model {model_directory} to {model_path}")

    model.model.save(model_path)  # Using model.model since the "Model" type provides a save function

    # Create configuration .json
    config = {
        "name": model.model_id,
        # Further data will be saved here in future updates, such as model history,
        # input shape, time steps, features etc.
    }

    # Save configuration .json
    config_filepath = os.path.join(MODELS_DIR, "config.json")
    with open(config_filepath, "w") as fp:
        json.dump(config, fp)

    print("Model saved successfully, let the AI takeover BEGIN!!! >:D")


def load_model(name: str) -> BaseModel | None:
    model_dir = os.path.join(MODELS_DIR, name)
    metadata_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.keras")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No config found for model {name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found for model {name}")

    """
    # load configs for model
    with open(metadata_path) as f:
        config = json.load(f)
    """

    # rebuild model
    model = None

    return model


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
