
import os
import shutil
from .models import BaseModel
from ..config import MODELS_DIR
import json
#from typing import cast, Optional      #not needed anymore



def save_model(model : BaseModel):
    """
    model_dir = os.path.join(MODELS_DIR, model.name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.keras")
    save(model.model, model_path)

    config = {
        "name": model.name,
        "type": model.TYPE
        #"history": model_instance.history,
        #"input_shape": model_instance.input_shape,
        #"timesteps": model_instance.input_shape[0],
        #"features": model_instance.input_shape[1],
        #"note_to_int": model_instance.note_to_int
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    """

def load_model(name : str) -> BaseModel | None:
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


def delete_model(name : str):
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
