
import os
import shutil
from models.models import BaseModel, ModelFactory
from config import MODELS_DIR
from keras.src.saving.saving_api import save_model as save
from keras.src.saving.saving_api import load_model as load
from keras.src.models import Model
import json
from typing import cast, Optional

class ModelManager:

    @staticmethod
    def save_model(model : BaseModel):
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

    @staticmethod
    def load_model(name : str) -> BaseModel | None:
        model_dir = os.path.join(MODELS_DIR, name)
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "model.keras")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config found for model {name}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found for model {name}")

        with open(config_path) as f:
            config = json.load(f)

        # Reconstruct and load weights
        model_instance = ModelFactory.create_model(
            model_type=config["type"],
            model_name=config["name"],
            input_shape=config["input_shape"]
        )
        #model_instance.history = config["history"]
        #model_instance.note_to_int = config["note_to_int"]

        if model_instance is None:
            raise ValueError("Failed loading model")

        model_instance.model = cast(Optional[Model], load(model_path))

        return model_instance

    @staticmethod
    def delete_model(name : str):
        model_dir = os.path.join(MODELS_DIR, name)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Failed deleting folder {name} at {model_dir}")

        shutil.rmtree(model_dir)

    @staticmethod
    def get_all_models_str_list() -> list[str]:
        models_str_list = []
        os.makedirs(MODELS_DIR, exist_ok=True)
        for entry in os.listdir(MODELS_DIR):
            if not (entry == "metadata.json" or entry == ".gitkeep"):
                models_str_list.append(entry)

        return models_str_list
