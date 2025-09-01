"""
Contains all the static directories and the paths for our program.
Excluded are directories and paths that change frequently or during runtime.
"""

import os
from typing import Final

BASE_DIR: Final[str] = "data"

# Config
CONFIG_NAME: Final[str] = "config_julien"

CONFIG_DIR: Final[str] = os.path.join(BASE_DIR, "configs")

# Directory to save the models
MODELS_DIR: Final[str] = os.path.join(BASE_DIR, "models")

# Directories to save everything related to song generation
GENERATION_DIR: Final[str] = os.path.join(BASE_DIR, "generation")
INPUT_DIR: Final[str] = os.path.join(GENERATION_DIR, "input")
OUTPUT_DIR: Final[str] = os.path.join(GENERATION_DIR, "output")
RESULT_TOKEN_DIR: Final[str] = os.path.join(BASE_DIR, "tokens/results")

# Directories for datasets
DATASET_DIR: Final[str] = os.path.join(BASE_DIR, "datasets")
RAW_DATASET_DIR: Final[str] = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATASET_DIR: Final[str] = os.path.join(DATASET_DIR, "processed")
TOKEN_MAPS_DIR: Final[str] = os.path.join(DATASET_DIR, "token_maps")

# Others
LOG_DIR: Final[str] = os.path.join(BASE_DIR, "logs")
