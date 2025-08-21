"""
Contains all the static directories and the paths for our program.
Excluded are directories and paths that change frequently or during runtime.
"""

import os
from typing import Final

base_dir: Final[str] = "data"

# Config
config_name: Final[str] = "config"
config_dir: Final[str] = os.path.join(base_dir, "configs")


# Directory to save the models
models_dir: Final[str] = os.path.join(base_dir, "models")

# Directories to save everything related to song generation
generation_dir: Final[str] = os.path.join(base_dir, "generation")
input_dir: Final[str] = os.path.join(generation_dir, "input")
output_dir: Final[str] = os.path.join(generation_dir, "output")
result_tokens_dir: Final[str] = os.path.join(base_dir, "tokens/results")

# Directories for datasets
datasets_dir: Final[str] = os.path.join(base_dir, "datasets")
raw_datasets_dir: Final[str] = os.path.join(datasets_dir, "raw")
processed_datasets_dir: Final[str] = os.path.join(datasets_dir, "processed")
token_maps_dir: Final[str] = os.path.join(datasets_dir, "token_maps")

# Others
log_dir: Final[str] = os.path.join(base_dir, "logs")  # Should be added to models
