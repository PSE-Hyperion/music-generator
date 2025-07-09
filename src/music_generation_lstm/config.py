import re
from typing import Final

# Hyperparameters

SEQUENCE_LENGTH: Final = 8

GENERATION_LENGTH: Final = 50

TRAINING_EPOCHS: Final = 2

TRAINING_BATCH_SIZE: Final = 12

ALLOWED_MUSIC_FILE_EXTENSIONS: Final = [".mid", ".midi"]

MIDI_FILE_PATTERN: Final = re.compile(r".*\.(mid|midi)$")

# Paths

DATASETS_MIDI_DIR: Final = "data/midi/datasets"
INPUT_MIDI_DIR: Final = "data/midi/input"
RESULTS_MIDI_DIR: Final = "data/midi/results"
MODELS_DIR: Final = "data/models"
PROCESSED_DIR: Final = "data/processed"
TOKEN_MAPS_DIR: Final = "data/token_maps"
PLOT_DIR: Final = "data/plots"

# Debugging or diagnostics

PLOT_TRAINING: Final = True
SAVE_PLOT_TRAINING: Final = True
