from typing import Final

# Hyperparameters

SEQUENCE_LENGTH: Final = 16

GENERATION_LENGTH: Final = 50

TRAINING_EPOCHS: Final = 1

TRAINING_BATCH_SIZE: Final = 24

ALLOWED_MUSIC_FILE_EXTENSIONS: Final = [".mid", ".midi"]


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
