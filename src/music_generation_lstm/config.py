from typing import Final

# Hyperparameters

SEQUENCE_LENGTH: Final = 8  # Changed from 8/32

DEFAULT_GENERATION_LENGTH: Final = 50

TRAINING_EPOCHS: Final = 1

TRAINING_BATCH_SIZE: Final = 12

ALLOWED_MUSIC_FILE_EXTENSIONS: Final = [".mid", ".midi"]

FEATURE_NAMES = ["bar", "position", "pitch", "duration", "velocity", "tempo"]

NUMBER_OF_FEATURES = 6

TEMPERATURE = 0.5

TRAINING_ARCHITECTURE = "BASIC"  # Options are, BASIC and ADVANCED

LEARNING_RATE = 0.0003  # Default for Adam is 0.001

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
