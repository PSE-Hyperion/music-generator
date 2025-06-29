from typing import Final

# contains hyperparameters like sequence length

SEQUENCE_LENGTH: Final = 100

GENERATION_LENGTH: Final = 50

QUANTIZATION_PRECISION_DELTA_OFFSET: Final = 1/8

QUANTIZATION_PRECISION_DURATION: Final = 1/8

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
