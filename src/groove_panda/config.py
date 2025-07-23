from enum import Enum
from typing import Final


class TokenizeMode(Enum):
    ORIGINAL = 1
    ALL_KEYS = 2
    C_MAJOR_A_MINOR = 3


# Hyperparameters

SEQUENCE_LENGTH: Final = 16  # Important to match processed dataset sequence length to model sequence length!!

GENERATION_LENGTH: Final = 500

TRAINING_EPOCHS: Final = 2

TRAINING_BATCH_SIZE: Final = 64

# for the tokenizer: values smaller than this won't be recognized as tempo changes
TEMPO_TOLERANCE: Final = 0.01

DEFAULT_TEMPO: Final = 120

# rounds all tempo values
TEMPO_ROUND_VALUE = 10

# Temperature controls randomness in music generation:
# temp = 0   -> deterministic (always picks most likely token)
# temp < 1   -> more conservative/predictable (favors likely tokens)
# temp = 1   -> neutral sampling (uses original probabilities)
# temp > 1   -> more creative/random (gives unlikely tokens more chance)
GENERATION_TEMPERATURE: Final = 0.7

ALLOWED_MUSIC_FILE_EXTENSIONS: Final = [".mid", ".midi"]

FEATURE_NAMES: Final = ["bar", "position", "pitch", "duration", "velocity", "tempo"]
# Paths

DATASETS_MIDI_DIR: Final = "data/midi/datasets"
INPUT_MIDI_DIR: Final = "data/midi/input"
RESULTS_MIDI_DIR: Final = "data/midi/results"
MODELS_DIR: Final = "data/models"
PROCESSED_DIR: Final = "data/processed"
TOKEN_MAPS_DIR: Final = "data/token_maps"
PLOT_DIR: Final = "data/plots"
OUTPUT_SHEET_MUSIC_DIR: Final = "data/sheet_music"
LOG_DIR: Final = "data/logs"

# Debugging or diagnostics

PLOT_TRAINING: Final = True
SAVE_PLOT_TRAINING: Final = True

# Optional Settings

CREATE_SHEET_MUSIC: Final = False

# choose how to transpose the data here. Set TOKENIZE_MODE to:
# TokenizeMode.ORIGINAL   - if you want to keep the song's key intact.
# TokenizeMode.ALL_KEYS   - if you want to create copies of the song in all 12 possible keys

# TokenizeMode.C_MAJOR_A_MINOR  - if you want all songs to be in C major or A minor
# (Cmaj for major songs, Amin for minor songs)
TOKENIZE_MODE = TokenizeMode.ORIGINAL

# Model architecture presets as a mapping from preset name to its hyperparameter configuration

# Each config dict includes:
#   - sequence_length: int
#   - lstm_units: int
#   - num_lstm_layers: int
#   - dropout_rate: float
#   - learning_rate: float
#   - batch_size: int
#   - epochs: int
#   - embedding_dims: dict[str, int]  # -> dict[feature name, embedding size]

MODEL_PRESETS = {
    "light": {
        "sequence_length": 16,
        "lstm_units": 64,
        "num_lstm_layers": 1,
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,  # This is the default for ADAM
        "embedding_dims": {
            "pitch": 16,
            "duration": 8,
            "velocity": 8,
            "position": 8,
            "bar": 4,
            "tempo": 8,
        },
    },
    "basic": {
        "sequence_length": 32,
        "lstm_units": 128,
        "num_lstm_layers": 2,
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
        "embedding_dims": {
            "pitch": 32,
            "duration": 16,
            "velocity": 16,
            "position": 16,
            "bar": 8,
            "tempo": 16,
        },
    },
    "advanced": {
        "sequence_length": 64,
        "lstm_units": 512,
        "num_lstm_layers": 3,
        "dropout_rate": 0.3,
        "learning_rate": 5e-4,
        "embedding_dims": {
            "pitch": 128,
            "duration": 64,
            "velocity": 64,
            "position": 64,
            "bar": 32,
            "tempo": 64,
        },
    },
    "lightplus": {  # Takes about 1:42 per epoch on kpop16
        "sequence_length": 16,
        "lstm_units": 64,
        "num_lstm_layers": 2,
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,
        "embedding_dims": {
            "pitch": 16,
            "duration": 8,
            "velocity": 8,
            "position": 8,
            "bar": 4,
            "tempo": 8,
        },
    },
    "lightadjust": {
        "sequence_length": 32,
        "lstm_units": 128,
        "num_lstm_layers": 3,
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
        "embedding_dims": {
            "pitch": 64,
            "duration": 8,
            "velocity": 8,
            "position": 8,
            "bar": 2,
            "tempo": 8,
        },
    },
    "test": {  # Terrible but fast architecture. Use just for program testing/debugging.
        "sequence_length": 16,
        "lstm_units": 1,
        "num_lstm_layers": 1,
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,
        "embedding_dims": 1,
    },
    # Further presets can be added here
}
