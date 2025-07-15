from enum import Enum
from enum import Enum
from typing import Final


class TokenizeMode(Enum):
    ORIGINAL = 1
    ALL_KEYS = 2
    C_MAJOR_A_MINOR = 3


# Hyperparameters

SEQUENCE_LENGTH: Final = 8

GENERATION_LENGTH: Final = 1000

TRAINING_EPOCHS: Final = 50

TRAINING_BATCH_SIZE: Final = 12

# temp= 0 -> immer das was predicted wird
# temp > 1 -> mehr random und unwahrscheinlicherere noten
# temp < 1 -> wahrscheinliche noten aber trotzdem bissl random
GENERATION_TEMPERATURE: Final = 0.7

ALLOWED_MUSIC_FILE_EXTENSIONS: Final = [".mid", ".midi"]


# Paths

DATASETS_MIDI_DIR: Final = "data/midi/datasets"
INPUT_MIDI_DIR: Final = "data/midi/input"
RESULTS_MIDI_DIR: Final = "data/midi/results"
MODELS_DIR: Final = "data/models"
PROCESSED_DIR: Final = "data/processed"
TOKEN_MAPS_DIR: Final = "data/token_maps"
PLOT_DIR: Final = "data/plots"
OUTPUT_SHEET_MUSIC_DIR: Final = "data/detokenized_sheet_music"

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
