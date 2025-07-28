from enum import Enum, auto
from typing import Final

""" Enum Defenitions """


class TokenizeMode(Enum):
    """
    The available modes for tokenization.
    """

    ORIGINAL = auto()
    ALL_KEYS = auto()
    C_MAJOR_A_MINOR = auto()


class Parser(Enum):
    """
    The available modes for parsing.
    """

    MUSIC21 = auto()
    MIDO = auto()


""" Hyperparameters """

SEQUENCE_LENGTH: Final[int] = 32  # Important to match processed dataset sequence length to model sequence length!!

GENERATION_LENGTH: Final[int] = 400

TRAINING_EPOCHS: Final[int] = 3

TRAINING_BATCH_SIZE: Final[int] = 64

MODEL_TYPE: Final[str] = "LSTM"


""" Parsing """

"""
The parser variable sets which parser is used during both:
    - processing (parsing of datasets) and
    - generation (parsing of input song/sequence and parsing of generated content)

Since the parser only controls the quality of the parsed information and has no affect on the
tokenization, the program should still function, even when different parsers are used for the
same process-train-generate pipeline (should still be avoided).

CAUTION: In the current implementation, music21 converter is always used to parse the output, since the information
loss, which was the reason why we implemented mido as an alternate parser, only happens during parsing midi from the
datasets, not the when creating new midi files.
"""
PARSER: Final[Parser] = Parser.MUSIC21


""" Tokenization """

"""
Choose how to transpose the data here. Set TOKENIZE_MODE to:
    TokenizeMode.ORIGINAL   - if you want to keep the song's key intact.
    TokenizeMode.ALL_KEYS   - if you want to create copies of the song in all 12 possible keys

TokenizeMode.C_MAJOR_A_MINOR  - if you want all songs to be in C major or A minor
(Cmaj for major songs, Amin for minor songs)
"""
TOKENIZE_MODE: Final[TokenizeMode] = TokenizeMode.ORIGINAL

# for the tokenizer: values smaller than this won't be recognized as tempo changes
TEMPO_TOLERANCE: Final[float] = 0.01

DEFAULT_TEMPO: Final[int] = 120


""" Generation """

# rounds all tempo values
TEMPO_ROUND_VALUE = 10

# Temperature controls randomness in music generation:
# temp = 0   -> deterministic (always picks most likely token)
# temp < 1   -> more conservative/predictable (favors likely tokens)
# temp = 1   -> neutral sampling (uses original probabilities)
# temp > 1   -> more creative/random (gives unlikely tokens more chance)
GENERATION_TEMPERATURE: Final[float] = 0.7

""" Misc """

ALLOWED_MUSIC_FILE_EXTENSIONS: Final[list[str]] = [".mid", ".midi"]

FEATURE_NAMES: Final[list[str]] = ["bar", "position", "pitch", "duration", "velocity", "tempo"]


""" Directories """

DATASETS_MIDI_DIR: Final[str] = "data/midi/datasets"
INPUT_MIDI_DIR: Final[str] = "data/midi/input"
RESULTS_MIDI_DIR: Final[str] = "data/midi/results"
MODELS_DIR: Final[str] = "data/models"
PROCESSED_DIR: Final[str] = "data/processed"
TOKEN_MAPS_DIR: Final[str] = "data/token_maps"
PLOT_DIR: Final[str] = "data/plots"
OUTPUT_SHEET_MUSIC_DIR: Final[str] = "data/sheet_music"
LOG_DIR: Final[str] = "data/logs"


""" Debugging or diagnostics """

PLOT_TRAINING: Final[bool] = True
SAVE_PLOT_TRAINING: Final[bool] = True
CREATE_SHEET_MUSIC: Final[bool] = True


""" Model presets """

# Model architecture presets as a mapping from preset name to its hyperparameter configuration

# Each config dict includes:
#   - sequence_length: int
#   - stride: int
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
        "stride": 1,
        "lstm_units": 64,
        "num_lstm_layers": 1,
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,  # This is the default for ADAM
        "batch_size": 32,
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
        "stride": 1,
        "lstm_units": 128,
        "num_lstm_layers": 2,
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 64,
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
        "stride": 1,
        "lstm_units": 512,
        "num_lstm_layers": 3,
        "dropout_rate": 0.3,
        "learning_rate": 5e-4,
        "batch_size": 64,
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
        "stride": 1,
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
        "batch_size": 1024,
    },
    "test": {  # Terrible but fast architecture. Use just for program testing/debugging.
        "sequence_length": 16,
        "stride": 1,
        "lstm_units": 1,
        "num_lstm_layers": 1,
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,
        "embedding_dims": 1,
        "batch_size": 128,
    },
    # Further presets can be added here
}
