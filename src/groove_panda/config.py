from enum import Enum, auto
import json
import logging
import os
import threading
from typing import Final

from groove_panda.utils import overwrite_json

logger = logging.getLogger(__name__)

""" Enum Definitions """


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


class Config:
    # Special config settings - do not touch
    _instance: "Config | None" = None  # Holds the one-and-only instance
    _initialized: bool  # Used to check if the instance has been created
    _lock = threading.Lock()  # Ensures that creation is atomic

    # Default values that are required for efficient initialization. Will be overwritten by loaded configs.
    DEFAULT_NUMBER = 1
    DEFAULT_STR = ""
    DEFAULT_BOOL = False
    EMPTY_LIST = []  # noqa: RUF012

    # Hyperparameters
    sequence_length: int = DEFAULT_NUMBER
    generation_length: int = DEFAULT_NUMBER
    training_epochs: int = DEFAULT_NUMBER
    training_batch_size: int = DEFAULT_NUMBER

    # Further parameters
    early_stopping_epochs_to_wait: int = DEFAULT_NUMBER  # How many epochs with no change to wait until it will stop
    early_stopping_threshold: float = DEFAULT_NUMBER  # Difference that will be considered as no improvement
    """
    Seed for shuffling the songs before they get transformed to a dataset.
    Important for the split into training and validation dataset.
    Seed is useful for random but repeatable split.
    """
    training_validation_split_seed: int = DEFAULT_NUMBER
    dataset_shuffle_seed: int = DEFAULT_NUMBER
    model_init_params_seed: int = DEFAULT_NUMBER
    model_dropout_seed: int = DEFAULT_NUMBER
    """
    Relative share of the validation dataset (between 0 and 1)
    The validation will get executed after every epoch
    """
    validation_split_proportion: float = DEFAULT_NUMBER

    # General settings
    parser: Parser
    allowed_music_file_extensions: list[str] = EMPTY_LIST
    feature_names: list[str] = EMPTY_LIST
    model_type: str = DEFAULT_STR
    config: dict  # The entire config file will be saved here

    # Tokenization settings
    tokenize_mode: TokenizeMode
    tempo_tolerance: float = DEFAULT_NUMBER
    default_tempo: int = DEFAULT_NUMBER

    # Generation settings
    tempo_round_value: int = DEFAULT_NUMBER  # Rounds all tempo values
    generation_temperature: float = DEFAULT_NUMBER

    # Directories
    config_dir: Final[str] = "data/configs"
    config_path: str
    config_name: str
    datasets_midi_dir: str = DEFAULT_STR
    input_midi_dir: str = DEFAULT_STR
    results_midi_dir: str = DEFAULT_STR
    models_dir: str = DEFAULT_STR
    processed_dir: str = DEFAULT_STR
    token_maps_dir: str = DEFAULT_STR
    plot_dir: str = DEFAULT_STR
    output_sheet_music_dir: str = DEFAULT_STR
    log_dir: str = DEFAULT_STR
    result_tokens_dir: str = DEFAULT_STR

    # Debugging and diagnostics settings
    plot_training: bool = DEFAULT_BOOL
    save_plot_training: bool = DEFAULT_BOOL
    create_sheet_music: bool = DEFAULT_BOOL
    save_token_json: bool = DEFAULT_BOOL
    early_stopping_enabled: bool = (
        DEFAULT_BOOL  # Whether the model should stop when the validation loss doesn't improve
    )

    # Model presets
    model_presets: dict[str, dict] = {}  # noqa: RUF012

    logger = logging.getLogger(__name__)

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        """
        This method is called automatically when initialising objects.
        We must overwrite it to make sure only one instance exists.
        """

        """
        With this, we make it so that during multi-threading, when one thread
        starts creating our Config instance, other threads are locked from doing so.
        This is to prevent multiple threads creating the Config simultaneously.
        """
        with cls._lock:
            # Check if singleton has been created
            if cls._instance is None:
                # It hasn't, so we allocate this instance to memory and set this operation as done
                cls._instance = super().__new__(cls)

                """
                Now an instance exists, but has not been initialized
                It is necessary to let __init__ know that it still
                has to do its one-time set up
                """
                cls._instance._initialized = False
        # Always return the one instance
        return cls._instance

    def __init__(self):  # Default string allows for Config() initialization
        """
        Called when initializing a Config object.
        The very first time this occurs, an instance will be initialized.
        All subsequent attempts to create new Config objects will fail, returning the one we already have.
        (Singleton)
        """

        # If instance already exists, don't create a new one.
        if self._initialized:
            return

        # Make sure we remember that the first instance has been initialized.
        self._initialized = True

    def load_config(self, config_name: str):
        config_file_name = config_name + ".json"
        config_path = os.path.join(self.config_dir, config_file_name)

        logger.info(f"Loading config {config_name}...")

        if not os.path.exists(config_path):
            # Unsure whether to use error or to raise an exception.
            self.logger.error(f"No {config_name} config found")

        # Load config file
        with open(config_path) as fp:
            config = json.load(fp)

        # Config loading was successful, save settings
        self.config_name = config_name
        self.config_path = config_path

        # Go through all items in the given config file and set them as the attributes of this config instance
        for key, value in config.items():
            self.change_setting(key, value)

        # No errors occurred, set this config to our current config dict
        self.config = config
        self.logger.info(f"Loaded {config_name} config successfully!")

    def save_config(self, name: str, directory: str = DEFAULT_STR):
        """
        Saves the current settings as a config file in the configs folder with [name].json
        Additionally, a directory can be given as a parameter in case the user wants the config
        to be saved elsewhere other than the default "configs" folder.
        """
        # Update the config dictionary to have the most recent settings set by the user
        for key in self.config:
            current_setting = getattr(self, key)

            if isinstance(current_setting, Enum):
                self.config[key] = current_setting.name
            else:
                self.config[key] = current_setting

        # Create a path
        if directory is not self.DEFAULT_STR:
            config_path = os.path.join(directory, name + ".json")
        else:
            config_path = os.path.join(self.config_dir, name + ".json")

        # Convert the config dictionary to a .json and save it in the configs folder
        overwrite_json(config_path, self.config)

        # Update state to point to the newly created config if we saved to configs
        # Skip this step if we are just saving the config as metadata for something else
        if directory is self.DEFAULT_STR:
            self.config_name = name
            self.config_path = config_path

        self.logger.info(f"Saved {name} successfully as a config!.")

    def overwrite(self):
        """
        This method takes the current config settings and overwrites the file that they are based on.
        After this method is called, the file will reflect the config settings currently active in the program.
        """
        self.save_config(self.config_name, self.config_dir)

    def update(self):
        """
        Updates the settings to match the loaded config file.
        The purpose of this method is to make sure that this config instance
        reflects the current state of the file that has been loaded, including
        any changes the user may have made to it since loading it.
        """
        self.load_config(self.config_name)

    def change_setting(self, setting: str, value):  # noqa: PLR0912
        """
        Changes the given setting's value to be the new value in the current config's settings.
        """

        if setting == "parser":
            try:
                cast = Parser[value]
            except KeyError as e:
                raise ValueError(f"Unknown tokenize_mode '{value}' in {self.config_path}") from e
            setattr(self, setting, cast)

        elif setting == "tokenize_mode":
            try:
                cast = TokenizeMode[value]
            except KeyError as e:
                raise ValueError(f"Unknown tokenize_mode '{value}' in {self.config_path}") from e
            setattr(self, setting, cast)

        else:
            # Every setting other than parser and tokenize mode can be set directly
            if not hasattr(self, setting):
                self.logger.error(f"Unknown config setting: {setting}")

            # Checks the type of the attribute so that the cast is correct
            annotation = self.__class__.__annotations__.get(setting, None)
            if annotation is int:
                setattr(self, setting, int(value))
            elif annotation is float:
                setattr(self, setting, float(value))
            elif annotation is str:
                setattr(self, setting, str(value))
            elif annotation is bool:
                # "Truth" values can sometimes be strings and sometimes booleans. Both cases must be checked
                if isinstance(value, str):
                    setattr(self, setting, (value.lower() == "true"))
                else:
                    setattr(self, setting, value)
            else:
                setattr(self, setting, value)
                logger.info(f"The setting {setting} might be too complicated to change via the terminal")
                logger.info("If it failed, consider changing it in the file itself and using the '-update' command :D")
                return

            self.logger.debug(f"Set setting {setting} to value {value}")
