import glob
import logging
import os

from mido import MidiFile
from music21 import converter, key, stream

from groove_panda import directories
from groove_panda.config import Config, Parser

config = Config()
logger = logging.getLogger(__name__)


def get_midi_paths_from_dataset(dataset_id: str) -> list[str]:
    """
    Get all midi file paths in dataset

    This is used to avoid loading all files at same

    The returned files then can be used to process all songs of the dataset seperatly
    """

    logger.info("Started parsing %s...", dataset_id)

    path = os.path.join(directories.raw_datasets_dir, dataset_id)

    midi_paths = []

    if os.path.isdir(path):
        for extension in config.allowed_music_file_extensions:
            midi_paths.extend(glob.glob(os.path.join(path, f"*{extension}")))
        total = len(midi_paths)
        logger.info("Folder found with %s accepted midi files.", total)
    elif os.path.isfile(path):
        if path.lower().endswith(tuple(config.allowed_music_file_extensions)):
            midi_paths.append(path)
            logger.info("File found")
        else:
            raise Exception("File found, but doesn't have allowed extension.")
    else:
        raise Exception("Invalid path.")

    return midi_paths


def parse_midi(midi_path: str) -> stream.Score | tuple[MidiFile, key.Key]:
    """
    Selects the correct midi parse method according to the set PARSER in configurations.

    Returns a stream.Score from music21 or a MidiFile from mido, depending on the set PARSER.
    """
    if not isinstance(config.parser, Parser):
        raise TypeError("PARSER in configurations must be an instance of the Parser Enum.")

    if config.parser == Parser.MUSIC21:
        return _parse_midi_music21(midi_path)
    if config.parser == Parser.MIDO:
        return _parse_midi_mido_with_key(midi_path)

    raise Exception(f"Parser {config.parser} doesn't exist.")


def _parse_midi_music21(midi_path: str) -> stream.Score:
    """
    Parses midi file to score using music21 converter.

    Assumes correct midi path.

    Returns it, if the parsed result is a Score instance (not Opus or Part).

    Throws exception if parsing wasn't successful.
    """

    if os.path.isfile(midi_path):
        try:
            parsed = converter.parse(midi_path)

            # Optionally we could also transform opus objects into scores
            if isinstance(parsed, stream.Score):
                return parsed
            raise Exception("Parsed music file isn't Score.")
        except Exception as e:
            # Instead of exception, maybe ignore file and print warning
            raise Exception(f"Parsing of {midi_path} failed: {e}") from e
    else:
        raise Exception(f"Invalid path {midi_path}")


def _parse_midi_mido(midi_path: str) -> MidiFile:
    """
    Parses midi file to MidiFile using mido.

    Assumes correct midi path.

    Throws exception if parsing wasn't successful.
    """

    if os.path.isfile(midi_path):
        try:
            return MidiFile(midi_path)
        except Exception as e:
            # Instead of exception, maybe ignore file and print warning
            raise Exception(f"Parsing of {midi_path} failed: {e}") from e
    else:
        raise Exception(f"Invalid path {midi_path}")


def _parse_midi_mido_with_key(midi_path: str) -> tuple[MidiFile, key.Key]:
    """
    Parses midi file with mido and extracts key using music21.
    """
    # Parse with mido
    midi_file = _parse_midi_mido(midi_path)

    # Parse with music21 for key analysis
    try:
        music21_score = _parse_midi_music21(midi_path)
        analyzed_key = music21_score.analyze("key")

        if not isinstance(analyzed_key, key.Key):
            # Fallback to C major if analysis fails
            analyzed_key = key.Key("C", "major")
            logger.warning(f"Key analysis did not return a valid key for {midi_path}, using C major as fallback")

        return midi_file, analyzed_key

    except Exception as e:
        logger.warning(f"Music21 key analysis failed for {midi_path}: {e}, using C major")
        return midi_file, key.Key("C", "major")
